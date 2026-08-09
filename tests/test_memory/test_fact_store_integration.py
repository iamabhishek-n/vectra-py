import json
import uuid
from unittest.mock import AsyncMock
from vectra.memory.fact_store import FactStore


class InMemoryFakeConn:
    """Actually stores/updates/filters rows, unlike the narrower per-task mocks,
    to catch integration bugs the unit tests' shape-specific mocks could miss."""
    def __init__(self):
        self.rows = []

    async def execute(self, q, *args):
        if "UPDATE" in q and "invalid_at" in q:
            fact_id = args[0]
            for r in self.rows:
                if r["id"] == fact_id:
                    r["invalid_at"] = "now"
        elif "INSERT INTO" in q:
            fact_id, session_id, subject, predicate, obj = args[0], args[1], args[2], args[3], args[4]
            self.rows.append({"id": fact_id, "session_id": session_id, "subject": subject, "predicate": predicate, "object": obj, "valid_at": "then", "invalid_at": None})
        return None

    async def fetchrow(self, q, *args):
        session_id, subject, predicate = args
        for r in self.rows:
            if r["session_id"] == session_id and r["subject"] == subject and r["predicate"] == predicate and not r["invalid_at"]:
                return {"id": r["id"], "object": r["object"]}
        return None

    async def fetch(self, q, *args):
        session_id = args[0]
        return [
            {"id": r["id"], "subject": r["subject"], "predicate": r["predicate"], "object": r["object"], "valid_at": r["valid_at"], "invalid_at": r["invalid_at"]}
            for r in self.rows if r["session_id"] == session_id and not r["invalid_at"]
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeClient:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return self._conn


def make_store(conn, llm_responses):
    responses = iter(llm_responses)
    llm = type("LLM", (), {"generate": AsyncMock(side_effect=lambda *a, **k: next(responses))})()
    embedder = type("Embedder", (), {
        "embed_documents": AsyncMock(side_effect=lambda texts: [[0.1] for _ in texts]),
        "embed_query": AsyncMock(return_value=[0.1]),
    })()
    config = type("Cfg", (), {"client_instance": FakeClient(conn), "table_name": "VectraFact", "llm": llm, "embedder": embedder})()
    return FactStore(config)


class TestFactStoreEndToEnd:
    async def test_write_then_read_round_trips_a_fact(self):
        conn = InMemoryFakeConn()
        store = make_store(conn, [json.dumps({"facts": [{"subject": "user", "predicate": "likes", "object": "coffee"}]})])

        await store.write("session-1", {"user_message": "I like coffee", "assistant_message": "Noted!"})
        facts = await store.read("session-1", "what does the user like")

        assert len(facts) == 1
        assert facts[0]["object"] == "coffee"

    async def test_superseding_fact_replaces_old_one_never_both(self):
        conn = InMemoryFakeConn()
        store = make_store(conn, [
            json.dumps({"facts": [{"subject": "user", "predicate": "lives_in", "object": "Berlin"}]}),
            json.dumps({"facts": [{"subject": "user", "predicate": "lives_in", "object": "Tokyo"}]}),
        ])

        await store.write("session-1", {"user_message": "I live in Berlin", "assistant_message": "Cool!"})
        await store.write("session-1", {"user_message": "I moved to Tokyo", "assistant_message": "Wow!"})
        facts = await store.read("session-1", "where does the user live")

        lives_in = [f for f in facts if f["predicate"] == "lives_in"]
        assert len(lives_in) == 1
        assert lives_in[0]["object"] == "Tokyo"

    async def test_facts_never_leak_across_sessions(self):
        conn = InMemoryFakeConn()
        store = make_store(conn, [json.dumps({"facts": [{"subject": "user", "predicate": "likes", "object": "coffee"}]})])

        await store.write("session-A", {"user_message": "I like coffee", "assistant_message": "Noted!"})
        facts_b = await store.read("session-B", "what does the user like")

        assert len(facts_b) == 0

    async def test_a_read_cached_before_a_write_is_not_served_stale_after_the_write_changes_that_sessions_facts(self):
        conn = InMemoryFakeConn()
        store = make_store(conn, [
            json.dumps({"facts": [{"subject": "user", "predicate": "lives_in", "object": "Berlin"}]}),
            json.dumps({"facts": [{"subject": "user", "predicate": "lives_in", "object": "Tokyo"}]}),
        ])

        await store.write("session-1", {"user_message": "I live in Berlin", "assistant_message": "Cool!"})
        first_read = await store.read("session-1", "where does the user live")
        assert next(f for f in first_read if f["predicate"] == "lives_in")["object"] == "Berlin"

        await store.write("session-1", {"user_message": "I moved to Tokyo", "assistant_message": "Wow!"})
        second_read = await store.read("session-1", "where does the user live")  # same query string

        assert next(f for f in second_read if f["predicate"] == "lives_in")["object"] == "Tokyo"
