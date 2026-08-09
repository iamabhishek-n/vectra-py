import json
import pytest
from unittest.mock import AsyncMock
from vectra.memory.fact_store import FactStore


class FakeConn:
    def __init__(self, existing_facts=None):
        self.existing_facts = existing_facts or []
        self.inserted = []
        self.invalidated = []
        self.fetchrow = AsyncMock(side_effect=self._fetchrow)
        self.execute = AsyncMock(side_effect=self._execute)

    async def _fetchrow(self, q, *args):
        if "SELECT" in q and "invalid_at" in q and "IS NULL" in q:
            session_id, subject, predicate = args
            for f in self.existing_facts:
                if f["subject"] == subject and f["predicate"] == predicate:
                    return {"id": f["id"], "object": f["object"]}
        return None

    async def _execute(self, q, *args):
        if "UPDATE" in q and "invalid_at" in q:
            self.invalidated.append(args[0])
        elif "INSERT INTO" in q:
            self.inserted.append(args)
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeClient:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return self._conn


def make_store(existing_facts, llm_response):
    conn = FakeConn(existing_facts)
    llm = type("LLM", (), {"generate": AsyncMock(return_value=llm_response)})()
    embedder = type("Embedder", (), {"embed_documents": AsyncMock(side_effect=lambda texts: [[0.1, 0.2] for _ in texts])})()
    config = type("Cfg", (), {
        "client_instance": FakeClient(conn), "table_name": "VectraFact",
        "llm": llm, "embedder": embedder,
    })()
    return FactStore(config), conn


class TestFactStoreInvalidation:
    async def test_invalidates_old_fact_and_inserts_new_when_object_changes(self):
        store, conn = make_store(
            [{"id": "old-1", "subject": "user", "predicate": "lives_in", "object": "Berlin"}],
            json.dumps({"facts": [{"subject": "user", "predicate": "lives_in", "object": "Tokyo"}]}),
        )

        await store.write("session-1", {"user_message": "I moved to Tokyo", "assistant_message": "Nice!"})

        assert "old-1" in conn.invalidated
        assert len(conn.inserted) == 1
        assert conn.inserted[0][4] == "Tokyo"

    async def test_no_duplicate_when_identical_fact_already_valid(self):
        store, conn = make_store(
            [{"id": "old-1", "subject": "user", "predicate": "likes", "object": "coffee"}],
            json.dumps({"facts": [{"subject": "user", "predicate": "likes", "object": "coffee"}]}),
        )

        await store.write("session-1", {"user_message": "I still like coffee", "assistant_message": "Great!"})

        assert len(conn.invalidated) == 0
        assert len(conn.inserted) == 0

    async def test_inserts_as_new_when_no_existing_subject_predicate(self):
        store, conn = make_store([], json.dumps({"facts": [{"subject": "user", "predicate": "likes", "object": "tea"}]}))

        await store.write("session-1", {"user_message": "I like tea", "assistant_message": "Noted!"})

        assert len(conn.invalidated) == 0
        assert len(conn.inserted) == 1
