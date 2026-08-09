import json
import pytest
from unittest.mock import AsyncMock
from vectra.memory.fact_store import FactStore


class FakeConn:
    def __init__(self):
        self.inserted = []
        self.execute = AsyncMock(side_effect=self._record)
        self.fetchrow = AsyncMock(return_value=None)  # no existing facts, always inserts fresh

    async def _record(self, q, *args):
        if "INSERT INTO" in q:
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


def make_store(llm_response, embed_response=None):
    conn = FakeConn()
    llm = type("LLM", (), {"generate": AsyncMock(return_value=llm_response)})()
    embedder = type("Embedder", (), {"embed_documents": AsyncMock(side_effect=lambda texts: [embed_response or [0.1, 0.2] for _ in texts])})()
    config = type("Cfg", (), {
        "client_instance": FakeClient(conn), "table_name": "VectraFact",
        "llm": llm, "embedder": embedder,
    })()
    return FactStore(config), conn, llm, embedder


class TestFactStoreWrite:
    async def test_extracts_triples_and_inserts_as_new_facts(self):
        store, conn, llm, embedder = make_store(
            json.dumps({"facts": [{"subject": "user", "predicate": "likes", "object": "coffee"}]})
        )

        await store.write("session-1", {"user_message": "I really like coffee", "assistant_message": "Noted!"})

        llm.generate.assert_called_once()
        embedder.embed_documents.assert_called_once()
        assert len(conn.inserted) == 1
        args = conn.inserted[0]
        assert args[1] == "session-1"
        assert args[2] == "user"
        assert args[3] == "likes"
        assert args[4] == "coffee"

    async def test_extracts_triples_when_llm_wraps_json_in_a_markdown_code_fence(self):
        store, conn, _, _ = make_store(
            "```json\n" + json.dumps({"facts": [{"subject": "user", "predicate": "likes", "object": "tea"}]}) + "\n```"
        )

        await store.write("session-1", {"user_message": "I like tea", "assistant_message": "Noted!"})

        assert len(conn.inserted) == 1
        assert conn.inserted[0][4] == "tea"

    async def test_extracts_multiple_triples_from_one_turn(self):
        store, conn, _, _ = make_store(
            json.dumps({"facts": [
                {"subject": "user", "predicate": "likes", "object": "coffee"},
                {"subject": "user", "predicate": "lives_in", "object": "Berlin"},
            ]})
        )

        await store.write("session-1", {"user_message": "I like coffee and live in Berlin", "assistant_message": "Cool!"})

        assert len(conn.inserted) == 2

    async def test_does_not_raise_on_malformed_json(self):
        store, conn, _, _ = make_store("not json at all {{{")

        await store.write("session-1", {"user_message": "hi", "assistant_message": "hello"})

        assert len(conn.inserted) == 0

    async def test_does_not_raise_on_empty_facts_array(self):
        store, conn, _, _ = make_store(json.dumps({"facts": []}))

        await store.write("session-1", {"user_message": "hi", "assistant_message": "hello"})

        assert len(conn.inserted) == 0

    async def test_no_op_when_session_id_missing(self):
        store, conn, llm, _ = make_store(json.dumps({"facts": []}))

        await store.write(None, {"user_message": "hi", "assistant_message": "hello"})

        llm.generate.assert_not_called()
        assert len(conn.inserted) == 0
