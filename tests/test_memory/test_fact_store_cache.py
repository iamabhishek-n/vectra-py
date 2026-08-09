import asyncio
import json
import pytest
from unittest.mock import AsyncMock
from vectra.memory.fact_store import FactStore


class FakeConn:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.fetch = AsyncMock(return_value=self.rows)
        self.fetchrow = AsyncMock(return_value=None)
        self.execute = AsyncMock(return_value=None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeClient:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return self._conn


def make_store(cache_ttl_ms=30000, rows=None, llm_response=None):
    conn = FakeConn(rows)
    embedder = type("Embedder", (), {
        "embed_query": AsyncMock(return_value=[0.1]),
        "embed_documents": AsyncMock(side_effect=lambda texts: [[0.1] for _ in texts]),
    })()
    llm = type("LLM", (), {"generate": AsyncMock(return_value=llm_response or json.dumps({"facts": []}))})()
    config = type("Cfg", (), {
        "client_instance": FakeClient(conn), "table_name": "VectraFact",
        "embedder": embedder, "llm": llm, "cache_ttl_ms": cache_ttl_ms,
    })()
    return FactStore(config), conn


class TestFactStoreCache:
    async def test_does_not_requery_within_ttl(self):
        store, conn = make_store(cache_ttl_ms=30000, rows=[{"id": "1", "subject": "a", "predicate": "b", "object": "c"}])

        await store.read("session-1", "same query")
        await store.read("session-1", "same query")

        conn.fetch.assert_called_once()

    async def test_requeries_for_different_query_text(self):
        store, conn = make_store(cache_ttl_ms=30000)

        await store.read("session-1", "query A")
        await store.read("session-1", "query B")

        assert conn.fetch.call_count == 2

    async def test_requeries_after_expiry(self):
        store, conn = make_store(cache_ttl_ms=1)

        await store.read("session-1", "query")
        await asyncio.sleep(0.01)
        await store.read("session-1", "query")

        assert conn.fetch.call_count == 2

    async def test_write_invalidates_the_session_cache(self):
        store, conn = make_store(
            cache_ttl_ms=30000,
            rows=[{"id": "1", "subject": "user", "predicate": "lives_in", "object": "Berlin"}],
            llm_response=json.dumps({"facts": [{"subject": "user", "predicate": "lives_in", "object": "Tokyo"}]}),
        )

        await store.read("session-1", "where does the user live")  # populate cache
        await store.write("session-1", {"user_message": "I moved", "assistant_message": "Cool!"})
        await store.read("session-1", "where does the user live")  # must not hit stale cache

        assert conn.fetch.call_count == 2
