import pytest
from unittest.mock import AsyncMock
from vectra.memory.fact_store import FactStore


class FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.last_query = None
        self.last_args = None
        self.fetch = AsyncMock(side_effect=self._fetch)

    async def _fetch(self, q, *args):
        self.last_query = q
        self.last_args = args
        return self.rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeClient:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return self._conn


class TestFactStoreRead:
    async def test_single_query_combining_vector_and_temporal_validity(self):
        rows = [{"id": "1", "subject": "user", "predicate": "likes", "object": "coffee", "valid_at": "2026-01-01", "invalid_at": None}]
        conn = FakeConn(rows)
        embedder = type("Embedder", (), {"embed_query": AsyncMock(return_value=[0.1, 0.2])})()
        config = type("Cfg", (), {"client_instance": FakeClient(conn), "table_name": "VectraFact", "embedder": embedder})()
        store = FactStore(config)

        facts = await store.read("session-1", "what does the user like?", limit=5)

        conn.fetch.assert_called_once()
        assert '"invalid_at" IS NULL' in conn.last_query
        assert '"session_id" = $1' in conn.last_query
        assert "order by" in conn.last_query.lower()
        assert conn.last_args[0] == "session-1"

        assert len(facts) == 1
        assert facts[0]["object"] == "coffee"
        assert "embedding" not in facts[0]

    async def test_returns_empty_and_does_not_query_when_session_id_missing(self):
        conn = FakeConn([])
        embedder = type("Embedder", (), {"embed_query": AsyncMock(return_value=[0.1])})()
        config = type("Cfg", (), {"client_instance": FakeClient(conn), "table_name": "VectraFact", "embedder": embedder})()
        store = FactStore(config)

        facts = await store.read(None, "query")

        assert facts == []
        conn.fetch.assert_not_called()

    async def test_respects_limit(self):
        conn = FakeConn([])
        embedder = type("Embedder", (), {"embed_query": AsyncMock(return_value=[0.1])})()
        config = type("Cfg", (), {"client_instance": FakeClient(conn), "table_name": "VectraFact", "embedder": embedder})()
        store = FactStore(config)

        await store.read("session-1", "query", limit=3)

        assert "limit" in conn.last_query.lower()
        assert 3 in conn.last_args
