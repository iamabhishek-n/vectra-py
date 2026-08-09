import pytest
from unittest.mock import AsyncMock
from vectra.memory.fact_store import FactStore


class FakeConn:
    def __init__(self):
        self.queries = []
        self.execute = AsyncMock(side_effect=self._record)

    async def _record(self, q, *args):
        self.queries.append(q)
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


def make_config(client):
    return type("Cfg", (), {"client_instance": client, "table_name": "VectraFact"})()


class TestFactStoreEnsureIndexes:
    async def test_creates_table_with_given_dimension_and_indexes(self):
        conn = FakeConn()
        store = FactStore(make_config(FakeClient(conn)))

        await store.ensure_indexes(768)

        create_table = next((q for q in conn.queries if "CREATE TABLE IF NOT EXISTS" in q), None)
        assert create_table is not None
        assert "vector(768)" in create_table
        assert '"subject"' in create_table
        assert '"predicate"' in create_table
        assert '"object"' in create_table
        assert '"valid_at"' in create_table
        assert '"invalid_at"' in create_table
        assert '"session_id"' in create_table

        vec_index = next((q for q in conn.queries if "USING hnsw" in q or "USING ivfflat" in q), None)
        assert vec_index is not None

        temporal_index = next((q for q in conn.queries if "index" in q.lower() and '"session_id"' in q and '"valid_at"' in q), None)
        assert temporal_index is not None

    def test_rejects_unsafe_table_name(self):
        bad_config = type("Cfg", (), {"client_instance": FakeClient(FakeConn()), "table_name": "Fact; DROP TABLE users;--"})()
        with pytest.raises(ValueError, match="[Uu]nsafe SQL identifier"):
            FactStore(bad_config)
