from unittest.mock import AsyncMock
from vectra.backends.postgres_store import PostgresVectorStore
from vectra.backends.prisma_store import PrismaVectorStore


class FakeConn:
    def __init__(self):
        self.executemany = AsyncMock(return_value=None)
        self.fetch = AsyncMock(return_value=[])
        self.execute = AsyncMock(return_value="CREATE TABLE")


def make_postgres_config(client):
    return type("Cfg", (), {"table_name": "document", "column_map": {}, "client_instance": client})()


def make_prisma_config(client):
    return type("Cfg", (), {
        "table_name": "Document",
        "column_map": {"content": "content", "metadata": "metadata", "vector": "embedding"},
        "client_instance": client,
    })()


class TestDimensionConfig:
    async def test_postgres_creates_table_with_given_dimension(self):
        client = FakeConn()
        store = PostgresVectorStore(make_postgres_config(client))

        await store.ensure_indexes(dimensions=768)

        create_calls = [c for c in client.execute.call_args_list if "vector(" in c.args[0]]
        assert any("vector(768)" in c.args[0] for c in create_calls)
        assert not any("vector(1536)" in c.args[0] for c in create_calls)

    async def test_postgres_defaults_to_1536_when_no_dimension_given(self):
        client = FakeConn()
        store = PostgresVectorStore(make_postgres_config(client))

        await store.ensure_indexes()

        create_calls = [c for c in client.execute.call_args_list if "vector(" in c.args[0]]
        assert any("vector(1536)" in c.args[0] for c in create_calls)

    async def test_prisma_alters_column_with_given_dimension(self):
        client = AsyncMock()
        client.query_raw = AsyncMock(return_value=[])  # no existing columns -> triggers ADD COLUMN path
        client.execute_raw = AsyncMock(return_value=None)
        store = PrismaVectorStore(make_prisma_config(client))

        await store.ensure_indexes(dimensions=768)

        alter_calls = [c for c in client.execute_raw.call_args_list if "ADD COLUMN" in str(c.args[0]) and "vector(" in str(c.args[0])]
        assert any("vector(768)" in str(c.args[0]) for c in alter_calls)
        assert not any("vector(1536)" in str(c.args[0]) for c in alter_calls)
