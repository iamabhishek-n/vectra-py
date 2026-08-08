from unittest.mock import AsyncMock
from vectra.backends.postgres_store import PostgresVectorStore


class FakeConn:
    """Plain class, not AsyncMock — deliberately has no 'acquire' attribute
    so PostgresVectorStore._get_connection() treats it as a bare connection,
    not a pool. See Task 4's Interfaces note for why AsyncMock() is unsafe here."""
    def __init__(self):
        self.executemany = AsyncMock(return_value=None)
        self.fetch = AsyncMock(return_value=[])
        self.execute = AsyncMock(return_value="UPDATE 0")


def make_config(client):
    return type("Cfg", (), {"table_name": "document", "column_map": {}, "client_instance": client})()


class TestPostgresVectorStore:
    async def test_add_documents_batches_all_rows_into_one_executemany_call(self):
        client = FakeConn()
        store = PostgresVectorStore(make_config(client))

        await store.add_documents([
            {"id": "doc-1", "content": "hello world", "metadata": {"a": 1}, "embedding": [0.1, 0.2, 0.3]},
        ])

        client.executemany.assert_called_once()
        sql, rows = client.executemany.call_args[0]
        assert 'INSERT INTO "document"' in sql
        assert rows[0][0] == "doc-1"
        assert rows[0][1] == "hello world"

    async def test_similarity_search_converts_distance_to_score(self):
        client = FakeConn()
        client.fetch = AsyncMock(return_value=[
            {"id": "doc-1", "content": "hello world", "metadata": '{"a": 1}', "distance": 0.13},
        ])
        store = PostgresVectorStore(make_config(client))

        results = await store.similarity_search([0.1, 0.2, 0.3], limit=5)

        assert results == [{"id": "doc-1", "content": "hello world", "metadata": {"a": 1}, "score": 0.87}]

    async def test_update_documents_sets_content_and_metadata(self):
        client = FakeConn()
        client.execute = AsyncMock(return_value="UPDATE 2")
        store = PostgresVectorStore(make_config(client))

        updated = await store.update_documents({"category": "docs"}, {"content": "new text"})

        assert updated == 2
        sql = client.execute.call_args[0][0]
        assert "UPDATE" in sql
        assert '"content" = $1' in sql

    async def test_update_documents_returns_zero_for_empty_update_data(self):
        client = FakeConn()
        store = PostgresVectorStore(make_config(client))

        updated = await store.update_documents({"category": "docs"}, {})

        assert updated == 0
        client.execute.assert_not_called()
