import pytest
from unittest.mock import AsyncMock
from vectra.backends.prisma_store import PrismaVectorStore


def make_config(client, table_name="Document"):
    return type("Cfg", (), {"table_name": table_name, "column_map": {}, "client_instance": client})()


class TestPrismaVectorStore:
    async def test_rejects_unsafe_table_name_on_first_operation(self):
        store = PrismaVectorStore(make_config(AsyncMock(), table_name='a"; DROP TABLE x; --'))
        with pytest.raises(ValueError, match="Unsafe SQL identifier"):
            await store.add_documents([{"content": "x", "metadata": {}, "embedding": [0.1]}])

    async def test_add_documents_issues_one_execute_raw_per_document(self):
        client = AsyncMock()
        store = PrismaVectorStore(make_config(client))

        await store.add_documents([
            {"id": "doc-1", "content": "hello world", "metadata": {"a": 1}, "embedding": [0.3, 0.4]},
        ])

        client.execute_raw.assert_called_once()
        query, doc_id, content = client.execute_raw.call_args[0][:3]
        assert 'INSERT INTO "Document"' in query
        assert doc_id == "doc-1"
        assert content == "hello world"

    async def test_similarity_search_returns_mapped_rows(self):
        client = AsyncMock()
        client.query_raw = AsyncMock(return_value=[{"content": "hello world", "metadata": {"a": 1}, "score": 0.9}])
        store = PrismaVectorStore(make_config(client))

        results = await store.similarity_search([0.3, 0.4], limit=5)

        assert results == [{"content": "hello world", "metadata": {"a": 1}, "score": 0.9}]
