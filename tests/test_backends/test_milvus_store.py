from unittest.mock import AsyncMock
from vectra.backends.milvus_store import MilvusVectorStore


def make_config(client):
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": client})()


class TestMilvusVectorStore:
    async def test_add_documents_inserts_vector_content_and_json_metadata(self):
        client = AsyncMock()
        store = MilvusVectorStore(make_config(client))

        await store.add_documents([{"content": "hello world", "metadata": {"a": 1}, "embedding": [0.1, 0.2]}])

        client.insert.assert_called_once_with(
            collection_name="rag_collection",
            fields_data=[{"vector": [0.1, 0.2], "content": "hello world", "metadata": '{"a": 1}'}],
        )

    async def test_delete_documents_returns_the_actual_delete_count(self):
        client = AsyncMock()
        client.delete = AsyncMock(return_value={"delete_count": 3})
        store = MilvusVectorStore(make_config(client))

        deleted = await store.delete_documents({"category": "docs"})

        assert deleted == 3

    async def test_delete_documents_handles_object_style_result(self):
        client = AsyncMock()
        result = type("Result", (), {"delete_count": 7})()
        client.delete = AsyncMock(return_value=result)
        store = MilvusVectorStore(make_config(client))

        deleted = await store.delete_documents({"category": "docs"})

        assert deleted == 7
