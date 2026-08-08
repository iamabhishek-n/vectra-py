from unittest.mock import AsyncMock
from vectra.backends.milvus_store import MilvusVectorStore


def make_config(client):
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": client})()


class TestMilvusVectorStore:
    async def test_add_documents_inserts_vector_content_and_metadata(self):
        client = AsyncMock()
        store = MilvusVectorStore(make_config(client))

        await store.add_documents([{"content": "hello world", "metadata": {"a": 1}, "embedding": [0.1, 0.2]}])

        client.insert.assert_called_once_with(
            collection_name="rag_collection",
            fields_data=[{"vector": [0.1, 0.2], "content": "hello world", "metadata": {"a": 1}}],
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

    async def test_update_documents_merges_metadata_and_reupserts_with_existing_vector(self):
        client = AsyncMock()
        client.query = AsyncMock(return_value=[
            {"id": 1, "vector": [0.1, 0.2], "content": "old text", "metadata": {"a": 1}},
        ])
        client.upsert = AsyncMock()
        store = MilvusVectorStore(make_config(client))

        updated = await store.update_documents({"category": "docs"}, {"metadata": {"b": 2}})

        assert updated == 1
        client.upsert.assert_called_once()
        fields = client.upsert.call_args.kwargs["fields_data"]
        assert fields[0]["vector"] == [0.1, 0.2]
        assert fields[0]["content"] == "old text"
        assert fields[0]["metadata"] == {"a": 1, "b": 2}

    async def test_update_documents_returns_zero_when_nothing_matches(self):
        client = AsyncMock()
        client.query = AsyncMock(return_value=[])
        store = MilvusVectorStore(make_config(client))

        updated = await store.update_documents({"category": "docs"}, {"metadata": {"b": 2}})

        assert updated == 0
