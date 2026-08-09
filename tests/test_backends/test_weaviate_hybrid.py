import json
from unittest.mock import AsyncMock
from vectra.backends.weaviate_store import WeaviateVectorStore


def make_config(collection):
    client = type("Client", (), {"collections": type("Collections", (), {"get": lambda self, name: collection})()})()
    return type("Cfg", (), {"table_name": "Document", "client_instance": client})()


def make_collection():
    return type("Collection", (), {})()


class TestWeaviateVectorStoreHybridAndCrud:
    async def test_hybrid_search_delegates_to_native_weaviate_hybrid_query(self):
        collection = make_collection()
        collection.query = type("Query", (), {"hybrid": AsyncMock(return_value={
            "objects": [
                {"properties": {"content": "quick fox jumps high", "metadata": "{}"}, "metadata": {"score": 0.95}},
            ]
        })})()
        store = WeaviateVectorStore(make_config(collection))

        results = await store.hybrid_search("quick fox", [0.1, 0.2], limit=5)

        args, kwargs = collection.query.hybrid.call_args
        assert args[0] == "quick fox"
        assert kwargs["vector"] == [0.1, 0.2]
        assert kwargs["limit"] == 5
        assert len(results) == 1
        assert results[0]["content"] == "quick fox jumps high"

    async def test_list_documents_fetches_by_filter_with_cursor_pagination(self):
        collection = make_collection()
        collection.query = type("Query", (), {"fetch_objects": AsyncMock(return_value={
            "objects": [{"uuid": "id-1", "properties": {"content": "doc a", "metadata": "{}"}}]
        })})()
        store = WeaviateVectorStore(make_config(collection))

        docs, cursor = await store.list_documents(limit=1)

        _, kwargs = collection.query.fetch_objects.call_args
        assert kwargs["limit"] == 1
        assert len(docs) == 1
        assert docs[0]["id"] == "id-1"
        assert cursor == "id-1"

    async def test_delete_documents_deletes_matching_ids_and_returns_real_count(self):
        collection = make_collection()
        collection.query = type("Query", (), {"fetch_objects": AsyncMock(return_value={
            "objects": [
                {"uuid": "id-1", "properties": {"content": "a", "metadata": "{}"}},
                {"uuid": "id-2", "properties": {"content": "b", "metadata": "{}"}},
            ]
        })})()
        collection.data = type("Data", (), {"delete_by_id": AsyncMock(return_value=None)})()
        store = WeaviateVectorStore(make_config(collection))

        result = await store.delete_documents({"source": "a.md"})

        assert collection.data.delete_by_id.call_count == 2
        assert result == 2

    async def test_delete_documents_requires_a_filter(self):
        store = WeaviateVectorStore(make_config(make_collection()))

        try:
            await store.delete_documents({})
            assert False, "expected ValueError"
        except ValueError:
            pass

    async def test_update_documents_updates_matching_ids(self):
        collection = make_collection()
        collection.query = type("Query", (), {"fetch_objects": AsyncMock(return_value={
            "objects": [{"uuid": "id-1", "properties": {"content": "old", "metadata": "{}"}}]
        })})()
        collection.data = type("Data", (), {"update": AsyncMock(return_value=None)})()
        store = WeaviateVectorStore(make_config(collection))

        result = await store.update_documents({"source": "a.md"}, {"content": "new"})

        collection.data.update.assert_called_once()
        assert result == 1

    async def test_file_exists_queries_by_metadata_filter_and_returns_true(self):
        collection = make_collection()
        collection.query = type("Query", (), {"fetch_objects": AsyncMock(return_value={
            "objects": [{"uuid": "id-1", "properties": {}}]
        })})()
        store = WeaviateVectorStore(make_config(collection))

        exists = await store.file_exists("abc123", 100, 12345)

        assert exists is True

    async def test_file_exists_returns_false_when_no_match_found(self):
        collection = make_collection()
        collection.query = type("Query", (), {"fetch_objects": AsyncMock(return_value={"objects": []})})()
        store = WeaviateVectorStore(make_config(collection))

        exists = await store.file_exists("abc123", 100, 12345)

        assert exists is False
