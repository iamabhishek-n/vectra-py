import json
from unittest.mock import AsyncMock
from vectra.backends.weaviate_store import WeaviateVectorStore


def make_config(collection):
    client = type("Client", (), {"collections": type("Collections", (), {"get": lambda self, name: collection})()})()
    return type("Cfg", (), {"table_name": "Document", "client_instance": client})()


class TestWeaviateVectorStore:
    async def test_add_documents_inserts_objects_with_content_metadata_and_vector(self):
        collection = type("Collection", (), {})()
        collection.data = type("Data", (), {"insert_many": AsyncMock(return_value={})})()
        store = WeaviateVectorStore(make_config(collection))

        await store.add_documents([
            {"id": "doc-1", "content": "hello world", "embedding": [0.1, 0.2], "metadata": {"source": "a.md"}},
        ])

        collection.data.insert_many.assert_called_once()
        objects = collection.data.insert_many.call_args[0][0]
        assert objects[0]["properties"]["content"] == "hello world"
        assert objects[0]["properties"]["metadata"] == json.dumps({"source": "a.md"})
        assert objects[0]["vector"] == [0.1, 0.2]

    async def test_upsert_documents_behaves_the_same_as_add_documents(self):
        collection = type("Collection", (), {})()
        collection.data = type("Data", (), {"insert_many": AsyncMock(return_value={})})()
        store = WeaviateVectorStore(make_config(collection))

        await store.upsert_documents([{"id": "doc-1", "content": "x", "embedding": [0.1], "metadata": {}}])

        collection.data.insert_many.assert_called_once()

    async def test_similarity_search_converts_distance_to_higher_is_better_score(self):
        collection = type("Collection", (), {})()
        collection.query = type("Query", (), {"near_vector": AsyncMock(return_value={
            "objects": [
                {"properties": {"content": "hello world", "metadata": json.dumps({"source": "a.md"})}, "metadata": {"distance": 0.2}},
            ]
        })})()
        store = WeaviateVectorStore(make_config(collection))

        results = await store.similarity_search([0.1, 0.2], limit=5)

        _, kwargs = collection.query.near_vector.call_args
        assert kwargs["limit"] == 5
        assert len(results) == 1
        assert results[0]["content"] == "hello world"
        assert results[0]["metadata"]["source"] == "a.md"
        assert results[0]["score"] == 0.8
