from unittest.mock import AsyncMock
from vectra.backends.qdrant_store import QdrantVectorStore


def make_config(client):
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": client})()


class TestQdrantVectorStore:
    async def test_add_documents_upserts_points_with_vector_and_payload(self):
        client = AsyncMock()
        store = QdrantVectorStore(make_config(client))

        await store.add_documents([{"content": "hello world", "metadata": {"a": 1}, "embedding": [0.1, 0.2]}])

        client.upsert.assert_called_once()
        collection, kwargs = client.upsert.call_args[0][0], client.upsert.call_args[1]
        assert collection == "rag_collection"
        assert kwargs["points"][0]["vector"] == [0.1, 0.2]
        assert kwargs["points"][0]["payload"] == {"content": "hello world", "metadata": {"a": 1}}

    async def test_similarity_search_maps_hits(self):
        client = AsyncMock()
        client.search = AsyncMock(return_value=[
            {"payload": {"content": "hello world", "metadata": {"a": 1}}, "score": 0.92},
        ])
        store = QdrantVectorStore(make_config(client))

        results = await store.similarity_search([0.1, 0.2], limit=5)

        assert results == [{"content": "hello world", "metadata": {"a": 1}, "score": 0.92}]

    def test_normalize_filter_builds_must_clause(self):
        store = QdrantVectorStore(make_config(AsyncMock()))
        assert store._normalize_filter({"category": "docs"}) == {
            "must": [{"key": "metadata.category", "match": {"value": "docs"}}]
        }
