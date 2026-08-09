from unittest.mock import AsyncMock
from vectra.backends.pinecone_store import PineconeVectorStore


def make_config(client):
    return type("Cfg", (), {"table_name": "my-namespace", "client_instance": client, "dimensions": None})()


class TestPineconeVectorStore:
    async def test_add_documents_upserts_vectors_with_content_bundled_into_metadata(self):
        client = AsyncMock()
        store = PineconeVectorStore(make_config(client))

        await store.add_documents([
            {"id": "doc-1", "content": "hello world", "embedding": [0.1, 0.2], "metadata": {"source": "a.md"}},
        ])

        client.upsert.assert_called_once()
        vectors = client.upsert.call_args[0][0]
        assert vectors[0]["id"] == "doc-1"
        assert vectors[0]["values"] == [0.1, 0.2]
        assert vectors[0]["metadata"]["content"] == "hello world"
        assert vectors[0]["metadata"]["source"] == "a.md"

    async def test_upsert_documents_behaves_the_same_as_add_documents(self):
        client = AsyncMock()
        store = PineconeVectorStore(make_config(client))

        await store.upsert_documents([{"id": "doc-1", "content": "x", "embedding": [0.1], "metadata": {}}])

        client.upsert.assert_called_once()

    async def test_similarity_search_queries_and_unbundles_content_out_of_metadata(self):
        client = AsyncMock()
        client.query = AsyncMock(return_value={
            "matches": [
                {"id": "doc-1", "score": 0.95, "metadata": {"content": "hello world", "source": "a.md"}},
            ]
        })
        store = PineconeVectorStore(make_config(client))

        results = await store.similarity_search([0.1, 0.2], limit=5, filter={"source": "a.md"})

        _, kwargs = client.query.call_args
        assert kwargs["vector"] == [0.1, 0.2]
        assert kwargs["top_k"] == 5
        assert len(results) == 1
        assert results[0]["content"] == "hello world"
        assert results[0]["metadata"]["source"] == "a.md"
        assert results[0]["score"] == 0.95
        assert "content" not in results[0]["metadata"]
