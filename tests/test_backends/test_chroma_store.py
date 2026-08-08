from unittest.mock import MagicMock
from vectra.backends.chroma_store import ChromaVectorStore


def make_config(client, collection):
    client.get_or_create_collection = MagicMock(return_value=collection)
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": client})()


class TestChromaVectorStore:
    async def test_add_documents_calls_collection_add(self):
        collection = MagicMock()
        store = ChromaVectorStore(make_config(MagicMock(), collection))

        await store.add_documents([{"content": "hello world", "metadata": {"a": 1}, "embedding": [0.1, 0.2]}])

        collection.add.assert_called_once()
        call = collection.add.call_args.kwargs
        assert call["embeddings"] == [[0.1, 0.2]]
        assert call["metadatas"] == [{"a": 1}]
        assert call["documents"] == ["hello world"]

    async def test_similarity_search_maps_batched_response(self):
        collection = MagicMock()
        collection.query = MagicMock(return_value={
            "documents": [["hello world"]],
            "metadatas": [[{"a": 1}]],
            "distances": [[0.13]],
        })
        store = ChromaVectorStore(make_config(MagicMock(), collection))

        results = await store.similarity_search([0.1, 0.2], limit=5)

        assert results == [{"content": "hello world", "metadata": {"a": 1}, "score": 0.87}]
