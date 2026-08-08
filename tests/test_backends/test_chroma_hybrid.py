from unittest.mock import MagicMock
from vectra.backends.chroma_store import ChromaVectorStore


def make_config(client, collection):
    client.get_or_create_collection = MagicMock(return_value=collection)
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": client})()


class TestChromaHybridSearch:
    async def test_fuses_semantic_and_lexical_rank(self):
        collection = MagicMock()
        collection.query = MagicMock(return_value={
            "documents": [["the quick brown fox", "a completely unrelated sentence", "quick fox jumps high"]],
            "metadatas": [[{}, {}, {}]],
            "distances": [[0.1, 0.2, 0.15]],
        })
        store = ChromaVectorStore(make_config(MagicMock(), collection))

        results = await store.hybrid_search("quick fox", [0.1, 0.2], limit=2)

        assert len(results) == 2
        contents = [r["content"] for r in results]
        assert "a completely unrelated sentence" not in contents

    async def test_returns_results_with_no_lexical_overlap(self):
        collection = MagicMock()
        collection.query = MagicMock(return_value={
            "documents": [["alpha content", "beta content"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.3]],
        })
        store = ChromaVectorStore(make_config(MagicMock(), collection))

        results = await store.hybrid_search("zzz", [0.1, 0.2], limit=2)

        assert len(results) == 2
