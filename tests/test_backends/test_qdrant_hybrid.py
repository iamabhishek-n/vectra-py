from unittest.mock import AsyncMock
from vectra.backends.qdrant_store import QdrantVectorStore


def make_config(client):
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": client})()


class TestQdrantHybridSearch:
    async def test_fuses_semantic_and_lexical_rank(self):
        client = AsyncMock()
        client.search = AsyncMock(return_value=[
            {"payload": {"content": "the quick brown fox", "metadata": {}}, "score": 0.9},
            {"payload": {"content": "a completely unrelated sentence", "metadata": {}}, "score": 0.8},
            {"payload": {"content": "quick fox jumps high", "metadata": {}}, "score": 0.85},
        ])
        store = QdrantVectorStore(make_config(client))

        results = await store.hybrid_search("quick fox", [0.1, 0.2], limit=2)

        assert len(results) == 2
        contents = [r["content"] for r in results]
        assert "a completely unrelated sentence" not in contents
