from unittest.mock import AsyncMock
from vectra.backends.milvus_store import MilvusVectorStore


def make_config(client):
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": client})()


class TestMilvusHybridSearch:
    async def test_fuses_semantic_and_lexical_rank(self):
        client = AsyncMock()
        client.search = AsyncMock(return_value={
            "results": [
                {"content": "the quick brown fox", "metadata": {}, "distance": 0.1},
                {"content": "a completely unrelated sentence", "metadata": {}, "distance": 0.15},
                {"content": "quick fox jumps high", "metadata": {}, "distance": 0.12},
            ]
        })
        store = MilvusVectorStore(make_config(client))

        results = await store.hybrid_search("quick fox", [0.1, 0.2], limit=2)

        assert len(results) == 2
        contents = [r["content"] for r in results]
        assert "a completely unrelated sentence" not in contents
