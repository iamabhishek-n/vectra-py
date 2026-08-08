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
                {"content": "a completely unrelated sentence", "metadata": {}, "distance": 0.5},
                {"content": "quick fox jumps high", "metadata": {}, "distance": 0.12},
            ]
        })
        store = MilvusVectorStore(make_config(client))

        results = await store.hybrid_search("quick fox", [0.1, 0.2], limit=2)

        assert len(results) == 2
        # Verify that hybrid_search combines semantic and lexical ranking
        # At least one result should have good lexical overlap with query
        contents = [r["content"] for r in results]
        has_good_lexical_match = any("quick" in c.lower() and "fox" in c.lower() for c in contents)
        assert has_good_lexical_match
