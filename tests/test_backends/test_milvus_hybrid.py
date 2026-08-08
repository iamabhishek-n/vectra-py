import pytest
from unittest.mock import AsyncMock
from vectra.backends.milvus_store import MilvusVectorStore


def make_config(client):
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": client})()


class TestMilvusHybridSearch:
    async def test_lexical_ranking_favors_query_term_overlap(self):
        # Weaker than Chroma/Qdrant's equivalent test by necessity: Milvus's
        # similarity_search intentionally still returns raw distance as `score`
        # (lower-is-better) per Task 5's brief, so hybrid_search's semantic
        # ranking (which sorts `score` descending, assuming higher-is-better)
        # is inverted for Milvus until Task 9 adds metric-type-aware
        # normalization. RRF then fuses a correct lexical ranking with an
        # inverted semantic one, so exclusion of irrelevant results can't be
        # asserted yet (see test_semantic_ranking_is_inverted_until_task_9_fix
        # below). This test only checks the lexical signal is present at all.
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
        contents = [r["content"] for r in results]
        has_good_lexical_match = any("quick" in c.lower() and "fox" in c.lower() for c in contents)
        assert has_good_lexical_match

    @pytest.mark.xfail(
        reason="Known limitation: MilvusVectorStore.similarity_search exposes raw "
        "distance as `score` (lower-is-better), but hybrid_search's RRF fusion "
        "sorts semantic rank descending assuming higher-is-better, inverting "
        "semantic ranking for Milvus. Fixed by Task 9's metric_type-aware "
        "score normalization.",
        strict=True,
    )
    async def test_semantic_ranking_is_inverted_until_task_9_fix(self):
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

        contents = [r["content"] for r in results]
        assert "a completely unrelated sentence" not in contents
