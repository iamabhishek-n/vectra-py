from unittest.mock import AsyncMock
from vectra.backends.milvus_store import MilvusVectorStore


def make_config(client, metric_type=None):
    return type("Cfg", (), {
        "table_name": "rag_collection",
        "client_instance": client,
        "metric_type": metric_type,
    })()


class TestMilvusHybridSearch:
    async def test_lexical_ranking_favors_query_term_overlap(self):
        # Uses the default COSINE metric_type, under which similarity_search
        # passes the raw `distance` value through unchanged (Task 9's
        # metric-type-aware normalization only transforms L2 scores). This
        # test only checks the lexical signal is present at all; the
        # semantic-ranking-direction case (which requires the L2 metric to be
        # discriminating) is covered by
        # test_semantic_ranking_is_inverted_until_task_9_fix below.
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

    async def test_semantic_ranking_respects_l2_metric_type(self):
        # Regression test for Task 9's Bug A fix. These `distance` values are
        # genuine L2-style raw distances (lower is better): 0.1 and 0.12 are
        # close matches, 0.5 is a distant/irrelevant one. Before the fix,
        # similarity_search passed `distance` through unnormalized and
        # hybrid_search's semantic ranking sorted it descending (assuming
        # higher-is-better), so the *farthest* match ("a completely unrelated
        # sentence", distance 0.5) would rank as the best semantic result and
        # survive RRF fusion into the top 2. With metric_type='L2' correctly
        # inverting scores via 1/(1+distance), the closest matches win the
        # semantic ranking and the unrelated sentence is fused out of the
        # top-2 results.
        client = AsyncMock()
        client.search = AsyncMock(return_value={
            "results": [
                {"content": "the quick brown fox", "metadata": {}, "distance": 0.1},
                {"content": "a completely unrelated sentence", "metadata": {}, "distance": 0.5},
                {"content": "quick fox jumps high", "metadata": {}, "distance": 0.12},
            ]
        })
        store = MilvusVectorStore(make_config(client, metric_type="L2"))

        results = await store.hybrid_search("quick fox", [0.1, 0.2], limit=2)

        contents = [r["content"] for r in results]
        assert "a completely unrelated sentence" not in contents
