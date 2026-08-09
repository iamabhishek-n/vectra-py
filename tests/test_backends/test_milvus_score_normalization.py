import pytest
from unittest.mock import AsyncMock

from vectra.backends.milvus_store import MilvusVectorStore
from vectra.core import VectraClient
from vectra.config import (
    VectraConfig,
    EmbeddingConfig,
    LLMConfig,
    DatabaseConfig,
    ProviderType,
)


def make_config(client, metric_type=None):
    return type("Cfg", (), {
        "table_name": "rag_collection",
        "client_instance": client,
        "metric_type": metric_type,
    })()


class TestMilvusScoreNormalizationCosine:
    async def test_cosine_negative_score_is_passed_through_unchanged(self):
        # COSINE (and IP) scores are already higher-is-better in Milvus's
        # convention, including negative values for dissimilar vectors in the
        # real [-1, 1] COSINE range. A magnitude-based heuristic (e.g. "invert
        # anything <= 1") would misclassify this negative score; the
        # metric_type-aware fix must leave it untouched.
        client = AsyncMock()
        client.search = AsyncMock(return_value={
            "results": [
                {"content": "dissimilar doc", "metadata": {}, "distance": -0.5},
            ]
        })
        store = MilvusVectorStore(make_config(client, metric_type="COSINE"))

        results = await store.similarity_search([0.1, 0.2], limit=1)

        assert results[0]["score"] == -0.5

    async def test_default_metric_type_is_cosine_passthrough(self):
        # No metric_type set at all -> defaults to COSINE passthrough.
        client = AsyncMock()
        client.search = AsyncMock(return_value={
            "results": [
                {"content": "some doc", "metadata": {}, "distance": -0.5},
            ]
        })
        store = MilvusVectorStore(make_config(client))

        results = await store.similarity_search([0.1, 0.2], limit=1)

        assert results[0]["score"] == -0.5


class TestMilvusScoreNormalizationL2:
    async def test_l2_zero_distance_normalizes_to_best_score(self):
        # A perfect match (L2 distance 0) must normalize to the best possible
        # score, 1.0, via 1 / (1 + 0).
        client = AsyncMock()
        client.search = AsyncMock(return_value={
            "results": [
                {"content": "perfect match", "metadata": {}, "distance": 0.0},
            ]
        })
        store = MilvusVectorStore(make_config(client, metric_type="L2"))

        results = await store.similarity_search([0.1, 0.2], limit=1)

        assert results[0]["score"] == pytest.approx(1.0)

    async def test_l2_larger_distance_normalizes_lower_monotonically(self):
        # A distance of 2.0 (farther away) must normalize to a strictly lower
        # score than a distance of 0.5 (closer). This also exercises the
        # region around the old, unsound "score <= 1 means already normalized"
        # heuristic boundary: 0.5 is below 1.0 and 2.0 is above it, but both
        # must be treated consistently as raw L2 distances, not as if one were
        # already a normalized score.
        client = AsyncMock()
        client.search = AsyncMock(return_value={
            "results": [
                {"content": "far doc", "metadata": {}, "distance": 2.0},
                {"content": "near doc", "metadata": {}, "distance": 0.5},
            ]
        })
        store = MilvusVectorStore(make_config(client, metric_type="L2"))

        results = await store.similarity_search([0.1, 0.2], limit=2)
        scores = {r["content"]: r["score"] for r in results}

        assert scores["far doc"] == pytest.approx(1.0 / 3.0)
        assert scores["near doc"] == pytest.approx(1.0 / 1.5)
        assert scores["far doc"] < scores["near doc"]

    async def test_l2_negative_distance_does_not_raise(self):
        # Real L2 distance is never negative, but a misbehaving client
        # returning one must not crash the store with a ZeroDivisionError
        # (1/(1+n) is undefined at n=-1) or produce a non-monotonic result.
        client = AsyncMock()
        client.search = AsyncMock(return_value={
            "results": [
                {"content": "bad distance", "metadata": {}, "distance": -1.0},
            ]
        })
        store = MilvusVectorStore(make_config(client, metric_type="L2"))

        results = await store.similarity_search([0.1, 0.2], limit=1)

        assert results[0]["score"] == pytest.approx(1.0)

    async def test_metric_type_is_case_and_whitespace_insensitive(self):
        client = AsyncMock()
        client.search = AsyncMock(return_value={
            "results": [{"content": "doc", "metadata": {}, "distance": 1.0}],
        })
        store = MilvusVectorStore(make_config(client, metric_type=" l2 "))

        results = await store.similarity_search([0.1, 0.2], limit=1)

        assert results[0]["score"] == pytest.approx(0.5)


class TestMilvusMetricTypeThroughPublicConfig:
    async def test_metric_type_set_via_vectra_client_config_reaches_milvus_store(self):
        # End-to-end: metric_type must survive VectraConfig -> DatabaseConfig
        # (a Pydantic model) -> VectraClient.__init__ -> MilvusVectorStore's
        # constructor, and actually change normalization behavior. This is
        # deliberately NOT constructing MilvusVectorStore directly, to catch a
        # regression where the Pydantic schema silently drops an unrecognized
        # `metric_type` field before it ever reaches the store.
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value={
            "results": [
                {"content": "close doc", "metadata": {}, "distance": 0.0},
            ]
        })

        config = VectraConfig(
            embedding=EmbeddingConfig(provider=ProviderType.OPENAI, api_key="test-key"),
            llm=LLMConfig(provider=ProviderType.OPENAI, api_key="test-key", model_name="gpt-4o-mini"),
            database=DatabaseConfig(
                type="milvus",
                client_instance=mock_client,
                metric_type="L2",
            ),
        )

        client = VectraClient(config)

        assert isinstance(client.vector_store, MilvusVectorStore)
        assert client.vector_store.metric_type == "L2"

        results = await client.vector_store.similarity_search([0.1, 0.2], limit=1)

        # A distance of 0.0 under L2 normalization must become the best
        # possible score (1.0). If `metric_type='L2'` failed to reach the
        # store (e.g. dropped by config validation, or defaulted to COSINE
        # passthrough), the score would instead be 0.0 -- so this assertion
        # is discriminating proof the value made the full round trip.
        assert results[0]["score"] == pytest.approx(1.0)
