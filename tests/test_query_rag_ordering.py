"""Regression tests for Task 9's Bug B fix.

query_rag()'s keyword-boost re-sort (`boosted.sort(...)`) used to always
re-sort retrieved documents by raw vector `score`, silently discarding any
order already established by reranking, HYBRID search, MULTI_QUERY's RRF
fusion, or MMR's diversity selection. Each test below builds a fixture where
the authoritative (reranked / fused / diversity-selected) order deliberately
disagrees with a plain descending-raw-score sort -- so the old, buggy code
(which always re-sorts by score) would visibly reorder the fixture, while the
fixed code (which skips the re-sort whenever one of those orderings applies)
preserves it.

The final document order is observed indirectly via the prompt passed to
`llm.generate`, since `query_rag`'s `context` (built from the post-boost
document list) is what actually encodes retrieval order in the response --
the `sources` field in the return value is built from the pre-boost `docs`
list and would not catch a regression in the boost re-sort itself.
"""
from unittest.mock import AsyncMock, MagicMock

from vectra.core import VectraClient
from vectra.config import (
    VectraConfig,
    EmbeddingConfig,
    LLMConfig,
    DatabaseConfig,
    RetrievalConfig,
    RerankingConfig,
    RerankingProvider,
    ProviderType,
    RetrievalStrategy,
)


def base_config(**overrides):
    defaults = dict(
        embedding=EmbeddingConfig(provider=ProviderType.OPENAI, api_key="test-key"),
        llm=LLMConfig(provider=ProviderType.OPENAI, api_key="test-key", model_name="gpt-4o-mini"),
        database=DatabaseConfig(type="chroma", client_instance=MagicMock()),
    )
    defaults.update(overrides)
    return VectraConfig(**defaults)


def wire_common_mocks(client, embed_dim=4):
    client.embedder.embed_query = AsyncMock(return_value=[0.1] * embed_dim)
    client.llm = MagicMock()
    client.llm.generate = AsyncMock(return_value="the answer")
    client.config.guardrails = None
    return client


def prompt_sent_to_llm(client) -> str:
    args, _kwargs = client.llm.generate.call_args
    return args[0]


class TestRerankingOrderIsPreserved:
    async def test_final_order_matches_reranker_not_raw_score(self):
        # "preferred" has a LOWER raw score than "distractor" -- a leftover
        # boost re-sort (score + 0.1*boost, boost=0 for both since neither has
        # matching metadata keywords) would put "distractor" first. Only the
        # reranker's returned order (preferred first) is authoritative.
        config = base_config(
            reranking=RerankingConfig(enabled=True, provider=RerankingProvider.COHERE, api_key="test-key"),
        )
        client = VectraClient(config)
        wire_common_mocks(client)

        preferred = {"content": "PREFERRED_MARKER relevant document", "score": 0.1, "metadata": {}}
        distractor = {"content": "DISTRACTOR_MARKER irrelevant document", "score": 0.9, "metadata": {}}

        client.vector_store.similarity_search = AsyncMock(return_value=[distractor, preferred])
        # The reranker is authoritative: it puts the low-raw-score doc first.
        client.reranker.rerank = AsyncMock(return_value=[preferred, distractor])

        await client.query_rag("test query")

        prompt = prompt_sent_to_llm(client)
        assert prompt.index("PREFERRED_MARKER") < prompt.index("DISTRACTOR_MARKER")


class TestMultiQueryOrderIsPreserved:
    async def test_final_order_matches_rrf_fusion_not_raw_score(self):
        # "preferred" ranks first in every per-query result list (so RRF fuses
        # it to the top), but has a LOWER raw score than "distractor". A
        # leftover boost re-sort would put "distractor" (raw score 0.9) ahead
        # of "preferred" (raw score 0.1), undoing the fusion.
        config = base_config(
            retrieval=RetrievalConfig(
                strategy=RetrievalStrategy.MULTI_QUERY,
                llm_config=LLMConfig(provider=ProviderType.OPENAI, api_key="test-key", model_name="gpt-4o-mini"),
            ),
        )
        client = VectraClient(config)
        wire_common_mocks(client)
        client._generate_multi_queries = AsyncMock(return_value=["a variant query"])

        preferred = {"content": "PREFERRED_MARKER relevant document", "score": 0.1, "metadata": {}}
        distractor = {"content": "DISTRACTOR_MARKER irrelevant document", "score": 0.9, "metadata": {}}

        # Both per-query searches rank `preferred` first -- RRF fuses it to
        # rank 0 regardless of the `score` field.
        per_query_result = [preferred, distractor]
        client.vector_store.similarity_search = AsyncMock(return_value=per_query_result)

        await client.query_rag("test query")

        prompt = prompt_sent_to_llm(client)
        assert prompt.index("PREFERRED_MARKER") < prompt.index("DISTRACTOR_MARKER")


class TestMmrOrderIsPreserved:
    async def test_final_order_matches_diversity_selection_not_raw_score(self):
        # Three candidates: `top` has the highest raw score and is picked
        # first by MMR (as it always is -- MMR's first pick is always the
        # single highest-scoring candidate). `near_dup` has the second-highest
        # raw score but is embedding-near-identical to `top`, so MMR penalizes
        # it for redundancy. `diverse` has the lowest raw score but is
        # embedding-orthogonal to `top`, so MMR prefers it over `near_dup`.
        # MMR's authoritative order is therefore [top, diverse, near_dup] --
        # a leftover boost re-sort (by raw score) would produce
        # [top, near_dup, diverse] instead, swapping the last two.
        config = base_config(
            retrieval=RetrievalConfig(
                strategy=RetrievalStrategy.MMR,
                mmr_fetch_k=10,  # > k (5, the reranking-disabled default) so MMR selection runs
                mmr_lambda=0.5,
            ),
        )
        client = VectraClient(config)
        wire_common_mocks(client, embed_dim=3)

        top = {"content": "TOP_MARKER widget document", "score": 0.9, "metadata": {}}
        near_dup = {"content": "NEARDUP_MARKER widget document variant", "score": 0.85, "metadata": {}}
        diverse = {"content": "DIVERSE_MARKER gardening document", "score": 0.5, "metadata": {}}

        client.vector_store.similarity_search = AsyncMock(return_value=[top, near_dup, diverse])
        client.embedder.embed_documents = AsyncMock(return_value=[
            [1.0, 0.0, 0.0],   # top
            [0.99, 0.01, 0.0],  # near_dup: near-identical to top
            [0.0, 1.0, 0.0],   # diverse: orthogonal to top
        ])

        await client.query_rag("test query")

        prompt = prompt_sent_to_llm(client)
        assert prompt.index("TOP_MARKER") < prompt.index("DIVERSE_MARKER") < prompt.index("NEARDUP_MARKER")


class TestNaivePathStillAppliesBoostResort:
    async def test_keyword_boosted_low_score_doc_still_wins_on_naive_path(self):
        # Control case: on the plain NAIVE path (no reranking, no HYBRID/
        # MULTI_QUERY/MMR strategy), the keyword-boost re-sort must still run.
        # `boosted` has a lower raw score than `unboosted` but matches a query
        # keyword, so it must be promoted ahead of `unboosted` by the boost
        # formula (score + 0.1*_boost).
        config = base_config()
        client = VectraClient(config)
        wire_common_mocks(client)

        # Without the boost, unboosted_doc (0.5) would rank above boosted_doc
        # (0.45). The keyword match on "widget" adds 0.1*1=0.1 to
        # boosted_doc's sort key (0.55), flipping the order.
        boosted_doc = {
            "content": "BOOSTED_MARKER document",
            "score": 0.45,
            "metadata": {"keywords": ["widget"]},
        }
        unboosted_doc = {
            "content": "UNBOOSTED_MARKER document",
            "score": 0.5,
            "metadata": {},
        }
        client.vector_store.similarity_search = AsyncMock(return_value=[unboosted_doc, boosted_doc])

        await client.query_rag("widget query")

        prompt = prompt_sent_to_llm(client)
        assert prompt.index("BOOSTED_MARKER") < prompt.index("UNBOOSTED_MARKER")
