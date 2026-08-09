import pytest
from unittest.mock import AsyncMock, MagicMock
from vectra.core import VectraClient, RetrievalStrategy
from vectra.config import (
    VectraConfig,
    EmbeddingConfig,
    LLMConfig,
    DatabaseConfig,
    RetrievalConfig,
    RerankingConfig,
    ProviderType,
)


def make_config_with_mmr(fetch_k, window_size, mmr_lambda=0.5):
    """Create a VectraConfig with MMR retrieval strategy.

    window_size sets k (the number of documents to retrieve), which is determined
    by reranking.window_size in query_rag().
    """
    return VectraConfig(
        embedding=EmbeddingConfig(provider=ProviderType.OPENAI, api_key="test-key"),
        llm=LLMConfig(provider=ProviderType.OPENAI, api_key="test-key", model_name="gpt-4o-mini"),
        database=DatabaseConfig(type="chroma", client_instance=MagicMock()),
        retrieval=RetrievalConfig(
            strategy=RetrievalStrategy.MMR,
            mmr_fetch_k=fetch_k,
            mmr_lambda=mmr_lambda,
        ),
        reranking=RerankingConfig(
            enabled=True,
            window_size=window_size,
        ),
    )


class TestQueryRagMMRShortCircuit:
    @pytest.mark.asyncio
    async def test_skips_embed_documents_when_fetch_k_equals_k(self):
        """When fetch_k == k, MMR returns all candidates, so embedding is unnecessary."""
        config = make_config_with_mmr(fetch_k=20, window_size=20)
        client = VectraClient(config)

        # Mock embedder and vector store
        client.embedder.embed_query = AsyncMock(return_value=[0.1] * 10)
        client.embedder.embed_documents = AsyncMock()

        mock_candidates = [
            {"content": "doc 1", "score": 0.9, "metadata": {}},
            {"content": "doc 2", "score": 0.8, "metadata": {}},
            {"content": "doc 3", "score": 0.7, "metadata": {}},
        ]
        client.vector_store.similarity_search = AsyncMock(return_value=mock_candidates)

        # Mock LLM for generation
        client.llm = MagicMock()
        client.llm.generate = AsyncMock(return_value="response")

        # Mock guardrails
        client.config.guardrails = None

        # Execute query
        await client.query_rag("test query")

        # Assert embed_documents was NOT called since fetch_k <= k
        client.embedder.embed_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_embed_documents_when_fetch_k_less_than_k(self):
        """When fetch_k < k, MMR returns all candidates, so embedding is unnecessary."""
        config = make_config_with_mmr(fetch_k=10, window_size=20)
        client = VectraClient(config)

        # Mock embedder and vector store
        client.embedder.embed_query = AsyncMock(return_value=[0.1] * 10)
        client.embedder.embed_documents = AsyncMock()

        mock_candidates = [
            {"content": "doc 1", "score": 0.9, "metadata": {}},
            {"content": "doc 2", "score": 0.8, "metadata": {}},
        ]
        client.vector_store.similarity_search = AsyncMock(return_value=mock_candidates)

        # Mock LLM for generation
        client.llm = MagicMock()
        client.llm.generate = AsyncMock(return_value="response")

        # Mock guardrails
        client.config.guardrails = None

        # Execute query
        await client.query_rag("test query")

        # Assert embed_documents was NOT called since fetch_k <= k
        client.embedder.embed_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_embed_documents_when_fetch_k_greater_than_k(self):
        """When fetch_k > k, MMR needs embeddings for diversity selection."""
        config = make_config_with_mmr(fetch_k=30, window_size=20)
        client = VectraClient(config)

        # Mock embedder and vector store
        client.embedder.embed_query = AsyncMock(return_value=[0.1] * 10)
        mock_embeddings = [[0.1] * 10, [0.2] * 10, [0.3] * 10]
        client.embedder.embed_documents = AsyncMock(return_value=mock_embeddings)

        mock_candidates = [
            {"content": "doc 1", "score": 0.9, "metadata": {}},
            {"content": "doc 2", "score": 0.8, "metadata": {}},
            {"content": "doc 3", "score": 0.7, "metadata": {}},
        ]
        client.vector_store.similarity_search = AsyncMock(return_value=mock_candidates)

        # Mock LLM for generation
        client.llm = MagicMock()
        client.llm.generate = AsyncMock(return_value="response")

        # Mock guardrails
        client.config.guardrails = None

        # Execute query
        await client.query_rag("test query")

        # Assert embed_documents WAS called since fetch_k > k
        client.embedder.embed_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_config_does_not_embed(self):
        """Default config has fetch_k=20 and window_size=20, so no embedding should occur."""
        config = VectraConfig(
            embedding=EmbeddingConfig(provider=ProviderType.OPENAI, api_key="test-key"),
            llm=LLMConfig(provider=ProviderType.OPENAI, api_key="test-key", model_name="gpt-4o-mini"),
            database=DatabaseConfig(type="chroma", client_instance=MagicMock()),
            retrieval=RetrievalConfig(strategy=RetrievalStrategy.MMR),
            reranking=RerankingConfig(enabled=True),
        )
        client = VectraClient(config)

        # Mock embedder and vector store
        client.embedder.embed_query = AsyncMock(return_value=[0.1] * 10)
        client.embedder.embed_documents = AsyncMock()

        mock_candidates = [
            {"content": "doc 1", "score": 0.9, "metadata": {}},
            {"content": "doc 2", "score": 0.8, "metadata": {}},
        ]
        client.vector_store.similarity_search = AsyncMock(return_value=mock_candidates)

        # Mock LLM for generation
        client.llm = MagicMock()
        client.llm.generate = AsyncMock(return_value="response")

        # Mock guardrails
        client.config.guardrails = None

        # Execute query
        await client.query_rag("test query")

        # With default config (fetch_k=20, window_size=20), embed_documents should NOT be called
        client.embedder.embed_documents.assert_not_called()
