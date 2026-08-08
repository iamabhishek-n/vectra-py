import pytest
from unittest.mock import AsyncMock, MagicMock
from vectra.core import VectraClient
from vectra.config import VectraConfig, EmbeddingConfig, LLMConfig, DatabaseConfig, ProviderType, GuardrailConfig


def make_config(guardrails):
    # ChromaVectorStore.__init__ eagerly calls client_instance.get_or_create_collection(...),
    # so a bare object() (as used elsewhere for a lightweight non-chroma sentinel) doesn't
    # satisfy it. Use MagicMock() here, matching tests/test_backends/test_chroma_store.py's
    # pattern, so construction succeeds without needing a real chromadb client.
    return VectraConfig(
        embedding=EmbeddingConfig(provider=ProviderType.OPENAI, api_key="test-key"),
        llm=LLMConfig(provider=ProviderType.OPENAI, api_key="test-key", model_name="gpt-4o-mini"),
        database=DatabaseConfig(type="chroma", client_instance=MagicMock()),
        guardrails=guardrails,
    )


class TestQueryRagGuardrailEnforcement:
    async def test_rejects_over_length_query_before_any_embedding_call(self):
        client = VectraClient(make_config(GuardrailConfig(max_query_length=10)))
        client.embedder.embed_query = AsyncMock()
        with pytest.raises(ValueError, match="GuardrailViolation: query exceeds max_query_length"):
            await client.query_rag("this query is way too long for the limit")
        client.embedder.embed_query.assert_not_called()

    async def test_rejects_pii_query_before_any_embedding_call(self):
        client = VectraClient(make_config(GuardrailConfig(block_pii=True)))
        client.embedder.embed_query = AsyncMock()
        with pytest.raises(ValueError, match="GuardrailViolation: possible PII detected"):
            await client.query_rag("email me at test@example.com")
        client.embedder.embed_query.assert_not_called()
