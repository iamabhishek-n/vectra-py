from unittest.mock import AsyncMock
from vectra.core import VectraClient
from vectra.config import VectraConfig, ProviderType, EmbeddingConfig, LLMConfig, DatabaseConfig
from vectra.backends.pinecone_store import PineconeVectorStore


class TestCreateVectorStorePinecone:
    def test_returns_pinecone_vector_store_for_type_pinecone(self):
        config = VectraConfig(
            embedding=EmbeddingConfig(provider=ProviderType.OPENAI, api_key="test-key"),
            llm=LLMConfig(provider=ProviderType.OPENAI, api_key="test-key", model_name="gpt-4o-mini"),
            database=DatabaseConfig(type="pinecone", client_instance=AsyncMock()),
        )

        client = VectraClient(config)

        assert isinstance(client.vector_store, PineconeVectorStore)
