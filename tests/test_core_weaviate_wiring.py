from unittest.mock import AsyncMock
from vectra.core import VectraClient
from vectra.config import VectraConfig, ProviderType, EmbeddingConfig, LLMConfig, DatabaseConfig
from vectra.backends.weaviate_store import WeaviateVectorStore


class TestCreateVectorStoreWeaviate:
    def test_returns_weaviate_vector_store_for_type_weaviate(self):
        collection = type("Collection", (), {})()
        client = type("Client", (), {"collections": type("Collections", (), {"get": lambda self, name: collection})()})()

        config = VectraConfig(
            embedding=EmbeddingConfig(provider=ProviderType.OPENAI, api_key="test-key"),
            llm=LLMConfig(provider=ProviderType.OPENAI, api_key="test-key", model_name="gpt-4o-mini"),
            database=DatabaseConfig(type="weaviate", client_instance=client),
        )

        vectra_client = VectraClient(config)

        assert isinstance(vectra_client.vector_store, WeaviateVectorStore)
