import pytest
from pydantic import ValidationError
from vectra.config import VectraConfig, EmbeddingConfig, LLMConfig, DatabaseConfig, ProviderType


def make_minimal_config(**overrides):
    base = dict(
        embedding=EmbeddingConfig(provider=ProviderType.OPENAI, api_key="test-key"),
        llm=LLMConfig(provider=ProviderType.OPENAI, api_key="test-key", model_name="gpt-4o-mini"),
        database=DatabaseConfig(type="chroma", client_instance=object()),
    )
    base.update(overrides)
    return base


class TestVectraConfig:
    def test_accepts_minimal_valid_config_and_fills_defaults(self):
        parsed = VectraConfig(**make_minimal_config())
        assert parsed.embedding.model_name == "text-embedding-3-small"
        assert parsed.telemetry.enabled is False
        assert parsed.database.column_map == {"content": "content", "vector": "vector", "metadata": "metadata"}

    def test_rejects_config_missing_embedding_provider(self):
        with pytest.raises(ValidationError):
            VectraConfig(**make_minimal_config(embedding={"api_key": "test-key"}))

    def test_rejects_config_missing_database_type(self):
        with pytest.raises(ValidationError):
            VectraConfig(**make_minimal_config(database={"client_instance": object()}))

    def test_rejects_agentic_chunking_with_no_agentic_llm(self):
        with pytest.raises(ValidationError, match="agentic_llm required"):
            VectraConfig(**make_minimal_config(chunking={"strategy": "agentic"}))

    def test_rejects_hyde_retrieval_with_no_llm_config(self):
        with pytest.raises(ValidationError, match="llm_config required"):
            VectraConfig(**make_minimal_config(retrieval={"strategy": "hyde"}))
