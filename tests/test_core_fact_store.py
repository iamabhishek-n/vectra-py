from vectra.core import VectraClient
from vectra.config import VectraConfig, EmbeddingConfig, LLMConfig, DatabaseConfig


def make_config(memory=None):
    return VectraConfig(
        embedding=EmbeddingConfig(provider="openai", api_key="test-key", model_name="text-embedding-3-small"),
        llm=LLMConfig(provider="openai", api_key="test-key", model_name="gpt-4o-mini"),
        database=DatabaseConfig(type="postgres", client_instance=object()),
        memory=memory,
    )


class TestMemoryFactsConfig:
    def test_fact_store_not_created_when_facts_not_enabled(self):
        client = VectraClient(make_config(memory={"enabled": False}))
        assert getattr(client, "fact_store", None) is None

    def test_fact_store_created_when_facts_enabled(self):
        client = VectraClient(make_config(memory={"enabled": True, "facts": {"enabled": True, "client_instance": object(), "table_name": "MyFacts"}}))
        assert client.fact_store is not None
        assert client.fact_store.table_name == "MyFacts"
        assert client.fact_store.llm is client.llm
        assert client.fact_store.embedder is client.embedder
