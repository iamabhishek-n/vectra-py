import pytest
from unittest.mock import AsyncMock
from vectra.core import VectraClient
from vectra.config import VectraConfig, EmbeddingConfig, LLMConfig, DatabaseConfig


def make_config(memory=None, guardrails=None, context_layer=None):
    kwargs = dict(
        embedding=EmbeddingConfig(provider="openai", api_key="test-key", model_name="text-embedding-3-small"),
        llm=LLMConfig(provider="openai", api_key="test-key", model_name="gpt-4o-mini"),
        database=DatabaseConfig(type="postgres", client_instance=object()),
        memory=memory,
    )
    if guardrails is not None:
        kwargs["guardrails"] = guardrails
    if context_layer is not None:
        kwargs["context_layer"] = context_layer
    return VectraConfig(**kwargs)


class TestContextAsk:
    async def test_embeds_query_retrieves_docs_returns_packed_context(self):
        client = VectraClient(make_config())
        client.embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
        client.vector_store.similarity_search = AsyncMock(return_value=[
            {"content": "Vectra is a RAG SDK.", "metadata": {"source": "readme.md"}},
        ])

        result = await client.context.ask("what is vectra?")

        client.embedder.embed_query.assert_called_with("what is vectra?")
        client.vector_store.similarity_search.assert_called()
        assert any(p["type"] == "docs" for p in result["parts"])
        assert "Vectra is a RAG SDK." in result["text"]

    async def test_includes_memory_when_session_id_given_and_fact_store_configured(self):
        client = VectraClient(make_config(memory={"enabled": True, "facts": {"enabled": True, "client_instance": object(), "table_name": "F"}}))
        client.embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
        client.vector_store.similarity_search = AsyncMock(return_value=[])
        client.fact_store.read = AsyncMock(return_value=[{"subject": "user", "predicate": "likes", "object": "coffee"}])

        result = await client.context.ask("what does the user like", session_id="session-1")

        client.fact_store.read.assert_called_with("session-1", "what does the user like")
        assert any(p["type"] == "memory" for p in result["parts"])

    async def test_skips_memory_cleanly_when_no_session_id(self):
        client = VectraClient(make_config(memory={"enabled": True, "facts": {"enabled": True, "client_instance": object(), "table_name": "F"}}))
        client.embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
        client.vector_store.similarity_search = AsyncMock(return_value=[])
        client.fact_store.read = AsyncMock()

        await client.context.ask("q")

        client.fact_store.read.assert_not_called()

    async def test_includes_tool_results_when_given(self):
        client = VectraClient(make_config())
        client.embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
        client.vector_store.similarity_search = AsyncMock(return_value=[])

        result = await client.context.ask("q", tools=[{"name": "get_weather", "output": "Sunny"}])

        assert any(p["type"] == "tools" for p in result["parts"])

    async def test_enforces_guardrails_before_any_embedding_call(self):
        client = VectraClient(make_config(guardrails={"max_query_length": 10}))
        client.embedder.embed_query = AsyncMock()

        with pytest.raises(Exception, match="GuardrailViolation"):
            await client.context.ask("this query is way too long for the configured limit")
        client.embedder.embed_query.assert_not_called()

    async def test_runs_on_before_retrieve_middleware(self):
        client = VectraClient(make_config())
        client.embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
        client.vector_store.similarity_search = AsyncMock(return_value=[])
        client._run_middlewares = AsyncMock(side_effect=lambda name, *args: args)

        await client.context.ask("q")

        client._run_middlewares.assert_any_call("on_before_retrieve", "q", [0.1, 0.2])

    async def test_applies_custom_context_layer_budget_through_real_config_parsing(self):
        # Regression guard: VectraConfig previously had no context_layer field
        # declared, so Pydantic's default extra='ignore' silently dropped it
        # during construction. A caller setting context_layer['budget']
        # never actually changed the packing budget, context.ask always fell
        # back to the hardcoded 2048 default. This goes through VectraConfig's
        # real construction, not a hand-built dict, so it fails if the field
        # regresses to being dropped.
        client = VectraClient(make_config(context_layer={"budget": {"max_tokens": 12}}))
        client.embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
        client.vector_store.similarity_search = AsyncMock(return_value=[
            {"content": "a" * 500, "metadata": {}},
        ])

        result = await client.context.ask("q")

        assert result["tokens_budget"] == 12
