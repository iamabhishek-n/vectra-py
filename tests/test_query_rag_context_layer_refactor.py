from unittest.mock import AsyncMock
from vectra.core import VectraClient
from vectra.config import VectraConfig, EmbeddingConfig, LLMConfig, DatabaseConfig


def make_config():
    return VectraConfig(
        embedding=EmbeddingConfig(provider="openai", api_key="test-key", model_name="text-embedding-3-small"),
        llm=LLMConfig(provider="openai", api_key="test-key", model_name="gpt-4o-mini"),
        database=DatabaseConfig(type="postgres", client_instance=object()),
        memory={"enabled": True, "type": "in-memory"},
        query_planning={"token_budget": 15},
    )


class TestQueryRagHistoryBudget:
    async def test_long_history_no_longer_bypasses_the_budget_uncounted(self):
        client = VectraClient(make_config())
        client.embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
        client.vector_store.similarity_search = AsyncMock(return_value=[{"content": "short doc", "metadata": {}}])
        client.llm.generate = AsyncMock(return_value="an answer")

        for i in range(20):
            client.history.add_message("session-1", "user", f"This is a fairly long historical message number {i} with a lot of extra real words padded in to make it substantial and unmistakably large for the purposes of this specific regression test.")

        await client.query_rag("what is this?", None, False, "session-1")

        prompt_sent = client.llm.generate.call_args[0][0]
        prompt_token_estimate = client._token_estimate(prompt_sent)

        assert prompt_token_estimate < 150
