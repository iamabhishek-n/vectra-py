from unittest.mock import AsyncMock
from vectra.context_layer import build_context, _clear_token_cache


def fake_fact_store(facts):
    store = type("FS", (), {})()
    store.read = AsyncMock(return_value=facts)
    return store


class TestBuildContextMemory:
    def setup_method(self):
        _clear_token_cache()

    async def test_packs_facts_from_fact_store_read_counted_against_budget(self):
        fact_store = fake_fact_store([{"subject": "user", "predicate": "likes", "object": "coffee"}])

        result = await build_context({
            "query": "what does the user like",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "memory", "fact_store": fact_store, "session_id": "session-1"}],
        })

        fact_store.read.assert_called_once_with("session-1", "what does the user like")
        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "memory"
        assert "coffee" in result["parts"][0]["content"]
        assert result["tokens_used"] > 0

    async def test_memory_competes_for_budget_with_docs_real_regression_test(self):
        long_doc = "word " * 50
        fact_store = fake_fact_store([{"subject": "user", "predicate": "likes", "object": "coffee"}])

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 5},
            "sources": [
                {"type": "docs", "items": [{"content": long_doc, "metadata": {}}]},
                {"type": "memory", "fact_store": fact_store, "session_id": "session-1"},
            ],
        })

        assert any(d["source"] == "docs" for d in result["dropped"])

    async def test_skips_memory_cleanly_when_no_fact_store_or_session_id(self):
        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 100},
            "sources": [{"type": "memory", "fact_store": None, "session_id": "session-1"}],
        })
        assert result["parts"] == []
