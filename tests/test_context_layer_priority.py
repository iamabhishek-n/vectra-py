from unittest.mock import AsyncMock
from vectra.context_layer import build_context, _clear_token_cache


def fake_fact_store(facts):
    store = type("FS", (), {})()
    store.read = AsyncMock(return_value=facts)
    return store


class TestBuildContextPriority:
    def setup_method(self):
        _clear_token_cache()

    async def test_gives_earlier_priority_sources_first_claim_on_tight_budget(self):
        fact_store = fake_fact_store([{"subject": "user", "predicate": "likes", "object": "coffee"}])
        big_doc = "word " * 50

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 10},
            "sources": [
                {"type": "docs", "items": [{"content": big_doc, "metadata": {}}]},
                {"type": "memory", "fact_store": fact_store, "session_id": "s1"},
            ],
            "priority": ["memory", "docs"],
        })

        assert any(p["type"] == "memory" for p in result["parts"])
        assert not any(p["type"] == "docs" for p in result["parts"])
        assert any(d["source"] == "docs" for d in result["dropped"])

    async def test_unlisted_source_type_falls_after_listed_ones_not_before(self):
        fact_store = fake_fact_store([{"subject": "user", "predicate": "likes", "object": "coffee"}])

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [
                {"type": "tools", "results": [{"name": "x", "output": "y"}]},
                {"type": "memory", "fact_store": fact_store, "session_id": "s1"},
            ],
            "priority": ["memory", "docs"],
        })

        assert result["parts"][0]["type"] == "memory"
        assert result["parts"][1]["type"] == "tools"

    async def test_no_priority_processes_in_supplied_order(self):
        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [
                {"type": "tools", "results": [{"name": "x", "output": "y"}]},
                {"type": "docs", "items": [{"content": "doc content", "metadata": {}}]},
            ],
        })
        assert result["parts"][0]["type"] == "tools"
        assert result["parts"][1]["type"] == "docs"
