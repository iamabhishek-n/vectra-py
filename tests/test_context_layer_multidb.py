from unittest.mock import AsyncMock
from vectra.context_layer import build_context, _clear_token_cache


def make_store(results):
    store = type("Store", (), {})()
    store.similarity_search = AsyncMock(return_value=results)
    return store


class TestBuildContextMultiDb:
    def setup_method(self):
        _clear_token_cache()

    async def test_fans_out_to_all_stores_concurrently_and_fuses_results(self):
        store_a = make_store([{"content": "doc from A", "metadata": {}, "score": 0.9}])
        store_b = make_store([{"content": "doc from B", "metadata": {}, "score": 0.8}])

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "stores": [store_a, store_b], "vector": [0.1, 0.2], "limit": 5}],
        })

        store_a.similarity_search.assert_called_with([0.1, 0.2], 5, None)
        store_b.similarity_search.assert_called_with([0.1, 0.2], 5, None)
        contents = [p["content"] for p in result["parts"]]
        assert "doc from A" in contents
        assert "doc from B" in contents

    async def test_doc_findable_only_in_one_store_survives_fusion(self):
        store_a = make_store([
            {"content": "shared doc", "metadata": {}, "score": 0.5},
            {"content": "only in A", "metadata": {}, "score": 0.4},
        ])
        store_b = make_store([{"content": "shared doc", "metadata": {}, "score": 0.5}])

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "stores": [store_a, store_b], "vector": [0.1], "limit": 5}],
        })

        contents = [p["content"] for p in result["parts"]]
        assert "only in A" in contents
        assert "shared doc" in contents

    async def test_dedupes_by_content_across_stores(self):
        store_a = make_store([{"content": "dup", "metadata": {}, "score": 0.9}])
        store_b = make_store([{"content": "dup", "metadata": {}, "score": 0.9}])

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "stores": [store_a, store_b], "vector": [0.1], "limit": 5}],
        })

        assert len([p for p in result["parts"] if p["content"] == "dup"]) == 1

    async def test_uses_hybrid_search_when_strategy_is_hybrid(self):
        store = type("Store", (), {})()
        store.similarity_search = AsyncMock()
        store.hybrid_search = AsyncMock(return_value=[{"content": "hybrid result", "metadata": {}, "score": 0.9}])

        await build_context({
            "query": "the query text",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "stores": [store], "vector": [0.1], "limit": 5, "strategy": "hybrid"}],
        })

        store.hybrid_search.assert_called_with("the query text", [0.1], 5, None)
        store.similarity_search.assert_not_called()
