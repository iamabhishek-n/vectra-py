import asyncio
import time
from unittest.mock import AsyncMock
from vectra.context_layer import build_context, _clear_token_cache


def slow_store(delay_s, result):
    async def _search(*args, **kwargs):
        await asyncio.sleep(delay_s)
        return result
    store = type("Store", (), {})()
    store.similarity_search = _search
    return store


def throwing_store():
    store = type("Store", (), {})()
    store.similarity_search = AsyncMock(side_effect=Exception("connection refused"))
    return store


class TestBuildContextMultiDbTimeout:
    def setup_method(self):
        _clear_token_cache()

    async def test_a_store_that_throws_produces_a_warning_not_a_failed_call(self):
        good_store = type("Store", (), {})()
        good_store.similarity_search = AsyncMock(return_value=[{"content": "good doc", "metadata": {}, "score": 0.9}])
        bad_store = throwing_store()

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "stores": [good_store, bad_store], "vector": [0.1], "limit": 5}],
        })

        assert any(p["content"] == "good doc" for p in result["parts"])
        assert len(result["warnings"]) > 0
        assert "connection refused" in result["warnings"][0]["error"]

    async def test_a_store_that_exceeds_timeout_produces_a_warning_does_not_hang(self):
        fast_store = type("Store", (), {})()
        fast_store.similarity_search = AsyncMock(return_value=[{"content": "fast doc", "metadata": {}, "score": 0.9}])
        hanging_store = slow_store(10, [{"content": "never arrives", "metadata": {}, "score": 0.9}])

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "stores": [fast_store, hanging_store], "vector": [0.1], "limit": 5, "timeout_s": 0.05}],
        })

        assert any(p["content"] == "fast doc" for p in result["parts"])
        assert not any(p["content"] == "never arrives" for p in result["parts"])
        assert any("timeout" in str(w["error"]).lower() for w in result["warnings"])

    async def test_total_wallclock_bounded_by_slowest_allowed_store_not_the_sum(self):
        store_a = slow_store(0.03, [{"content": "a", "metadata": {}, "score": 0.9}])
        store_b = slow_store(0.03, [{"content": "b", "metadata": {}, "score": 0.9}])
        store_c = slow_store(0.03, [{"content": "c", "metadata": {}, "score": 0.9}])

        start = time.monotonic()
        await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "stores": [store_a, store_b, store_c], "vector": [0.1], "limit": 5}],
        })
        elapsed = time.monotonic() - start

        assert elapsed < 0.08
