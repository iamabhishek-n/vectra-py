import json
from vectra.context_layer import build_context, _clear_token_cache


class TestBuildContextTools:
    def setup_method(self):
        _clear_token_cache()

    async def test_packs_precomputed_tool_results_into_parts(self):
        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "tools", "results": [{"name": "get_weather", "output": "Sunny, 22C"}]}],
        })

        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "tools"
        assert "get_weather" in result["parts"][0]["content"]
        assert "Sunny, 22C" in result["parts"][0]["content"]

    async def test_does_not_execute_anything_results_used_verbatim(self):
        output = json.dumps({"note": "this is data, not a function"})
        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "tools", "results": [{"name": "x", "output": output}]}],
        })
        assert "this is data, not a function" in result["parts"][0]["content"]

    async def test_honestly_drops_tool_results_that_dont_fit(self):
        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 2},
            "sources": [{"type": "tools", "results": [{"name": "x", "output": "word " * 50}]}],
        })
        assert result["parts"] == []
        assert len(result["dropped"]) == 1
        assert result["dropped"][0]["source"] == "tools"
