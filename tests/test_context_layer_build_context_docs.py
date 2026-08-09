from vectra.context_layer import build_context, _clear_token_cache


class TestBuildContextDocs:
    def setup_method(self):
        _clear_token_cache()

    async def test_packs_doc_content_into_parts_within_budget(self):
        result = await build_context({
            "query": "what is vectra?",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "items": [
                {"content": "Vectra is a RAG orchestration SDK.", "metadata": {"source": "readme.md"}},
            ]}],
        })

        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "docs"
        assert "Vectra is a RAG orchestration SDK." in result["parts"][0]["content"]
        assert result["parts"][0]["tokens"] > 0
        assert "Vectra is a RAG orchestration SDK." in result["text"]
        assert result["tokens_used"] > 0
        assert result["tokens_budget"] == 1000
        assert result["dropped"] == []

    async def test_honestly_reports_dropped_items_when_budget_exceeded(self):
        long_content = "word " * 200
        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 10},
            "sources": [{"type": "docs", "items": [
                {"content": long_content, "metadata": {"source": "a.md"}},
                {"content": "short", "metadata": {"source": "b.md"}},
            ]}],
        })

        assert len(result["dropped"]) > 0
        assert result["dropped"][0]["source"] == "docs"
        assert len(result["parts"]) + len(result["dropped"]) == 2

    async def test_returns_empty_result_for_no_sources(self):
        result = await build_context({"query": "q", "budget": {"max_tokens": 100}, "sources": []})
        assert result["parts"] == []
        assert result["text"] == ""
        assert result["tokens_used"] == 0
