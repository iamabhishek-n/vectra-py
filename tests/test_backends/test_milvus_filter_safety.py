import pytest
from vectra.backends.milvus_store import MilvusVectorStore


def make_config():
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": object()})()


class TestMilvusFilterSafety:
    def test_builds_normal_expression_for_safe_filter(self):
        store = MilvusVectorStore(make_config())
        expr = store._filter_to_expr({"category": "docs", "count": 3, "active": True})
        assert expr == 'metadata["category"] == "docs" and metadata["count"] == 3 and metadata["active"] == true'

    def test_returns_empty_string_for_no_filter(self):
        store = MilvusVectorStore(make_config())
        assert store._filter_to_expr(None) == ""
        assert store._filter_to_expr({}) == ""

    def test_escapes_double_quotes_in_string_value(self):
        store = MilvusVectorStore(make_config())
        expr = store._filter_to_expr({"category": 'docs" or 1==1 or "'})
        assert expr == 'metadata["category"] == "docs\\" or 1==1 or \\""'
        # The escaped quote must not terminate the string literal early.
        assert expr.count('"') % 2 == 0

    def test_rejects_unsafe_filter_key(self):
        store = MilvusVectorStore(make_config())
        with pytest.raises(ValueError, match="Unsafe filter key"):
            store._filter_to_expr({'category"] == "x" or metadata["injected': "docs"})

    def test_rejects_unsupported_filter_value_type(self):
        # Regression test: an unsupported value type (e.g. a list) must fail
        # loudly instead of being silently skipped. Silently dropping it
        # could turn a caller's intended scoped filter into an unfiltered
        # query that returns every document (a fail-open, multi-tenant-unsafe
        # bug) instead of raising.
        store = MilvusVectorStore(make_config())
        with pytest.raises(ValueError, match="Unsupported filter value type"):
            store._filter_to_expr({"tags": ["a", "b"]})
