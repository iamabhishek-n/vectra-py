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
