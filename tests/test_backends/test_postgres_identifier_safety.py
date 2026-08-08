import pytest
from vectra.backends.postgres_store import (
    PostgresVectorStore, is_safe_identifier, assert_safe_identifier,
)


def make_config(table_name="document", column_map=None):
    return type("Cfg", (), {
        "table_name": table_name,
        "column_map": column_map or {},
        "client_instance": object(),
    })()


class TestIdentifierSafety:
    def test_accepts_plain_alphanumeric_identifier(self):
        assert is_safe_identifier("content") is True
        assert is_safe_identifier("_private_col") is True

    def test_rejects_identifier_with_sql_metacharacters(self):
        assert is_safe_identifier('content"; DROP TABLE users; --') is False
        assert is_safe_identifier("content' OR '1'='1") is False
        assert is_safe_identifier("content column") is False

    def test_rejects_identifier_starting_with_digit(self):
        assert is_safe_identifier("1content") is False

    def test_assert_safe_identifier_returns_value_when_safe(self):
        assert assert_safe_identifier("content", "test") == "content"

    def test_assert_safe_identifier_raises_when_unsafe(self):
        with pytest.raises(ValueError, match="Unsafe SQL identifier"):
            assert_safe_identifier('a"; DROP TABLE x; --', "test")


class TestPostgresVectorStoreConstructionValidation:
    def test_accepts_safe_table_name_and_columns(self):
        store = PostgresVectorStore(make_config(
            table_name="documents",
            column_map={"content": "body", "metadata": "meta", "vector": "embedding"},
        ))
        assert store.table_name == "documents"
        assert store.c_content == "body"

    def test_rejects_unsafe_table_name_at_construction(self):
        with pytest.raises(ValueError, match="Unsafe SQL identifier"):
            PostgresVectorStore(make_config(table_name='documents"; DROP TABLE x; --'))

    def test_rejects_unsafe_column_name_at_construction(self):
        with pytest.raises(ValueError, match="Unsafe SQL identifier"):
            PostgresVectorStore(make_config(column_map={"content": 'c"; DROP TABLE x; --'}))
