import pytest
from unittest.mock import AsyncMock
from vectra.backends.pinecone_store import PineconeVectorStore


def make_config(client):
    return type("Cfg", (), {"table_name": "ns", "client_instance": client, "dimensions": None})()


class TestPineconeVectorStoreHybridAndCrud:
    async def test_hybrid_search_fuses_semantic_and_lexical_rank(self):
        client = AsyncMock()
        client.query = AsyncMock(return_value={
            "matches": [
                {"id": "1", "score": 0.9, "metadata": {"content": "the quick brown fox"}},
                # Semantic score deliberately kept below "quick fox jumps high" so
                # the RRF fusion doesn't land on an exact tie broken by insertion
                # order (a real bug found and fixed in the vectra-js mirror of
                # this test — see pinecone_store.hybrid.test.js history).
                {"id": "2", "score": 0.8, "metadata": {"content": "a completely unrelated sentence"}},
                {"id": "3", "score": 0.85, "metadata": {"content": "quick fox jumps high"}},
            ]
        })
        store = PineconeVectorStore(make_config(client))

        results = await store.hybrid_search("quick fox", [0.1, 0.2], limit=2)

        assert len(results) == 2
        contents = [r["content"] for r in results]
        assert "a completely unrelated sentence" not in contents

    async def test_delete_documents_calls_native_filter_delete_and_returns_unknown_count(self):
        client = AsyncMock()
        store = PineconeVectorStore(make_config(client))

        result = await store.delete_documents({"source": "a.md"})

        client.delete.assert_called_once()
        assert result == -1

    async def test_delete_documents_requires_a_filter(self):
        client = AsyncMock()
        store = PineconeVectorStore(make_config(client))

        with pytest.raises(ValueError):
            await store.delete_documents({})

    async def test_list_documents_raises_not_implemented(self):
        store = PineconeVectorStore(make_config(AsyncMock()))

        with pytest.raises(NotImplementedError):
            await store.list_documents()

    async def test_update_documents_raises_not_implemented(self):
        store = PineconeVectorStore(make_config(AsyncMock()))

        with pytest.raises(NotImplementedError):
            await store.update_documents({"source": "a.md"}, {"content": "new"})

    async def test_file_exists_queries_by_metadata_filter_and_returns_true(self):
        client = AsyncMock()
        client.query = AsyncMock(return_value={"matches": [{"id": "1", "score": 1, "metadata": {}}]})
        store = PineconeVectorStore(make_config(client))

        exists = await store.file_exists("abc123", 100, 12345)

        assert exists is True

    async def test_file_exists_returns_false_when_no_match_found(self):
        client = AsyncMock()
        client.query = AsyncMock(return_value={"matches": []})
        store = PineconeVectorStore(make_config(client))

        exists = await store.file_exists("abc123", 100, 12345)

        assert exists is False
