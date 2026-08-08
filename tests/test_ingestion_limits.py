import pytest
from unittest.mock import AsyncMock, MagicMock
from vectra.core import VectraClient
from vectra.config import VectraConfig, EmbeddingConfig, LLMConfig, DatabaseConfig, ProviderType, IngestionConfig


def make_client(max_file_size_bytes=None):
    kwargs = {}
    if max_file_size_bytes is not None:
        kwargs["ingestion"] = IngestionConfig(max_file_size_bytes=max_file_size_bytes)
    # ChromaVectorStore.__init__ eagerly calls client_instance.get_or_create_collection(...),
    # so a bare object() doesn't satisfy it. Use MagicMock(), matching
    # tests/test_core_guardrails.py and tests/test_backends/test_chroma_store.py.
    return VectraClient(VectraConfig(
        embedding=EmbeddingConfig(provider=ProviderType.OPENAI, api_key="test-key"),
        llm=LLMConfig(provider=ProviderType.OPENAI, api_key="test-key", model_name="gpt-4o-mini"),
        database=DatabaseConfig(type="chroma", client_instance=MagicMock()),
        **kwargs,
    ))


class TestIngestionFileSizeLimit:
    async def test_rejects_file_over_configured_limit(self, tmp_path):
        big_file = tmp_path / "big.txt"
        big_file.write_bytes(b"x" * 2000)
        client = make_client(max_file_size_bytes=1000)
        with pytest.raises(Exception, match="File exceeds maximum allowed size"):
            await client.ingest_batch([str(big_file)])

    async def test_accepts_file_within_limit(self, tmp_path):
        small_file = tmp_path / "small.txt"
        small_file.write_text("hello world")
        client = make_client(max_file_size_bytes=1000)
        # Stub out everything past the size check so this test only exercises the guard.
        client.processor.load_document = AsyncMock(return_value="hello world")
        client.processor.process = AsyncMock(return_value=["hello world"])
        client.processor.compute_chunk_metadata = MagicMock(return_value=[{}])
        client.embedder.embed_documents = AsyncMock(return_value=[[0.1]])
        client.vector_store.add_documents = AsyncMock()
        await client.ingest_batch([str(small_file)])  # should not raise for size reasons

    async def test_uses_default_50mb_limit_when_not_configured(self, tmp_path):
        client = make_client()
        assert client.config.ingestion.max_file_size_bytes == 52428800

    async def test_rejects_all_files_when_configured_limit_is_zero(self, tmp_path):
        # Regression test: max_file_size_bytes=0 must reject every file, including a
        # 1-byte file. A truthy `or` fallback (`max_file_size_bytes or 52428800`) would
        # silently treat 0 as "not configured" and fall back to the 50MB default instead
        # of rejecting everything, so this must be checked with an explicit type/None
        # check rather than truthiness.
        tiny_file = tmp_path / "tiny.txt"
        tiny_file.write_bytes(b"x")
        client = make_client(max_file_size_bytes=0)
        with pytest.raises(Exception, match="File exceeds maximum allowed size"):
            await client.ingest_batch([str(tiny_file)])
