from unittest.mock import patch
from vectra.reranker import CrossEncoderReranker
from vectra.config import RerankingConfig, RerankingProvider
from tests.test_reranker_cohere import MockResponse, MockSession, make_docs


class TestJinaReranker:
    async def test_calls_jina_rerank_api_and_reorders_by_relevance(self):
        docs = make_docs(3)
        response = MockResponse(200, {"results": [{"index": 1, "relevance_score": 0.88}, {"index": 0, "relevance_score": 0.4}]})
        session = MockSession(response)
        config = RerankingConfig(provider=RerankingProvider.JINA, api_key="test-key", top_n=2)
        reranker = CrossEncoderReranker(config)

        with patch("vectra.reranker.aiohttp.ClientSession", return_value=session):
            result = await reranker.rerank("a query", docs)

        assert result == [docs[1], docs[0]]
        url, kwargs = session.post_calls[0]
        assert url == "https://api.jina.ai/v1/rerank"
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["json"]["documents"] == ["doc 0", "doc 1", "doc 2"]
        assert kwargs["json"]["top_n"] == 2

    async def test_falls_back_to_original_order_on_non_200_response(self):
        docs = make_docs(3)
        response = MockResponse(429, {})
        session = MockSession(response)
        config = RerankingConfig(provider=RerankingProvider.JINA, api_key="test-key", top_n=2)
        reranker = CrossEncoderReranker(config)

        with patch("vectra.reranker.aiohttp.ClientSession", return_value=session):
            result = await reranker.rerank("q", docs)

        assert result == docs[:2]

    async def test_falls_back_to_original_order_with_no_api_key(self, monkeypatch):
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        docs = make_docs(2)
        config = RerankingConfig(provider=RerankingProvider.JINA, top_n=2)
        reranker = CrossEncoderReranker(config)

        result = await reranker.rerank("q", docs)

        assert result == docs
