import pytest
from unittest.mock import patch
from vectra.reranker import CrossEncoderReranker
from vectra.config import RerankingConfig, RerankingProvider


def make_docs(n):
    return [{"content": f"doc {i}", "metadata": {}, "score": 1 - i * 0.01} for i in range(n)]


class MockResponse:
    def __init__(self, status, json_data):
        self.status = status
        self._json = json_data

    async def json(self):
        return self._json

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockSession:
    def __init__(self, response):
        self._response = response
        self.post_calls = []

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class TestCohereReranker:
    async def test_calls_cohere_rerank_api_and_reorders_by_relevance(self):
        docs = make_docs(3)
        response = MockResponse(200, {"results": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.5}]})
        session = MockSession(response)
        config = RerankingConfig(provider=RerankingProvider.COHERE, api_key="test-key", top_n=2)
        reranker = CrossEncoderReranker(config)

        with patch("vectra.reranker.aiohttp.ClientSession", return_value=session):
            result = await reranker.rerank("a query", docs)

        assert result == [docs[2], docs[0]]
        assert len(session.post_calls) == 1
        url, kwargs = session.post_calls[0]
        assert url == "https://api.cohere.com/v2/rerank"
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["json"]["query"] == "a query"
        assert kwargs["json"]["documents"] == ["doc 0", "doc 1", "doc 2"]
        assert kwargs["json"]["top_n"] == 2

    async def test_falls_back_to_original_order_with_no_api_key(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        docs = make_docs(2)
        config = RerankingConfig(provider=RerankingProvider.COHERE, top_n=2)
        reranker = CrossEncoderReranker(config)

        result = await reranker.rerank("q", docs)

        assert result == docs

    async def test_falls_back_to_original_order_on_non_200_response(self):
        docs = make_docs(3)
        response = MockResponse(500, {})
        session = MockSession(response)
        config = RerankingConfig(provider=RerankingProvider.COHERE, api_key="test-key", top_n=2)
        reranker = CrossEncoderReranker(config)

        with patch("vectra.reranker.aiohttp.ClientSession", return_value=session):
            result = await reranker.rerank("q", docs)

        assert result == docs[:2]
