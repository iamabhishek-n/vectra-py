# Phase 3 — vectra-py Feature Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mirror vectra-js's Phase 3 in Python: real Cohere/Jina rerankers (currently mocked), real hybrid search on Chroma/Qdrant/Milvus (currently silent fallback to pure-vector search), the hardcoded 1536-dimension assumption in Postgres/Prisma, lexical-only MMR (switch to embedding-space when embeddings are available), and the character-count token estimator (switch to a real tokenizer).

**Architecture:** Same shape as the vectra-js sibling plan, adapted to Python idiom. Rerankers use `aiohttp` (already a project dependency, and async-appropriate — unlike `requests`, which is what `telemetry.py` uses but which would block the event loop inside these `async def rerank` methods). Hybrid search on the three fallback stores gains the identical client-side lexical-overlap + RRF fusion pattern already used by `postgres_store.py` (JS)/`prisma_store.py` (both languages). `_mmr_select` gains an embedding-space path that activates only when every candidate carries an `embedding` key, added by `query_rag`'s MMR branch via one batch `embed_documents` call — backward compatible, falls back to the existing lexical Jaccard path when embeddings aren't present.

**Tech Stack:** Python ≥3.8, pytest (already set up), `tiktoken` (new dependency).

## Global Constraints

- Local `RerankingProvider.CROSS_ENCODER` is explicitly OUT of scope — same reasoning as the JS sibling (needs a bundled ML runtime). Must raise a clear "not implemented" error instead of silently mocking.
- `RerankingConfig.api_key`/`model_name` already exist in `vectra/config.py` but are currently dead (never read by `reranker.py`) — this plan wires them up, no new schema fields.
- No direct-to-master commits, no force-push, no skipped hooks.
- Every task ends with `pytest` passing before moving to the next task.
- Do not touch guardrails, ingestion limits, SQL-identifier/filter-expression sanitization, or telemetry — Phase 1/2, already done.

---

### Task 1: Real Cohere reranker

**Files:**
- Modify: `vectra/reranker.py`
- Create: `tests/test_reranker_cohere.py`

**Interfaces:**
- Produces: `CrossEncoderReranker.rerank(query, documents)` now makes a real HTTP call to Cohere's rerank API when `config.provider == RerankingProvider.COHERE`, via `aiohttp`. Falls back to `documents[:config.top_n]` on any HTTP/parse error, matching `LLMReranker`'s existing fail-soft convention.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reranker_cohere.py`:

```python
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
```

- [ ] **Step 2: Run and verify they fail**

Run: `pytest tests/test_reranker_cohere.py -v`
Expected: FAIL — current `CrossEncoderReranker.rerank` calls `_mock_api_rerank`, never touches `aiohttp`, so the call-count/URL/body assertions fail.

- [ ] **Step 3: Implement the real Cohere call**

Edit `vectra/reranker.py`. Add `import os` and `import aiohttp` to the top imports (currently `import re`, `import json`, `from typing import ...`, `from .config import ...`):

```python
import re
import json
import os
import aiohttp
from typing import List, Dict, Any, Union
from .config import RerankingConfig, RerankingProvider
```

Replace the entire `CrossEncoderReranker` class (currently lines 53-77) with:

```python
class CrossEncoderReranker:
    """Dedicated reranker model (Cohere, Jina, or local Cross-Encoder)"""
    def __init__(self, config: RerankingConfig):
        self.config = config

    async def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []

        docs_to_rank = documents[:self.config.window_size]

        try:
            if self.config.provider == RerankingProvider.COHERE:
                return await self._cohere_rerank(query, docs_to_rank)
            if self.config.provider == RerankingProvider.JINA:
                return await self._jina_rerank(query, docs_to_rank)
            if self.config.provider == RerankingProvider.CROSS_ENCODER:
                raise NotImplementedError(
                    "RerankingProvider.CROSS_ENCODER (local model) is not implemented in vectra-py. "
                    "Use RerankingProvider.COHERE, RerankingProvider.JINA, or RerankingProvider.LLM instead."
                )
            return documents[:self.config.top_n]
        except NotImplementedError:
            raise
        except Exception:
            return docs_to_rank[:self.config.top_n]

    async def _cohere_rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        api_key = getattr(self.config, "api_key", None) or os.getenv("COHERE_API_KEY")
        if not api_key:
            return docs[:self.config.top_n]
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.cohere.com/v2/rerank",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json={
                    "model": getattr(self.config, "model_name", None) or "rerank-v3.5",
                    "query": query,
                    "documents": [d["content"] for d in docs],
                    "top_n": min(self.config.top_n, len(docs)),
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as res:
                if res.status != 200:
                    raise Exception(f"Cohere rerank API error: {res.status}")
                data = await res.json()
                return [docs[r["index"]] for r in data["results"]]

    async def _jina_rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Implemented in Task 2.
        return docs[:self.config.top_n]
```

- [ ] **Step 4: Run and verify all pass**

Run: `pytest tests/test_reranker_cohere.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Run the full suite**

Run: `pytest`
Expected: all tests pass. (Reranking has zero prior test coverage per the Phase 3 research pass, so no existing test depends on `_mock_api_rerank`.)

- [ ] **Step 6: Commit**

```bash
git add vectra/reranker.py tests/test_reranker_cohere.py
git commit -m "feat: implement real Cohere reranker via aiohttp"
```

---

### Task 2: Real Jina reranker

**Files:**
- Modify: `vectra/reranker.py`
- Create: `tests/test_reranker_jina.py`

**Interfaces:**
- Consumes: `CrossEncoderReranker` from Task 1.
- Produces: `_jina_rerank` now makes a real HTTP call to Jina AI's rerank API.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reranker_jina.py`:

```python
import pytest
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
```

*(This test file imports the `MockResponse`/`MockSession`/`make_docs` helpers from `tests/test_reranker_cohere.py` rather than duplicating them — confirm `tests/__init__.py` makes `tests` a real package, which Phase 1 already established.)*

- [ ] **Step 2: Run and verify they fail**

Run: `pytest tests/test_reranker_jina.py -v`
Expected: FAIL — `_jina_rerank` is still the Task 1 passthrough stub.

- [ ] **Step 3: Implement the real Jina call**

Edit `vectra/reranker.py`, replace the `_jina_rerank` stub:

```python
    async def _jina_rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Implemented in Task 2.
        return docs[:self.config.top_n]
```

with:

```python
    async def _jina_rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        api_key = getattr(self.config, "api_key", None) or os.getenv("JINA_API_KEY")
        if not api_key:
            return docs[:self.config.top_n]
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json={
                    "model": getattr(self.config, "model_name", None) or "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [d["content"] for d in docs],
                    "top_n": min(self.config.top_n, len(docs)),
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as res:
                if res.status != 200:
                    raise Exception(f"Jina rerank API error: {res.status}")
                data = await res.json()
                return [docs[r["index"]] for r in data["results"]]
```

- [ ] **Step 4: Run and verify all pass**

Run: `pytest tests/test_reranker_jina.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Run the full suite**

Run: `pytest`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add vectra/reranker.py tests/test_reranker_jina.py
git commit -m "feat: implement real Jina reranker via aiohttp"
```

---

### Task 3: Real hybrid search — ChromaVectorStore

**Files:**
- Modify: `vectra/backends/chroma_store.py`
- Create: `tests/test_backends/test_chroma_hybrid.py`

**Interfaces:**
- Produces: `ChromaVectorStore.hybrid_search(text, vector, limit, filter)` (currently `vectra/backends/chroma_store.py:81-82`, a pure `similarity_search` passthrough) now does client-side lexical + semantic RRF fusion.

- [ ] **Step 1: Write the failing test**

Create `tests/test_backends/test_chroma_hybrid.py`:

```python
from unittest.mock import MagicMock
from vectra.backends.chroma_store import ChromaVectorStore


def make_config(client, collection):
    client.get_or_create_collection = MagicMock(return_value=collection)
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": client})()


class TestChromaHybridSearch:
    async def test_fuses_semantic_and_lexical_rank(self):
        collection = MagicMock()
        collection.query = MagicMock(return_value={
            "documents": [["the quick brown fox", "a completely unrelated sentence", "quick fox jumps high"]],
            "metadatas": [[{}, {}, {}]],
            "distances": [[0.1, 0.2, 0.15]],
        })
        store = ChromaVectorStore(make_config(MagicMock(), collection))

        results = await store.hybrid_search("quick fox", [0.1, 0.2], limit=2)

        assert len(results) == 2
        contents = [r["content"] for r in results]
        assert "a completely unrelated sentence" not in contents

    async def test_returns_results_with_no_lexical_overlap(self):
        collection = MagicMock()
        collection.query = MagicMock(return_value={
            "documents": [["alpha content", "beta content"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.3]],
        })
        store = ChromaVectorStore(make_config(MagicMock(), collection))

        results = await store.hybrid_search("zzz", [0.1, 0.2], limit=2)

        assert len(results) == 2
```

- [ ] **Step 2: Run and verify it fails**

Run: `pytest tests/test_backends/test_chroma_hybrid.py -v`
Expected: FAIL — current `hybrid_search` is a pure `similarity_search` passthrough, so the unrelated-but-close-scoring sentence stays in the top 2.

- [ ] **Step 3: Implement client-side lexical RRF**

Edit `vectra/backends/chroma_store.py`. Add `import re` to the top imports if not already present, and add these two methods to the `ChromaVectorStore` class, replacing the current `hybrid_search` method (`vectra/backends/chroma_store.py:81-82`):

```python
    def _lexical_overlap(self, query: str, content: str) -> float:
        def tokenize(s):
            return set(t for t in re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(t) > 2)
        query_tokens = tokenize(query)
        if not query_tokens:
            return 0.0
        content_tokens = tokenize(content)
        matches = len(query_tokens & content_tokens)
        return matches / len(query_tokens)

    async def hybrid_search(self, text: str, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        pool = await self.similarity_search(vector, max(limit * 4, 20), filter)
        if not pool:
            return []
        with_lexical = [{**d, "_lexical": self._lexical_overlap(text, d["content"])} for d in pool]
        semantic_ranked = sorted(with_lexical, key=lambda d: d["score"], reverse=True)
        lexical_ranked = sorted(with_lexical, key=lambda d: d["_lexical"], reverse=True)
        rrf_scores: Dict[str, float] = {}
        for ranked in (semantic_ranked, lexical_ranked):
            for idx, d in enumerate(ranked):
                key = d["content"]
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + idx + 1)
        seen: Dict[str, Dict[str, Any]] = {}
        for d in with_lexical:
            if d["content"] not in seen:
                seen[d["content"]] = d
        ordered = sorted(seen.values(), key=lambda d: rrf_scores.get(d["content"], 0.0), reverse=True)
        return [{k: v for k, v in d.items() if k != "_lexical"} for d in ordered[:limit]]
```

- [ ] **Step 4: Run and verify it passes**

Run: `pytest tests/test_backends/test_chroma_hybrid.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Run the existing Chroma test file to confirm nothing broke**

Run: `pytest tests/test_backends/test_chroma_store.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add vectra/backends/chroma_store.py tests/test_backends/test_chroma_hybrid.py
git commit -m "feat: implement real hybrid search for ChromaVectorStore"
```

---

### Task 4: Real hybrid search — QdrantVectorStore

**Files:**
- Modify: `vectra/backends/qdrant_store.py`
- Create: `tests/test_backends/test_qdrant_hybrid.py`

**Interfaces:**
- Produces: `QdrantVectorStore.hybrid_search` replaces its passthrough (`vectra/backends/qdrant_store.py:39-40`) with the same lexical+semantic RRF pattern as Task 3.

- [ ] **Step 1: Write the failing test**

Create `tests/test_backends/test_qdrant_hybrid.py`:

```python
from unittest.mock import AsyncMock
from vectra.backends.qdrant_store import QdrantVectorStore


def make_config(client):
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": client})()


class TestQdrantHybridSearch:
    async def test_fuses_semantic_and_lexical_rank(self):
        client = AsyncMock()
        client.search = AsyncMock(return_value=[
            {"payload": {"content": "the quick brown fox", "metadata": {}}, "score": 0.9},
            {"payload": {"content": "a completely unrelated sentence", "metadata": {}}, "score": 0.8},
            {"payload": {"content": "quick fox jumps high", "metadata": {}}, "score": 0.85},
        ])
        store = QdrantVectorStore(make_config(client))

        results = await store.hybrid_search("quick fox", [0.1, 0.2], limit=2)

        assert len(results) == 2
        contents = [r["content"] for r in results]
        assert "a completely unrelated sentence" not in contents
```

- [ ] **Step 2: Run and verify it fails**

Run: `pytest tests/test_backends/test_qdrant_hybrid.py -v`
Expected: FAIL — current `hybrid_search` is `return await self.similarity_search(vector, limit, filter)`, ranking purely by `score`.

- [ ] **Step 3: Implement client-side lexical RRF**

Edit `vectra/backends/qdrant_store.py`. Add `import re` to the imports if not present. Replace the current `hybrid_search` method (`vectra/backends/qdrant_store.py:39-40`):

```python
    async def hybrid_search(self, text: str, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        return await self.similarity_search(vector, limit, filter)
```

with:

```python
    def _lexical_overlap(self, query: str, content: str) -> float:
        def tokenize(s):
            return set(t for t in re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(t) > 2)
        query_tokens = tokenize(query)
        if not query_tokens:
            return 0.0
        content_tokens = tokenize(content)
        matches = len(query_tokens & content_tokens)
        return matches / len(query_tokens)

    async def hybrid_search(self, text: str, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        pool = await self.similarity_search(vector, max(limit * 4, 20), filter)
        if not pool:
            return []
        with_lexical = [{**d, "_lexical": self._lexical_overlap(text, d["content"])} for d in pool]
        semantic_ranked = sorted(with_lexical, key=lambda d: d["score"], reverse=True)
        lexical_ranked = sorted(with_lexical, key=lambda d: d["_lexical"], reverse=True)
        rrf_scores: Dict[str, float] = {}
        for ranked in (semantic_ranked, lexical_ranked):
            for idx, d in enumerate(ranked):
                key = d["content"]
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + idx + 1)
        seen: Dict[str, Dict[str, Any]] = {}
        for d in with_lexical:
            if d["content"] not in seen:
                seen[d["content"]] = d
        ordered = sorted(seen.values(), key=lambda d: rrf_scores.get(d["content"], 0.0), reverse=True)
        return [{k: v for k, v in d.items() if k != "_lexical"} for d in ordered[:limit]]
```

- [ ] **Step 4: Run and verify it passes**

Run: `pytest tests/test_backends/test_qdrant_hybrid.py -v`
Expected: 1 test passes.

- [ ] **Step 5: Run the existing Qdrant test file to confirm nothing broke**

Run: `pytest tests/test_backends/test_qdrant_store.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add vectra/backends/qdrant_store.py tests/test_backends/test_qdrant_hybrid.py
git commit -m "feat: implement real hybrid search for QdrantVectorStore"
```

---

### Task 5: Real hybrid search — MilvusVectorStore

**Files:**
- Modify: `vectra/backends/milvus_store.py`
- Create: `tests/test_backends/test_milvus_hybrid.py`

**Interfaces:**
- Produces: `MilvusVectorStore.hybrid_search` replaces its passthrough with the same lexical+semantic RRF pattern.

- [ ] **Step 1: Write the failing test**

Create `tests/test_backends/test_milvus_hybrid.py`:

```python
from unittest.mock import AsyncMock
from vectra.backends.milvus_store import MilvusVectorStore


def make_config(client):
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": client})()


class TestMilvusHybridSearch:
    async def test_fuses_semantic_and_lexical_rank(self):
        client = AsyncMock()
        client.search = AsyncMock(return_value={
            "results": [
                {"content": "the quick brown fox", "metadata": {}, "distance": 0.1},
                {"content": "a completely unrelated sentence", "metadata": {}, "distance": 0.15},
                {"content": "quick fox jumps high", "metadata": {}, "distance": 0.12},
            ]
        })
        store = MilvusVectorStore(make_config(client))

        results = await store.hybrid_search("quick fox", [0.1, 0.2], limit=2)

        assert len(results) == 2
        contents = [r["content"] for r in results]
        assert "a completely unrelated sentence" not in contents
```

- [ ] **Step 2: Run and verify it fails**

Run: `pytest tests/test_backends/test_milvus_hybrid.py -v`
Expected: FAIL — current `hybrid_search` is a pure `similarity_search` passthrough.

- [ ] **Step 3: Implement client-side lexical RRF**

Edit `vectra/backends/milvus_store.py`. Add `import re` to the imports if not present. Replace the current `hybrid_search` method:

```python
    async def hybrid_search(self, text: str, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        return await self.similarity_search(vector, limit, filter)
```

with:

```python
    def _lexical_overlap(self, query: str, content: str) -> float:
        def tokenize(s):
            return set(t for t in re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(t) > 2)
        query_tokens = tokenize(query)
        if not query_tokens:
            return 0.0
        content_tokens = tokenize(content)
        matches = len(query_tokens & content_tokens)
        return matches / len(query_tokens)

    async def hybrid_search(self, text: str, vector: List[float], limit: int = 5, filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        pool = await self.similarity_search(vector, max(limit * 4, 20), filter)
        if not pool:
            return []
        with_lexical = [{**d, "_lexical": self._lexical_overlap(text, d["content"])} for d in pool]
        semantic_ranked = sorted(with_lexical, key=lambda d: d["score"], reverse=True)
        lexical_ranked = sorted(with_lexical, key=lambda d: d["_lexical"], reverse=True)
        rrf_scores: Dict[str, float] = {}
        for ranked in (semantic_ranked, lexical_ranked):
            for idx, d in enumerate(ranked):
                key = d["content"]
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + idx + 1)
        seen: Dict[str, Dict[str, Any]] = {}
        for d in with_lexical:
            if d["content"] not in seen:
                seen[d["content"]] = d
        ordered = sorted(seen.values(), key=lambda d: rrf_scores.get(d["content"], 0.0), reverse=True)
        return [{k: v for k, v in d.items() if k != "_lexical"} for d in ordered[:limit]]
```

- [ ] **Step 4: Run and verify it passes**

Run: `pytest tests/test_backends/test_milvus_hybrid.py -v`
Expected: 1 test passes.

- [ ] **Step 5: Run the existing Milvus test file to confirm nothing broke**

Run: `pytest tests/test_backends/test_milvus_store.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add vectra/backends/milvus_store.py tests/test_backends/test_milvus_hybrid.py
git commit -m "feat: implement real hybrid search for MilvusVectorStore"
```

---

### Task 6: Fix hardcoded 1536-dimension assumption (Postgres + Prisma)

**Files:**
- Modify: `vectra/backends/prisma_store.py:18,56`
- Modify: `vectra/core.py:346`
- Create: `tests/test_backends/test_dimension_config.py`

**Interfaces:**
- Produces: `PrismaVectorStore.ensure_indexes(dimensions=1536)` now accepts an explicit dimension (Postgres already accepts this parameter — confirmed during Phase 3 research — this task only needs to make `core.py` actually pass it; Prisma needs the parameter added). `VectraClient` now passes `self.config.embedding.dimensions` at its one `ensure_indexes` call site.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backends/test_dimension_config.py`:

```python
from unittest.mock import AsyncMock
from vectra.backends.postgres_store import PostgresVectorStore
from vectra.backends.prisma_store import PrismaVectorStore


class FakeConn:
    def __init__(self):
        self.executemany = AsyncMock(return_value=None)
        self.fetch = AsyncMock(return_value=[])
        self.execute = AsyncMock(return_value="CREATE TABLE")


def make_postgres_config(client):
    return type("Cfg", (), {"table_name": "document", "column_map": {}, "client_instance": client})()


def make_prisma_config(client):
    return type("Cfg", (), {
        "table_name": "Document",
        "column_map": {"content": "content", "metadata": "metadata", "vector": "embedding"},
        "client_instance": client,
    })()


class TestDimensionConfig:
    async def test_postgres_creates_table_with_given_dimension(self):
        client = FakeConn()
        store = PostgresVectorStore(make_postgres_config(client))

        await store.ensure_indexes(dimensions=768)

        create_calls = [c for c in client.execute.call_args_list if "vector(" in c.args[0]]
        assert any("vector(768)" in c.args[0] for c in create_calls)
        assert not any("vector(1536)" in c.args[0] for c in create_calls)

    async def test_postgres_defaults_to_1536_when_no_dimension_given(self):
        client = FakeConn()
        store = PostgresVectorStore(make_postgres_config(client))

        await store.ensure_indexes()

        create_calls = [c for c in client.execute.call_args_list if "vector(" in c.args[0]]
        assert any("vector(1536)" in c.args[0] for c in create_calls)

    async def test_prisma_alters_column_with_given_dimension(self):
        client = AsyncMock()
        client.query_raw = AsyncMock(return_value=[])  # no existing columns -> triggers ADD COLUMN path
        client.execute_raw = AsyncMock(return_value=None)
        store = PrismaVectorStore(make_prisma_config(client))

        await store.ensure_indexes(dimensions=768)

        alter_calls = [c for c in client.execute_raw.call_args_list if "ADD COLUMN" in str(c.args[0]) and "vector(" in str(c.args[0])]
        assert any("vector(768)" in str(c.args[0]) for c in alter_calls)
        assert not any("vector(1536)" in str(c.args[0]) for c in alter_calls)
```

- [ ] **Step 2: Run and verify the Prisma test fails**

Run: `pytest tests/test_backends/test_dimension_config.py -v`
Expected: `test_postgres_creates_table_with_given_dimension` and `test_postgres_defaults_to_1536_when_no_dimension_given` already PASS (Postgres already accepts a `dimensions` parameter, confirmed in research). `test_prisma_alters_column_with_given_dimension` FAILS — `PrismaVectorStore.ensure_indexes` currently takes no parameters and always hardcodes `vector(1536)`.

- [ ] **Step 3: Add the dimension parameter to PrismaVectorStore**

Edit `vectra/backends/prisma_store.py` line 18, change:

```python
    async def ensure_indexes(self):
```

to:

```python
    async def ensure_indexes(self, dimensions: int = 1536):
```

Then line 56, change:

```python
                    alter_stmts.append(f'ADD COLUMN "{c_vec}" vector(1536)')
```

to:

```python
                    alter_stmts.append(f'ADD COLUMN "{c_vec}" vector({dimensions})')
```

- [ ] **Step 4: Thread the config value through from core.py**

Edit `vectra/core.py` line 346, change:

```python
            try: await self.vector_store.ensure_indexes()
```

to:

```python
            try: await self.vector_store.ensure_indexes(self.config.embedding.dimensions or 1536)
```

- [ ] **Step 5: Run and verify all pass**

Run: `pytest tests/test_backends/test_dimension_config.py -v`
Expected: 3 tests pass.

- [ ] **Step 6: Run the full suite**

Run: `pytest`
Expected: all tests pass — check `tests/test_backends/test_postgres_store.py` and `tests/test_backends/test_prisma_store.py` don't assume the old zero-arg `ensure_indexes` signature (Phase 1's tests only exercise `add_documents`/`similarity_search`/`update_documents`, not `ensure_indexes`, so no conflict).

- [ ] **Step 7: Commit**

```bash
git add vectra/backends/prisma_store.py vectra/core.py tests/test_backends/test_dimension_config.py
git commit -m "fix: respect configured embedding dimension instead of hardcoding 1536"
```

---

### Task 7: Embedding-space MMR

**Files:**
- Modify: `vectra/core.py:558-616` (`_mmr_select`)
- Modify: `vectra/core.py:670-674` (MMR branch of `query_rag`)
- Create: `tests/test_retrieval_mmr_embedding_space.py`

**Interfaces:**
- Produces: `_mmr_select(candidates, k, mmr_lambda)` now computes diversity via cosine similarity on an `embedding` key when EVERY candidate has one; otherwise falls back to the existing lexical Jaccard path unchanged (`tests/test_retrieval.py`'s existing MMR tests keep passing, keep exercising the lexical path, since they never attach embeddings). `query_rag`'s MMR branch now batch-embeds candidate content via `self.embedder.embed_documents(...)` before calling `_mmr_select`, best-effort.

- [ ] **Step 1: Write the failing test**

Create `tests/test_retrieval_mmr_embedding_space.py`:

```python
from vectra.core import VectraClient


class TestMmrSelectEmbeddingSpace:
    def test_uses_cosine_similarity_when_every_candidate_has_an_embedding(self):
        # Two candidates share zero lexical tokens with the top pick, but one is
        # embedding-similar (should be penalized) and one is embedding-dissimilar
        # (should be preferred) — lexical Jaccard alone cannot distinguish these
        # (both would score 0 overlap).
        candidates = [
            {"content": "xyz abc def", "score": 0.9, "embedding": [1, 0, 0]},
            {"content": "qrs tuv wxy", "score": 0.85, "embedding": [0.99, 0.01, 0]},
            {"content": "lmn opq rst", "score": 0.8, "embedding": [0, 1, 0]},
        ]

        result = VectraClient._mmr_select(None, candidates, 2, 0.3)

        assert len(result) == 2
        assert result[0]["content"] == "xyz abc def"
        assert result[1]["content"] == "lmn opq rst"

    def test_falls_back_to_lexical_jaccard_when_no_embeddings(self):
        candidates = [
            {"content": "the quick brown fox jumps over the lazy dog", "score": 0.9},
            {"content": "the quick brown fox jumps over the lazy cat", "score": 0.85},
            {"content": "completely unrelated content about space travel", "score": 0.7},
        ]

        result = VectraClient._mmr_select(None, candidates, 2, 0.1)

        assert result[1]["content"] == "completely unrelated content about space travel"

    def test_falls_back_to_lexical_when_only_some_candidates_have_embeddings(self):
        candidates = [
            {"content": "alpha beta gamma delta", "score": 0.9, "embedding": [1, 0]},
            {"content": "epsilon zeta eta theta", "score": 0.8},
        ]

        result = VectraClient._mmr_select(None, candidates, 2, 0.5)

        assert len(result) == 2
```

- [ ] **Step 2: Run and verify the first test fails**

Run: `pytest tests/test_retrieval_mmr_embedding_space.py -v`
Expected: the first test FAILS (current `_mmr_select` always uses lexical Jaccard, sees zero overlap for both candidates, and would pick based on `score` alone — `qrs tuv wxy` at 0.85 over `lmn opq rst` at 0.8). The second and third tests should already pass.

- [ ] **Step 3: Implement the embedding-space path**

Edit `vectra/core.py`, replace the entire `_mmr_select` method (currently lines 558-616):

```python
    def _mmr_select(self, candidates: List[Dict[str, Any]], k: int, mmr_lambda: float) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        k_int = max(1, int(k))
        lam = max(0.0, min(1.0, float(mmr_lambda)))

        def tokens(text: str) -> set:
            return set(t for t in re.findall(r"[a-zA-Z0-9]+", (text or "").lower()) if len(t) > 2)

        def cosine_similarity(a, b) -> float:
            if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b) or len(a) == 0:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(y * y for y in b) ** 0.5
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            return dot / (norm_a * norm_b)

        use_embeddings = all(isinstance(d.get("embedding"), list) and len(d["embedding"]) > 0 for d in candidates)

        cand = []
        for d in candidates:
            dd = dict(d)
            dd["_tokens"] = None if use_embeddings else tokens(dd.get("content", ""))
            dd["_rel"] = float(dd.get("score", 0.0) or 0.0)
            cand.append(dd)

        cand.sort(key=lambda x: x.get("_rel", 0.0), reverse=True)
        selected: List[Dict[str, Any]] = []
        selected_diversity_keys: List[Any] = []

        first = cand.pop(0)
        selected.append(first)
        selected_diversity_keys.append(first.get("embedding") if use_embeddings else (first.get("_tokens") or set()))

        def jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            inter = len(a & b)
            if inter == 0:
                return 0.0
            union = len(a | b)
            return inter / union if union else 0.0

        while cand and len(selected) < k_int:
            best_idx = -1
            best_score = None
            for i, d in enumerate(cand):
                rel = d.get("_rel", 0.0)
                div = 0.0
                for key in selected_diversity_keys:
                    if use_embeddings:
                        div = max(div, cosine_similarity(d.get("embedding"), key))
                    else:
                        div = max(div, jaccard(d.get("_tokens") or set(), key))
                score = lam * rel - (1.0 - lam) * div
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx < 0:
                break
            picked = cand.pop(best_idx)
            selected.append(picked)
            selected_diversity_keys.append(picked.get("embedding") if use_embeddings else (picked.get("_tokens") or set()))

        out = []
        for d in selected[:k_int]:
            dd = dict(d)
            dd.pop("_tokens", None)
            dd.pop("_rel", None)
            out.append(dd)
        return out
```

- [ ] **Step 4: Run and verify all pass**

Run: `pytest tests/test_retrieval_mmr_embedding_space.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Run the existing retrieval tests to confirm no regression**

Run: `pytest tests/test_retrieval.py -v`
Expected: all pass unchanged.

- [ ] **Step 6: Wire batch-embedding into query_rag's MMR branch**

Edit `vectra/core.py`, replace the `RetrievalStrategy.MMR` branch (currently lines 670-674):

```python
            elif strategy == RetrievalStrategy.MMR:
                fetch_k = int(getattr(self.config.retrieval, "mmr_fetch_k", 20))
                mmr_lam = float(getattr(self.config.retrieval, "mmr_lambda", 0.5))
                candidates = await self.vector_store.similarity_search(query_vector, max(fetch_k, k), filter)
                docs = self._mmr_select(candidates, k, mmr_lam)
```

with:

```python
            elif strategy == RetrievalStrategy.MMR:
                fetch_k = int(getattr(self.config.retrieval, "mmr_fetch_k", 20))
                mmr_lam = float(getattr(self.config.retrieval, "mmr_lambda", 0.5))
                candidates = await self.vector_store.similarity_search(query_vector, max(fetch_k, k), filter)
                if candidates and hasattr(self.embedder, "embed_documents"):
                    try:
                        candidate_embeddings = await self.embedder.embed_documents([c["content"] for c in candidates])
                        for c, emb in zip(candidates, candidate_embeddings):
                            c["embedding"] = emb
                    except Exception:
                        pass  # Embedding-space MMR is best-effort; falls back to lexical Jaccard.
                docs = self._mmr_select(candidates, k, mmr_lam)
```

- [ ] **Step 7: Run the full suite**

Run: `pytest`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add vectra/core.py tests/test_retrieval_mmr_embedding_space.py
git commit -m "feat: embedding-space MMR diversity, falling back to lexical Jaccard when no embeddings"
```

---

### Task 8: Real tokenizer

**Files:**
- Modify: `pyproject.toml` (add `tiktoken` dependency)
- Modify: `vectra/core.py:1` (add import), `vectra/core.py:448-453` (`_token_estimate`)
- Create: `tests/test_token_estimate.py`

**Interfaces:**
- Produces: `VectraClient._token_estimate(text)` now returns a real BPE token count (via `tiktoken`'s `cl100k_base` encoding) instead of the character-count heuristic.

- [ ] **Step 1: Add the dependency**

Edit `pyproject.toml`'s `dependencies` list, add `"tiktoken"` (keep the existing alphabetical-ish ordering, or add it near the end before the closing `]`):

```toml
dependencies = [
    "pypdf",
    "mammoth",
    "openpyxl",
    "openai",
    "google-genai",
    "anthropic",
    "pydantic",
    "prisma",
    "chromadb",
    "requests",
    "asyncpg",
    "aiohttp",
    "pysbd",
    "tiktoken"
]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_token_estimate.py`:

```python
from vectra.core import VectraClient


class TestTokenEstimate:
    def test_returns_zero_for_empty_text(self):
        assert VectraClient._token_estimate(None, "") == 0
        assert VectraClient._token_estimate(None, None) == 0

    def test_matches_known_cl100k_base_token_count(self):
        # "Hello, world!" is a standard tiktoken example that tokenizes to 4 tokens
        # under cl100k_base — a real, verifiable BPE count, not a heuristic.
        assert VectraClient._token_estimate(None, "Hello, world!") == 4

    def test_returns_fewer_tokens_than_the_old_char_heuristic_for_repeated_short_words(self):
        text = ("cat " * 100).strip()
        old_heuristic = max(1, (len(text) + 3) // 4)
        real = VectraClient._token_estimate(None, text)
        assert real < old_heuristic
```

- [ ] **Step 3: Run and verify the count-specific tests fail**

Run: `pip install -e ".[dev]"`
Run: `pytest tests/test_token_estimate.py -v`
Expected: the empty-text test passes; the exact-count test and the heuristic-comparison test FAIL against the current character-count implementation.

- [ ] **Step 4: Replace the implementation**

Edit `vectra/core.py` near the top of the file, add to the existing import block:

```python
import tiktoken
```

Then add a module-level encoder right after the imports (before the `class LRUCache` / `class VectraClient` definitions):

```python
_token_encoder = tiktoken.get_encoding("cl100k_base")
```

Then edit `_token_estimate` (currently lines 448-453), change:

```python
    def _token_estimate(self, text: str) -> int:
        if not text:
            return 0
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        non_ascii = len(text) - ascii_chars
        return max(1, (ascii_chars + 3) // 4 + non_ascii)
```

to:

```python
    def _token_estimate(self, text: str) -> int:
        if not text:
            return 0
        return len(_token_encoder.encode(str(text)))
```

- [ ] **Step 5: Run and verify all pass**

Run: `pytest tests/test_token_estimate.py -v`
Expected: 3 tests pass.

- [ ] **Step 6: Run the full suite**

Run: `pytest`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml vectra/core.py tests/test_token_estimate.py
git commit -m "feat: replace character-count token heuristic with real tiktoken BPE tokenizer"
```

---

### Task 9: Preempt two known bugs found during vectra-js's Phase 3 final review

This task was added after Tasks 1-8 were planned, based on real bugs found (and fixed, across 3 review rounds) in the sibling vectra-js Phase 3 branch. Both bugs are structurally present in vectra-py's current code too — fixing them now avoids repeating the same review/fix cycle.

**Files:**
- Modify: `vectra/backends/milvus_store.py`
- Modify: `vectra/core.py`
- Create: `tests/test_backends/test_milvus_score_normalization.py`
- Modify: `tests/test_backends/test_milvus_hybrid.py` (if needed, to keep it passing under the new descending-sort-after-normalization behavior)
- Modify: `tests/test_retrieval_mmr_embedding_space.py` / create `tests/test_query_rag_ordering.py` (order-preservation regression tests)

**Bug A — Milvus score field is a raw, unnormalized, metric-dependent distance, not a consistent higher-is-better score.**

`vectra/backends/milvus_store.py`'s `similarity_search` currently does `'score': h.get('distance', 0.0)` with no normalization — this is Milvus's raw metric output, which for L2 distance is lower-is-better and for COSINE/IP is higher-is-better (COSINE range is `[-1, 1]`). Task 5's `hybrid_search` sorts `semantic_ranked` descending assuming higher-is-better, which is only correct for COSINE/IP, not L2.

Fix (mirrors the vectra-js fix): add an optional `metric_type` config attribute to `MilvusVectorStore` (read via `getattr(self.config, 'metric_type', None) or 'COSINE'`, uppercased), and normalize in `similarity_search`:
- `'COSINE'` or `'IP'`: pass the raw score through unchanged (already higher-is-better in Milvus's convention).
- `'L2'`: invert via `1.0 / (1.0 + score)` (L2 distance is always >= 0, so this is monotonic with no boundary/negative-value issues).

Do NOT use a heuristic based on the score's numeric value (e.g. "if score <= 1, assume already normalized") — that approach is unsound (misclassifies real L2 distances under 1.0, and breaks negative COSINE scores). Use the explicit `metric_type` config attribute only.

Also thread `metric_type` through wherever `vectra-py`'s config schema validates/constructs backend config objects (check `vectra/config.py` for how `database`/vector-store config is defined — e.g. a Pydantic model — and add `metric_type: Optional[str] = None` there if the schema would otherwise silently drop an unrecognized field, exactly as Pydantic/dataclass validation would). Confirm the value actually survives from a user-facing `VectraClient(config=...)` call through to `MilvusVectorStore`'s constructor — write a test that goes through the public config path (not by constructing `MilvusVectorStore` directly), proving `metric_type='L2'` set in the top-level config actually changes normalization behavior.

Test cases required in `tests/test_backends/test_milvus_score_normalization.py`:
- COSINE (default): a negative score (e.g. `-0.5`) stays `-0.5`, not inverted to a large positive number.
- L2: a distance of `0` (perfect match) normalizes to the best possible score (`1.0`), and a distance of `2.0` normalizes lower than a distance of `0.5` (monotonic, no boundary discontinuity around `1.0`).
- End-to-end: `metric_type='L2'` set via the public `VectraClient` config path is honored by the constructed `MilvusVectorStore`.

**Bug B — `query_rag`'s keyword-boost re-sort (`vectra/core.py` around line 701-709) silently discards the ordering that reranking, hybrid search, MULTI_QUERY (RRF fusion), and MMR (diversity selection) already established.**

The `boosted.sort(key=lambda x: (x.get('score', 0) + 0.1 * x.get('_boost', 0)), reverse=True)` line always re-sorts by raw vector `score`. This is correct ONLY for the plain NAIVE/HYDE retrieval path (where `score` is a meaningful ranking signal and no other ordering has been established). It is WRONG whenever:
- Reranking ran (`self.config.reranking.enabled and self.reranker` — the reranker's returned order is authoritative, e.g. real Cohere/Jina relevance ranking from Tasks 1-2).
- `strategy == RetrievalStrategy.HYBRID` (the hybrid store's RRF-fused order from Tasks 3-5 is authoritative).
- `strategy == RetrievalStrategy.MULTI_QUERY` (the `_reciprocal_rank_fusion` order is authoritative — note this fusion returns original doc dicts with their raw, pre-fusion `score` still attached, so the boost re-sort silently undoes the fusion exactly like the hybrid case).
- `strategy == RetrievalStrategy.MMR` (the `_mmr_select` greedy diversity order from Task 7 is authoritative).

Fix: compute whether the incoming `docs` order is already meaningful (reranking applied OR strategy is HYBRID/MULTI_QUERY/MMR), and skip the boost re-sort in that case — return `docs` (with the `_boost` field still computed/attached the same way, since downstream code may read it, but without reordering). Only apply the `sort(...)` call on the plain NAIVE/HYDE path where no other explicit ordering was established. Reuse the existing `strategy` variable and the existing reranking-enabled check already present in this function — don't introduce a second, differently-computed condition.

Add regression tests proving: (a) when reranking is enabled, the final document order in the response matches the reranker's returned order, not raw-score order (use a discriminating fixture — the reranker-preferred doc must have a LOWER raw score than another doc, so a leftover re-sort would visibly reorder it); (b) same for MULTI_QUERY (RRF-first doc has a lower raw score); (c) same for MMR (a diversity-preferred lower-score doc must rank ahead of a near-duplicate higher-score doc). Do not use fixtures where the fixed and buggy code would coincidentally produce the same order — verify each fixture is actually discriminating by reasoning through what the OLD (buggy, always-re-sort) code would produce vs. the NEW (fixed) code, and confirm they differ, before finalizing the test.

- [ ] Write failing tests for both bugs first (TDD), confirm they fail against the current code.
- [ ] Implement Bug A's fix in `milvus_store.py` (and config schema if needed).
- [ ] Implement Bug B's fix in `core.py`.
- [ ] Run the new tests, confirm they pass.
- [ ] Run `pytest tests/test_backends/test_milvus_hybrid.py tests/test_retrieval_mmr_embedding_space.py -v` to confirm Tasks 5 and 7's existing tests still pass unchanged.
- [ ] Run the full suite (`pytest`), confirm everything passes.
- [ ] Commit: `git add vectra/backends/milvus_store.py vectra/core.py vectra/config.py tests/ && git commit -m "fix: normalize Milvus scores by metric type and preserve reranker/hybrid/multi-query/MMR ordering through query_rag"`

---

## Self-Review Notes

- **Spec coverage:** real Cohere/Jina rerankers (Tasks 1-2), real hybrid search on all 3 remaining stores (Tasks 3-5), hardcoded-dimension fix (Task 6), embedding-space MMR (Task 7), real tokenizer (Task 8) — mirrors vectra-js's Phase 3 exactly, all 5 items covered for feature parity between the two SDKs.
- **Placeholder scan:** no TBD/TODO; every step has runnable code.
- **Type consistency:** `_lexical_overlap`/lexical-RRF pattern duplicated near-identically across Tasks 3-5, matching the codebase's existing convention (Postgres/Prisma RRF isn't shared either) and mirroring vectra-js's identical choice.
- **Async-appropriate HTTP client:** `aiohttp` chosen over `requests` for the rerankers specifically because `rerank()` is `async def` — using sync `requests` there would block the event loop, a real correctness concern the JS sibling doesn't have (Node's `fetch` is naturally async). This is documented as a deliberate deviation from `telemetry.py`'s sync-`requests` precedent, not an oversight.
- **Fewer call sites than JS:** vectra-py has exactly one `ensure_indexes()` call site in `core.py` (vs. two in JS) — confirmed via direct grep before writing Task 6, not assumed from the JS plan's shape.
