# Phase 1 — vectra-py Trust & Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give vectra-py a real test suite, a genuinely MIT license (currently self-contradicting: MIT in `pyproject.toml`, GPLv3 in the classifier and the LICENSE file), telemetry off by default, a corrected install command in its own README, and three real CRUD bugs fixed (Milvus delete count always 0, `update_documents` unimplemented for Postgres and Milvus).

**Architecture:** No architectural changes. Adds a pytest suite covering the pure algorithmic core (`_reciprocal_rank_fusion`, `_mmr_select`), Pydantic config validation, telemetry gating, one integration-style test per vector-store backend using a mocked `client_instance`, and tests-that-double-as-fixes for the three CRUD bugs (TDD: write the test proving the bug first, watch it fail against current code, then fix).

**Tech Stack:** Python ≥3.8, pytest + pytest-asyncio for async test support, Pydantic (already a dependency) for config validation.

## Global Constraints

- License target: MIT, exactly as decided in `vectra-js/docs/superpowers/specs/2026-08-06-vectra-productization-design.md` (this repo's sibling — the spec lives there since that's where this session started).
- Confirmed correct, live PyPI package name: `vectra-rag-py`. Any reference to `vectra-py` as an install target is a bug.
- No direct-to-master commits, no force-push, no skipped hooks.
- Every task ends with tests passing (`pytest <path>`) before moving to the next task.
- Do not touch reranking, guardrail enforcement, hybrid search, the missing SQL-identifier sanitization in `postgres_store.py`, or the hardcoded-dimension issue — those are Phase 2/3 scope. (The missing sanitization in particular is a real, separately-tracked security gap for Phase 2 — this plan does not add it.)

---

### Task 1: Test framework setup

**Files:**
- Modify: `pyproject.toml` (add `[project.optional-dependencies]` and `[tool.pytest.ini_options]`)
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: `pytest` runs the suite from the repo root; `pytest tests/path/test_x.py` runs a single file; async `def test_...()` functions work without needing `@pytest.mark.asyncio` on each one (via `asyncio_mode = "auto"`).

- [ ] **Step 1: Add pytest as an optional dev dependency**

Edit `pyproject.toml`, after the closing `]` of the `dependencies` list (currently ending at `"pysbd"` followed by `]`), add a new top-level table:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24"]
```

- [ ] **Step 2: Configure pytest-asyncio auto mode**

Add to the end of `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 3: Create the tests package**

Create `tests/__init__.py` (empty file).

- [ ] **Step 4: Install and verify**

Run: `pip install -e ".[dev]"`
Run: `pytest`
Expected: "no tests ran" (exit code 5) — confirms pytest itself works; no test files exist yet.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/__init__.py
git commit -m "test: add pytest test runner"
```

---

### Task 2: RRF and MMR unit tests

**Files:**
- Create: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `VectraClient._reciprocal_rank_fusion(self, doc_lists, k=60)` and `VectraClient._mmr_select(self, candidates, k, mmr_lambda)` from `vectra/core.py:532` and `vectra/core.py:549`. Neither reads `self`, so both can be called as plain functions off the class with `None` as the first argument.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retrieval.py`:

```python
from vectra.core import VectraClient


class TestReciprocalRankFusion:
    def test_merges_two_ranked_lists_favoring_docs_in_both(self):
        list_a = [{"content": "alpha"}, {"content": "beta"}]
        list_b = [{"content": "beta"}, {"content": "gamma"}]
        result = VectraClient._reciprocal_rank_fusion(None, [list_a, list_b])
        assert [d["content"] for d in result] == ["beta", "alpha", "gamma"]

    def test_returns_empty_list_for_empty_input(self):
        assert VectraClient._reciprocal_rank_fusion(None, []) == []

    def test_deduplicates_by_content_keeping_first_seen(self):
        list_a = [{"content": "same", "tag": "first"}]
        list_b = [{"content": "same", "tag": "second"}]
        result = VectraClient._reciprocal_rank_fusion(None, [list_a, list_b])
        assert len(result) == 1
        assert result[0]["tag"] == "first"


class TestMmrSelect:
    def test_returns_empty_list_for_empty_candidates(self):
        assert VectraClient._mmr_select(None, [], 5, 0.5) == []

    def test_highest_scoring_candidate_selected_first(self):
        candidates = [
            {"content": "low score doc about cats", "score": 0.2},
            {"content": "high score doc about dogs", "score": 0.9},
        ]
        result = VectraClient._mmr_select(None, candidates, 2, 0.5)
        assert result[0]["content"] == "high score doc about dogs"

    def test_prefers_diverse_second_pick_over_near_duplicate(self):
        candidates = [
            {"content": "the quick brown fox jumps over the lazy dog", "score": 0.9},
            {"content": "the quick brown fox jumps over the lazy cat", "score": 0.85},
            {"content": "completely unrelated content about space travel", "score": 0.7},
        ]
        result = VectraClient._mmr_select(None, candidates, 2, 0.9)
        assert len(result) == 2
        assert result[1]["content"] == "completely unrelated content about space travel"

    def test_respects_the_k_limit(self):
        candidates = [
            {"content": f"doc number {i} with unique words {i}{i}{i}", "score": 1 - i * 0.05}
            for i in range(10)
        ]
        result = VectraClient._mmr_select(None, candidates, 3, 0.5)
        assert len(result) == 3
```

- [ ] **Step 2: Run and verify all pass**

Run: `pytest tests/test_retrieval.py -v`
Expected: 7 tests pass (no implementation changes — this task only adds coverage).

- [ ] **Step 3: Commit**

```bash
git add tests/test_retrieval.py
git commit -m "test: cover _reciprocal_rank_fusion and _mmr_select"
```

---

### Task 3: Config validation tests, telemetry default-off, telemetry tests

**Files:**
- Modify: `vectra/config.py:107`
- Modify: `vectra/telemetry.py:44`, `vectra/telemetry.py:74-80`
- Create: `tests/test_config.py`
- Create: `tests/test_telemetry.py`

**Interfaces:**
- Consumes: `VectraConfig`, `EmbeddingConfig`, `LLMConfig`, `DatabaseConfig`, `ProviderType` from `vectra.config`.
- Produces: `TelemetryManager().enabled` is `False` by default, and stays `False` after `.init(config)` unless the config's `telemetry.enabled` is explicitly `True`.

- [ ] **Step 1: Flip the TelemetryConfig default to False**

Edit `vectra/config.py` line 107, change:

```python
class TelemetryConfig(BaseModel):
    enabled: bool = True
```

to:

```python
class TelemetryConfig(BaseModel):
    enabled: bool = False
```

- [ ] **Step 2: Flip the TelemetryManager default and require explicit opt-in**

Edit `vectra/telemetry.py` line 44, change:

```python
        self.enabled = True
```

to:

```python
        self.enabled = False
```

Edit `vectra/telemetry.py` lines 74-76, change:

```python
        if telemetry_cfg.get("enabled") is False:
            self.enabled = False
            return
```

to:

```python
        if telemetry_cfg.get("enabled") is not True:
            self.enabled = False
            return
```

- [ ] **Step 3: Write the telemetry test**

Create `tests/test_telemetry.py`:

```python
import os
import importlib


def _fresh_telemetry_module():
    import vectra.telemetry as telemetry_module
    importlib.reload(telemetry_module)
    return telemetry_module


class TestTelemetryDefaultOff:
    def test_disabled_by_construction_before_init(self):
        mod = _fresh_telemetry_module()
        assert mod.telemetry.enabled is False

    def test_stays_disabled_when_init_called_with_no_config(self):
        mod = _fresh_telemetry_module()
        mod.telemetry.init(None)
        assert mod.telemetry.enabled is False

    def test_stays_disabled_when_telemetry_enabled_is_omitted(self):
        mod = _fresh_telemetry_module()
        mod.telemetry.init({"telemetry": {}})
        assert mod.telemetry.enabled is False

    def test_enables_only_when_telemetry_enabled_is_explicitly_true(self, tmp_path, monkeypatch):
        mod = _fresh_telemetry_module()
        monkeypatch.setattr(mod, "TELEMETRY_DIR", tmp_path)
        monkeypatch.setattr(mod, "TELEMETRY_FILE", tmp_path / "telemetry.json")
        mod.telemetry.init({"telemetry": {"enabled": True}})
        assert mod.telemetry.enabled is True

    def test_stays_disabled_when_env_var_disables_even_if_config_enables(self, monkeypatch):
        mod = _fresh_telemetry_module()
        monkeypatch.setenv("VECTRA_TELEMETRY_DISABLED", "1")
        mod.telemetry.init({"telemetry": {"enabled": True}})
        assert mod.telemetry.enabled is False
```

*(`TelemetryManager` is a singleton via `__new__` — `importlib.reload` on the module re-runs the class body and creates a fresh `_instance`, which is why each test starts from a clean state instead of sharing state across tests.)*

- [ ] **Step 4: Write the config validation test**

Create `tests/test_config.py`:

```python
import pytest
from pydantic import ValidationError
from vectra.config import VectraConfig, EmbeddingConfig, LLMConfig, DatabaseConfig, ProviderType


def make_minimal_config(**overrides):
    base = dict(
        embedding=EmbeddingConfig(provider=ProviderType.OPENAI, api_key="test-key"),
        llm=LLMConfig(provider=ProviderType.OPENAI, api_key="test-key", model_name="gpt-4o-mini"),
        database=DatabaseConfig(type="chroma", client_instance=object()),
    )
    base.update(overrides)
    return base


class TestVectraConfig:
    def test_accepts_minimal_valid_config_and_fills_defaults(self):
        parsed = VectraConfig(**make_minimal_config())
        assert parsed.embedding.model_name == "text-embedding-3-small"
        assert parsed.telemetry.enabled is False
        assert parsed.database.column_map == {"content": "content", "vector": "vector", "metadata": "metadata"}

    def test_rejects_config_missing_embedding_provider(self):
        with pytest.raises(ValidationError):
            VectraConfig(**make_minimal_config(embedding={"api_key": "test-key"}))

    def test_rejects_config_missing_database_type(self):
        with pytest.raises(ValidationError):
            VectraConfig(**make_minimal_config(database={"client_instance": object()}))

    def test_rejects_agentic_chunking_with_no_agentic_llm(self):
        with pytest.raises(ValidationError, match="agentic_llm required"):
            VectraConfig(**make_minimal_config(chunking={"strategy": "agentic"}))

    def test_rejects_hyde_retrieval_with_no_llm_config(self):
        with pytest.raises(ValidationError, match="llm_config required"):
            VectraConfig(**make_minimal_config(retrieval={"strategy": "hyde"}))
```

- [ ] **Step 5: Run and verify all pass**

Run: `pytest tests/test_config.py tests/test_telemetry.py -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add vectra/config.py vectra/telemetry.py tests/test_config.py tests/test_telemetry.py
git commit -m "fix: default telemetry off, opt-in only; add config and telemetry tests"
```

---

### Task 4: Postgres backend integration test

**Files:**
- Create: `tests/test_backends/__init__.py`
- Create: `tests/test_backends/test_postgres_store.py`

**Interfaces:**
- Consumes: `PostgresVectorStore` from `vectra/backends/postgres_store.py:11-30`, `add_documents(documents)` at line 84 (batches via `conn.executemany(sql, data)` — one call for the whole batch, not one per document), `similarity_search(vector, limit, filter)` at line 132 (reads `row['distance']` from asyncpg and computes `score = 1 - distance`).
- `_get_connection()` (line 21) branches on `hasattr(self.client, 'acquire')`: if present, treats the client as a pool; if absent, wraps the client itself in a no-op async context manager so `conn` IS `self.client`. The test double below is a plain class exposing only `executemany`/`fetch`/`execute` — deliberately not `unittest.mock.AsyncMock()`, because `AsyncMock()` auto-creates *any* attribute on access, so `hasattr(mock, 'acquire')` is always `True` and silently sends the store down the wrong branch.

- [ ] **Step 1: Create the test package**

Create `tests/test_backends/__init__.py` (empty file).

- [ ] **Step 2: Write the failing test**

Create `tests/test_backends/test_postgres_store.py`:

```python
from unittest.mock import AsyncMock
from vectra.backends.postgres_store import PostgresVectorStore


class FakeConn:
    """Plain class, not AsyncMock — deliberately has no 'acquire' attribute
    so PostgresVectorStore._get_connection() treats it as a bare connection,
    not a pool. See Task 4's Interfaces note for why AsyncMock() is unsafe here."""
    def __init__(self):
        self.executemany = AsyncMock(return_value=None)
        self.fetch = AsyncMock(return_value=[])
        self.execute = AsyncMock(return_value="UPDATE 0")


def make_config(client):
    return type("Cfg", (), {"table_name": "document", "column_map": {}, "client_instance": client})()


class TestPostgresVectorStore:
    async def test_add_documents_batches_all_rows_into_one_executemany_call(self):
        client = FakeConn()
        store = PostgresVectorStore(make_config(client))

        await store.add_documents([
            {"id": "doc-1", "content": "hello world", "metadata": {"a": 1}, "embedding": [0.1, 0.2, 0.3]},
        ])

        client.executemany.assert_called_once()
        sql, rows = client.executemany.call_args[0]
        assert 'INSERT INTO "document"' in sql
        assert rows[0][0] == "doc-1"
        assert rows[0][1] == "hello world"

    async def test_similarity_search_converts_distance_to_score(self):
        client = FakeConn()
        client.fetch = AsyncMock(return_value=[
            {"id": "doc-1", "content": "hello world", "metadata": '{"a": 1}', "distance": 0.13},
        ])
        store = PostgresVectorStore(make_config(client))

        results = await store.similarity_search([0.1, 0.2, 0.3], limit=5)

        assert results == [{"id": "doc-1", "content": "hello world", "metadata": {"a": 1}, "score": 0.87}]
```

- [ ] **Step 3: Run and verify all pass**

Run: `pytest tests/test_backends/test_postgres_store.py -v`
Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_backends/__init__.py tests/test_backends/test_postgres_store.py
git commit -m "test: add PostgresVectorStore integration test"
```

---

### Task 5: Prisma backend integration test

**Files:**
- Create: `tests/test_backends/test_prisma_store.py`

**Interfaces:**
- Consumes: `PrismaVectorStore` from `vectra/backends/prisma_store.py:7-16`, `add_documents(documents)` at line 86, `similarity_search(vector, limit, filter)` at line 147. Unlike the Postgres store (no sanitization at all — see Task 4's note), `_safe_ident` (line 11) validates identifiers, but lazily: `__init__` just stores `config` with no validation, and each method (e.g. `add_documents`) calls `self._safe_ident(...)` on the table/column names the first time it runs. So an unsafe table name doesn't fail at construction — it fails on first use.
- The Prisma client (`config.client_instance`) is used directly for `.execute_raw`/`.query_raw` — no pool/connection indirection — so a plain `AsyncMock()` is safe to use as the client double here (contrast with Task 4's Postgres store, which needs a plain class instead).

- [ ] **Step 1: Write the failing test**

Create `tests/test_backends/test_prisma_store.py`:

```python
import pytest
from unittest.mock import AsyncMock
from vectra.backends.prisma_store import PrismaVectorStore


def make_config(client, table_name="Document"):
    return type("Cfg", (), {"table_name": table_name, "column_map": {}, "client_instance": client})()


class TestPrismaVectorStore:
    async def test_rejects_unsafe_table_name_on_first_operation(self):
        store = PrismaVectorStore(make_config(AsyncMock(), table_name='a"; DROP TABLE x; --'))
        with pytest.raises(ValueError, match="Unsafe SQL identifier"):
            await store.add_documents([{"content": "x", "metadata": {}, "embedding": [0.1]}])

    async def test_add_documents_issues_one_execute_raw_per_document(self):
        client = AsyncMock()
        store = PrismaVectorStore(make_config(client))

        await store.add_documents([
            {"id": "doc-1", "content": "hello world", "metadata": {"a": 1}, "embedding": [0.3, 0.4]},
        ])

        client.execute_raw.assert_called_once()
        query, doc_id, content = client.execute_raw.call_args[0][:3]
        assert 'INSERT INTO "Document"' in query
        assert doc_id == "doc-1"
        assert content == "hello world"

    async def test_similarity_search_returns_mapped_rows(self):
        client = AsyncMock()
        client.query_raw = AsyncMock(return_value=[{"content": "hello world", "metadata": {"a": 1}, "score": 0.9}])
        store = PrismaVectorStore(make_config(client))

        results = await store.similarity_search([0.3, 0.4], limit=5)

        assert results == [{"content": "hello world", "metadata": {"a": 1}, "score": 0.9}]
```

- [ ] **Step 2: Run and verify all pass**

Run: `pytest tests/test_backends/test_prisma_store.py -v`
Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_backends/test_prisma_store.py
git commit -m "test: add PrismaVectorStore integration test"
```

---

### Task 6: Chroma backend integration test

**Files:**
- Create: `tests/test_backends/test_chroma_store.py`

**Interfaces:**
- Consumes: `ChromaVectorStore` from `vectra/backends/chroma_store.py:7-17`, `add_documents(documents)` at line 19, `similarity_search(vector, limit, filter)` at line 60. Passing a non-`None` `client_instance` in config skips real `chromadb.PersistentClient` construction.

- [ ] **Step 1: Write the failing test**

Create `tests/test_backends/test_chroma_store.py`:

```python
from unittest.mock import MagicMock
from vectra.backends.chroma_store import ChromaVectorStore


def make_config(client, collection):
    client.get_or_create_collection = MagicMock(return_value=collection)
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": client})()


class TestChromaVectorStore:
    async def test_add_documents_calls_collection_add(self):
        collection = MagicMock()
        store = ChromaVectorStore(make_config(MagicMock(), collection))

        await store.add_documents([{"content": "hello world", "metadata": {"a": 1}, "embedding": [0.1, 0.2]}])

        collection.add.assert_called_once()
        call = collection.add.call_args.kwargs
        assert call["embeddings"] == [[0.1, 0.2]]
        assert call["metadatas"] == [{"a": 1}]
        assert call["documents"] == ["hello world"]

    async def test_similarity_search_maps_batched_response(self):
        collection = MagicMock()
        collection.query = MagicMock(return_value={
            "documents": [["hello world"]],
            "metadatas": [[{"a": 1}]],
            "distances": [[0.13]],
        })
        store = ChromaVectorStore(make_config(MagicMock(), collection))

        results = await store.similarity_search([0.1, 0.2], limit=5)

        assert results == [{"content": "hello world", "metadata": {"a": 1}, "score": 0.87}]
```

- [ ] **Step 2: Run and verify all pass**

Run: `pytest tests/test_backends/test_chroma_store.py -v`
Expected: 2 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_backends/test_chroma_store.py
git commit -m "test: add ChromaVectorStore integration test"
```

---

### Task 7: Qdrant backend integration test

**Files:**
- Create: `tests/test_backends/test_qdrant_store.py`

**Interfaces:**
- Consumes: `QdrantVectorStore` from `vectra/backends/qdrant_store.py:4-8`, `add_documents(documents)` at line 10, `similarity_search(vector, limit, filter)` at line 34, `_normalize_filter(filter)` at line 21.

- [ ] **Step 1: Write the failing test**

Create `tests/test_backends/test_qdrant_store.py`:

```python
from unittest.mock import AsyncMock
from vectra.backends.qdrant_store import QdrantVectorStore


def make_config(client):
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": client})()


class TestQdrantVectorStore:
    async def test_add_documents_upserts_points_with_vector_and_payload(self):
        client = AsyncMock()
        store = QdrantVectorStore(make_config(client))

        await store.add_documents([{"content": "hello world", "metadata": {"a": 1}, "embedding": [0.1, 0.2]}])

        client.upsert.assert_called_once()
        collection, kwargs = client.upsert.call_args[0][0], client.upsert.call_args[1]
        assert collection == "rag_collection"
        assert kwargs["points"][0]["vector"] == [0.1, 0.2]
        assert kwargs["points"][0]["payload"] == {"content": "hello world", "metadata": {"a": 1}}

    async def test_similarity_search_maps_hits(self):
        client = AsyncMock()
        client.search = AsyncMock(return_value=[
            {"payload": {"content": "hello world", "metadata": {"a": 1}}, "score": 0.92},
        ])
        store = QdrantVectorStore(make_config(client))

        results = await store.similarity_search([0.1, 0.2], limit=5)

        assert results == [{"content": "hello world", "metadata": {"a": 1}, "score": 0.92}]

    def test_normalize_filter_builds_must_clause(self):
        store = QdrantVectorStore(make_config(AsyncMock()))
        assert store._normalize_filter({"category": "docs"}) == {
            "must": [{"key": "metadata.category", "match": {"value": "docs"}}]
        }
```

- [ ] **Step 2: Run and verify all pass**

Run: `pytest tests/test_backends/test_qdrant_store.py -v`
Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_backends/test_qdrant_store.py
git commit -m "test: add QdrantVectorStore integration test"
```

---

### Task 8: Milvus backend — fix delete count bug, add integration tests

**Files:**
- Modify: `vectra/backends/milvus_store.py:95-100`
- Create: `tests/test_backends/test_milvus_store.py`

**Interfaces:**
- Consumes/fixes: `MilvusVectorStore.delete_documents(filter)` at `vectra/backends/milvus_store.py:95`, currently hardcoded to always `return 0` regardless of how many rows were actually deleted.

- [ ] **Step 1: Write the failing test proving the bug**

Create `tests/test_backends/test_milvus_store.py`:

```python
from unittest.mock import AsyncMock
from vectra.backends.milvus_store import MilvusVectorStore


def make_config(client):
    return type("Cfg", (), {"table_name": "rag_collection", "client_instance": client})()


class TestMilvusVectorStore:
    async def test_add_documents_inserts_vector_content_and_json_metadata(self):
        client = AsyncMock()
        store = MilvusVectorStore(make_config(client))

        await store.add_documents([{"content": "hello world", "metadata": {"a": 1}, "embedding": [0.1, 0.2]}])

        client.insert.assert_called_once_with(
            collection_name="rag_collection",
            fields_data=[{"vector": [0.1, 0.2], "content": "hello world", "metadata": '{"a": 1}'}],
        )

    async def test_delete_documents_returns_the_actual_delete_count(self):
        client = AsyncMock()
        client.delete = AsyncMock(return_value={"delete_count": 3})
        store = MilvusVectorStore(make_config(client))

        deleted = await store.delete_documents({"category": "docs"})

        assert deleted == 3

    async def test_delete_documents_handles_object_style_result(self):
        client = AsyncMock()
        result = type("Result", (), {"delete_count": 7})()
        client.delete = AsyncMock(return_value=result)
        store = MilvusVectorStore(make_config(client))

        deleted = await store.delete_documents({"category": "docs"})

        assert deleted == 7
```

- [ ] **Step 2: Run and verify the delete-count tests fail against current code**

Run: `pytest tests/test_backends/test_milvus_store.py -v`
Expected: `test_add_documents_...` passes; both `test_delete_documents_...` tests FAIL (`assert 0 == 3`) — this confirms the bug.

- [ ] **Step 3: Fix delete_documents to return the real count**

Edit `vectra/backends/milvus_store.py` lines 95-100, change:

```python
    async def delete_documents(self, filter: Dict[str, Any]) -> int:
        expr = self._filter_to_expr(filter)
        if not hasattr(self.client, "delete"):
            raise NotImplementedError("Milvus client does not support delete()")
        await self.client.delete(collection_name=self.collection, expr=expr)
        return 0
```

to:

```python
    async def delete_documents(self, filter: Dict[str, Any]) -> int:
        expr = self._filter_to_expr(filter)
        if not hasattr(self.client, "delete"):
            raise NotImplementedError("Milvus client does not support delete()")
        result = await self.client.delete(collection_name=self.collection, expr=expr)
        if isinstance(result, dict):
            return result.get("delete_count", 0)
        return getattr(result, "delete_count", 0)
```

- [ ] **Step 4: Run and verify all pass**

Run: `pytest tests/test_backends/test_milvus_store.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add vectra/backends/milvus_store.py tests/test_backends/test_milvus_store.py
git commit -m "fix: Milvus delete_documents returns the real delete count"
```

---

### Task 9: Implement update_documents for Postgres and Milvus

**Files:**
- Modify: `vectra/backends/postgres_store.py:173-174`
- Modify: `vectra/backends/milvus_store.py` (the `update_documents` stub, now at the end of the file after Task 8's edit)
- Modify: `tests/test_backends/test_postgres_store.py` (append a test)
- Modify: `tests/test_backends/test_milvus_store.py` (append a test)

**Interfaces:**
- Consumes/fixes: `PostgresVectorStore.update_documents(filter, update_data)` (currently `raise NotImplementedError`) and `MilvusVectorStore.update_documents(filter, update_data)` (currently `raise NotImplementedError("Milvus update_documents is not implemented")`).
- Follows the existing pattern in `qdrant_store.py:129-163` and `chroma_store.py:129-151`: `update_data` may contain `content` and/or `metadata` keys; only those present are changed; returns the count of documents affected.

- [ ] **Step 1: Write the failing test for Postgres**

Append to `tests/test_backends/test_postgres_store.py`:

```python
    async def test_update_documents_sets_content_and_metadata(self):
        client = FakeConn()
        client.execute = AsyncMock(return_value="UPDATE 2")
        store = PostgresVectorStore(make_config(client))

        updated = await store.update_documents({"category": "docs"}, {"content": "new text"})

        assert updated == 2
        sql = client.execute.call_args[0][0]
        assert "UPDATE" in sql
        assert '"content" = $1' in sql

    async def test_update_documents_returns_zero_for_empty_update_data(self):
        client = FakeConn()
        store = PostgresVectorStore(make_config(client))

        updated = await store.update_documents({"category": "docs"}, {})

        assert updated == 0
        client.execute.assert_not_called()
```

- [ ] **Step 2: Run and verify the new tests fail**

Run: `pytest tests/test_backends/test_postgres_store.py -v -k update_documents`
Expected: both FAIL (`NotImplementedError`).

- [ ] **Step 3: Implement update_documents in postgres_store.py**

Edit `vectra/backends/postgres_store.py` lines 173-174, change:

```python
    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        raise NotImplementedError
```

to:

```python
    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        if not update_data:
            return 0
        set_parts = []
        params: List[Any] = []
        if "content" in update_data and update_data["content"] is not None:
            params.append(update_data["content"])
            set_parts.append(f'"{self.c_content}" = ${len(params)}')
        if "metadata" in update_data and isinstance(update_data["metadata"], dict):
            params.append(json.dumps(update_data["metadata"]))
            set_parts.append(f'"{self.c_meta}" = ${len(params)}')
        if not set_parts:
            return 0
        params.append(json.dumps(filter))
        sql = f'UPDATE "{self.table_name}" SET {", ".join(set_parts)} WHERE "{self.c_meta}" @> ${len(params)}::jsonb'
        async with self._get_connection() as conn:
            res = await conn.execute(sql, *params)
            return int(res.split(' ')[1]) if res else 0
```

- [ ] **Step 4: Run and verify the Postgres tests pass**

Run: `pytest tests/test_backends/test_postgres_store.py -v`
Expected: all pass (4 tests total: 2 from Task 4, 2 new).

- [ ] **Step 5: Write the failing test for Milvus**

Append to `tests/test_backends/test_milvus_store.py`:

```python
    async def test_update_documents_merges_metadata_and_reupserts_with_existing_vector(self):
        client = AsyncMock()
        client.query = AsyncMock(return_value=[
            {"id": 1, "vector": [0.1, 0.2], "content": "old text", "metadata": {"a": 1}},
        ])
        client.upsert = AsyncMock()
        store = MilvusVectorStore(make_config(client))

        updated = await store.update_documents({"category": "docs"}, {"metadata": {"b": 2}})

        assert updated == 1
        client.upsert.assert_called_once()
        fields = client.upsert.call_args.kwargs["fields_data"]
        assert fields[0]["vector"] == [0.1, 0.2]
        assert fields[0]["content"] == "old text"
        assert fields[0]["metadata"] == {"a": 1, "b": 2}

    async def test_update_documents_returns_zero_when_nothing_matches(self):
        client = AsyncMock()
        client.query = AsyncMock(return_value=[])
        store = MilvusVectorStore(make_config(client))

        updated = await store.update_documents({"category": "docs"}, {"metadata": {"b": 2}})

        assert updated == 0
```

- [ ] **Step 6: Run and verify the new tests fail**

Run: `pytest tests/test_backends/test_milvus_store.py -v -k update_documents`
Expected: both FAIL (`NotImplementedError`).

- [ ] **Step 7: Implement update_documents in milvus_store.py**

Replace the stub:

```python
    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        raise NotImplementedError("Milvus update_documents is not implemented")
```

with:

```python
    async def update_documents(self, filter: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        if not update_data:
            return 0
        expr = self._filter_to_expr(filter)
        if not hasattr(self.client, "query"):
            raise NotImplementedError("Milvus client does not support query()")
        docs = await self.client.query(
            collection_name=self.collection,
            expr=expr,
            output_fields=["id", "vector", "content", "metadata"],
            limit=100000,
        )
        docs = docs or []
        if not docs:
            return 0
        new_content = update_data.get("content")
        update_meta = update_data.get("metadata")
        data = []
        for d in docs:
            metadata = d.get("metadata") or {}
            if isinstance(update_meta, dict):
                metadata = {**metadata, **update_meta}
            data.append({
                "id": d.get("id"),
                "vector": d.get("vector"),
                "content": new_content if isinstance(new_content, str) else d.get("content", ""),
                "metadata": metadata,
            })
        if hasattr(self.client, "upsert"):
            await self.client.upsert(collection_name=self.collection, fields_data=data)
        else:
            await self.client.insert(collection_name=self.collection, fields_data=data)
        return len(data)
```

- [ ] **Step 8: Run and verify all pass**

Run: `pytest tests/test_backends/test_milvus_store.py -v`
Expected: all pass (5 tests total: 3 from Task 8, 2 new).

- [ ] **Step 9: Commit**

```bash
git add vectra/backends/postgres_store.py vectra/backends/milvus_store.py tests/test_backends/test_postgres_store.py tests/test_backends/test_milvus_store.py
git commit -m "fix: implement update_documents for Postgres and Milvus stores"
```

---

### Task 10: Relicense to MIT

**Files:**
- Modify: `pyproject.toml:10`, `pyproject.toml:16`
- Modify: `LICENSE` (full replace)

**Interfaces:**
- Produces: `pyproject.toml`'s `license` field, its classifier, and the `LICENSE` file all agree on MIT — closing the three-way contradiction found in the audit.

- [ ] **Step 1: Fix the classifier**

Edit `pyproject.toml` line 16, change:

```toml
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
```

to:

```toml
    "License :: OSI Approved :: MIT License",
```

(Line 10, `license = { text = "MIT" }`, is already correct — leave it as-is.)

- [ ] **Step 2: Replace the LICENSE file**

Overwrite `LICENSE` with:

```
MIT License

Copyright (c) 2026 Abhishek N

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml LICENSE
git commit -m "fix: resolve license metadata contradiction, settle on MIT"
```

---

### Task 11: Fix the wrong pip install name in this repo's own README

**Files:**
- Modify: `README.md:129`, `README.md:131`

**Interfaces:**
- Produces: the README's own installation instructions match the real, live PyPI package name confirmed for this project: `vectra-rag-py`.

- [ ] **Step 1: Fix the install commands**

Edit `README.md` lines 128-132, change:

```
```bash
pip install vectra-py
# or
uv pip install vectra-py
```
```

to:

```
```bash
pip install vectra-rag-py
# or
uv pip install vectra-rag-py
```
```

- [ ] **Step 2: Confirm no other wrong references remain**

Run: `grep -n "pip install vectra-py" README.md`
Expected: no matches.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "fix: correct pip install command to vectra-rag-py in README"
```

---

## Self-Review Notes

- **Spec coverage:** relicense (Task 10), test suite covering config/RRF/MMR/all 5 backends (Tasks 1-8), telemetry default-off (Task 3), the three named CRUD bugs — Milvus delete count (Task 8), Postgres + Milvus `update_documents` (Task 9) — and the README install-name fix (Task 11, a bug found during plan research that the design spec didn't originally call out but belongs here since it's the same class of problem as the site's install-command bug) are all covered.
- **Placeholder scan:** no TBD/TODO; every step has runnable code.
- **Type consistency:** `update_documents` in both Postgres and Milvus stores returns an `int` count, matching the abstract signature in `vectra/interfaces.py:36`. Mock shapes in Tasks 4-9 match the exact `client_instance` calls read from each backend's source.
- **Scope note:** `postgres_store.py` has no SQL-identifier sanitization at all (confirmed absent repo-wide via search) — worse than the design spec assumed, since it described this as "hardening an existing allowlist." This plan does not add one; it's flagged for Phase 2 (security), which is where the spec places this class of fix.
