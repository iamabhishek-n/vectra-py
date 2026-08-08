# Phase 2 — vectra-py Security Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the security gaps identified in Phase 2 of the design spec for vectra-py, which is worse off than vectra-js in two specific ways found during Phase 1 planning/review: `PostgresVectorStore` has **zero** SQL-identifier sanitization (vectra-js has a regex allowlist; vectra-py's native Postgres store has none at all), and `MilvusVectorStore._filter_to_expr` string-interpolates filter keys and values into a Milvus expression with no escaping (an injection surface, same class of bug as SQL injection). Also enforces the guardrails schema at runtime, adds an ingestion file-size limit, wires dependency scanning into CI, corrects drifted telemetry docs, and publishes a SECURITY.md.

**Architecture:** Adds SQL-identifier sanitization to `postgres_store.py` mirroring the pattern already proven in vectra-js and in vectra-py's own `prisma_store.py` (which already has `_safe_ident`, just not applied eagerly at construction). Adds a `vectra/guardrails.py` module with pure, testable check functions, called at the top of `query_rag`. All other changes are small, targeted edits.

**Tech Stack:** Python ≥3.8, pytest (already set up from Phase 1), GitHub Actions, pip-audit.

## Global Constraints

- No direct-to-master commits, no force-push, no skipped hooks.
- Every task ends with `pytest` passing before moving to the next task.
- `prisma_store.py`'s existing `_safe_ident` method is untouched by this plan — it already works correctly (validates lazily, per-method-call). This plan only adds the missing equivalent to `postgres_store.py` and fixes the escaping gap in `milvus_store.py`.
- Guardrail violations raise a plain `ValueError` with a message prefixed `GuardrailViolation:` — callers catch it like any other error from `query_rag`; no special exception class.
- PII/content-filter detection is regex/keyword-based, matching vectra-js's approach for feature parity — not ML-based, not comprehensive moderation. State this limitation in code comments.

---

### Task 1: SQL-identifier sanitization for PostgresVectorStore

**Files:**
- Modify: `vectra/backends/postgres_store.py:1-19`
- Create: `tests/test_backends/test_postgres_identifier_safety.py`

**Interfaces:**
- Produces: `vectra.backends.postgres_store` module now exports `is_safe_identifier(value: str) -> bool` and `assert_safe_identifier(value: str, label: str) -> str` (raises `ValueError` on an unsafe value, otherwise returns the value unchanged). `PostgresVectorStore.__init__` now validates `table_name`, and the three `column_map` entries it uses, eagerly at construction — matching the pattern already used by vectra-js's `PostgresVectorStore` and `PrismaVectorStore`, and by vectra-py's own `PrismaVectorStore._safe_ident` (which validates lazily per-call; this task validates eagerly at construction instead, which is strictly safer since it fails before any query is built rather than on first use).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backends/test_postgres_identifier_safety.py`:

```python
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
```

- [ ] **Step 2: Run and verify they fail**

Run: `pytest tests/test_backends/test_postgres_identifier_safety.py -v`
Expected: FAIL — `ImportError: cannot import name 'is_safe_identifier'`.

- [ ] **Step 3: Add the sanitization helpers and wire them into the constructor**

Edit `vectra/backends/postgres_store.py`, add near the top of the file (after the existing imports, before `def to_db_vector`):

```python
import re

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_safe_identifier(value: str) -> bool:
    return isinstance(value, str) and bool(_SAFE_IDENTIFIER_RE.fullmatch(value))


def assert_safe_identifier(value: str, label: str) -> str:
    if not is_safe_identifier(value):
        raise ValueError(f"Unsafe SQL identifier for {label}: {value!r}")
    return value
```

Then change the constructor (lines 12-19), from:

```python
    def __init__(self, config: Any):
        self.config = config
        self.client = config.client_instance
        self.table_name = getattr(config, 'table_name', 'document')
        self.column_map = getattr(config, 'column_map', {})
        self.c_content = self.column_map.get('content', 'content')
        self.c_meta = self.column_map.get('metadata', 'metadata')
        self.c_vector = self.column_map.get('vector', 'vector')
```

to:

```python
    def __init__(self, config: Any):
        self.config = config
        self.client = config.client_instance
        self.table_name = assert_safe_identifier(getattr(config, 'table_name', 'document') or 'document', 'table_name')
        self.column_map = getattr(config, 'column_map', {})
        self.c_content = assert_safe_identifier(self.column_map.get('content', 'content'), 'column_map.content')
        self.c_meta = assert_safe_identifier(self.column_map.get('metadata', 'metadata'), 'column_map.metadata')
        self.c_vector = assert_safe_identifier(self.column_map.get('vector', 'vector'), 'column_map.vector')
```

- [ ] **Step 4: Run and verify all pass**

Run: `pytest tests/test_backends/test_postgres_identifier_safety.py -v`
Expected: 8 tests pass.

- [ ] **Step 5: Run the full existing Postgres test file to confirm nothing broke**

Run: `pytest tests/test_backends/test_postgres_store.py -v`
Expected: all pass — every existing test in that file uses the safe table name `"document"`, so eager validation shouldn't reject anything already covered.

- [ ] **Step 6: Commit**

```bash
git add vectra/backends/postgres_store.py tests/test_backends/test_postgres_identifier_safety.py
git commit -m "fix: add SQL identifier sanitization to PostgresVectorStore (was completely absent)"
```

---

### Task 2: Escape filter values in MilvusVectorStore._filter_to_expr

**Files:**
- Modify: `vectra/backends/milvus_store.py:60-71`
- Create: `tests/test_backends/test_milvus_filter_safety.py`

**Interfaces:**
- Produces: `MilvusVectorStore._filter_to_expr(filter)` now rejects unsafe metadata keys (raises `ValueError`) and escapes backslashes/double-quotes in string values before interpolating them into the Milvus boolean expression, closing an expression-injection surface where a caller-supplied filter key or value could break out of the intended `metadata["key"] == "value"` structure.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backends/test_milvus_filter_safety.py`:

```python
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
```

- [ ] **Step 2: Run and verify they fail**

Run: `pytest tests/test_backends/test_milvus_filter_safety.py -v`
Expected: FAIL on the escaping and unsafe-key tests (current code interpolates without escaping or key validation).

- [ ] **Step 3: Fix _filter_to_expr**

Edit `vectra/backends/milvus_store.py`, add near the top of the file (after the existing imports):

```python
import re

_SAFE_FILTER_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _escape_milvus_string(value: str) -> str:
    return value.replace('\\', '\\\\').replace('"', '\\"')
```

Then change `_filter_to_expr` (currently lines 60-71), from:

```python
    def _filter_to_expr(self, filter: Optional[Dict[str, Any]]) -> str:
        if not filter:
            return ""
        parts: List[str] = []
        for k, v in filter.items():
            if isinstance(v, str):
                parts.append(f'metadata["{k}"] == "{v}"')
            elif isinstance(v, bool):
                parts.append(f'metadata["{k}"] == {str(v).lower()}')
            elif isinstance(v, (int, float)):
                parts.append(f'metadata["{k}"] == {v}')
        return " and ".join(parts)
```

to:

```python
    def _filter_to_expr(self, filter: Optional[Dict[str, Any]]) -> str:
        if not filter:
            return ""
        parts: List[str] = []
        for k, v in filter.items():
            if not _SAFE_FILTER_KEY_RE.fullmatch(str(k)):
                raise ValueError(f"Unsafe filter key for Milvus expression: {k!r}")
            if isinstance(v, str):
                parts.append(f'metadata["{k}"] == "{_escape_milvus_string(v)}"')
            elif isinstance(v, bool):
                parts.append(f'metadata["{k}"] == {str(v).lower()}')
            elif isinstance(v, (int, float)):
                parts.append(f'metadata["{k}"] == {v}')
        return " and ".join(parts)
```

- [ ] **Step 4: Run and verify all pass**

Run: `pytest tests/test_backends/test_milvus_filter_safety.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Run the full existing Milvus test file to confirm nothing broke**

Run: `pytest tests/test_backends/test_milvus_store.py -v`
Expected: all pass — existing tests don't exercise `_filter_to_expr` with unsafe keys, so this is purely additive.

- [ ] **Step 6: Commit**

```bash
git add vectra/backends/milvus_store.py tests/test_backends/test_milvus_filter_safety.py
git commit -m "fix: escape filter values and reject unsafe keys in MilvusVectorStore._filter_to_expr"
```

---

### Task 3: Guardrails module — max_query_length and block_pii

**Files:**
- Create: `vectra/guardrails.py`
- Create: `tests/test_guardrails.py`

**Interfaces:**
- Produces: `check_guardrails(query: str, guardrails_config) -> None` — raises `ValueError` on violation. `guardrails_config` is a `GuardrailConfig` instance (or `None`) from `vectra/config.py:109-114`. This task implements `max_query_length` and `block_pii` only; Task 4 adds `content_filter`. `block_off_topic`/`hallucination_check` stay unenforced (Phase 3 feature work, needs semantic classification).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_guardrails.py`:

```python
import pytest
from vectra.guardrails import check_guardrails
from vectra.config import GuardrailConfig


class TestMaxQueryLength:
    def test_allows_query_at_or_under_limit(self):
        check_guardrails("a" * 2000, GuardrailConfig(max_query_length=2000))

    def test_rejects_query_over_limit(self):
        with pytest.raises(ValueError, match="GuardrailViolation: query exceeds max_query_length"):
            check_guardrails("a" * 2001, GuardrailConfig(max_query_length=2000))

    def test_does_nothing_when_guardrails_config_is_none(self):
        check_guardrails("a" * 100000, None)


class TestBlockPii:
    def test_allows_ordinary_query_when_off(self):
        check_guardrails("contact me at john@example.com", GuardrailConfig(block_pii=False))

    def test_rejects_email_when_on(self):
        with pytest.raises(ValueError, match="GuardrailViolation: possible PII detected"):
            check_guardrails("contact me at john@example.com", GuardrailConfig(block_pii=True))

    def test_rejects_phone_number_when_on(self):
        with pytest.raises(ValueError, match="GuardrailViolation: possible PII detected"):
            check_guardrails("call me at 555-123-4567", GuardrailConfig(block_pii=True))

    def test_rejects_ssn_shaped_number_when_on(self):
        with pytest.raises(ValueError, match="GuardrailViolation: possible PII detected"):
            check_guardrails("my ssn is 123-45-6789", GuardrailConfig(block_pii=True))

    def test_rejects_long_digit_run_when_on(self):
        with pytest.raises(ValueError, match="GuardrailViolation: possible PII detected"):
            check_guardrails("card number 4111111111111111", GuardrailConfig(block_pii=True))

    def test_allows_query_with_no_pii_when_on(self):
        check_guardrails("what is the refund policy?", GuardrailConfig(block_pii=True))
```

- [ ] **Step 2: Run and verify they fail**

Run: `pytest tests/test_guardrails.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vectra.guardrails'`.

- [ ] **Step 3: Write the implementation**

Create `vectra/guardrails.py`:

```python
import re

# Best-effort, regex-based PII detection — not a comprehensive moderation
# system. Flags emails, phone numbers, SSN-shaped numbers, and long digit
# runs (credit-card-shaped). False positives are possible on legitimate
# long numeric identifiers; that's an accepted tradeoff of a fast, local,
# no-external-dependency check. Mirrors vectra-js's src/guardrails.js for
# feature parity between the two SDKs.
PII_PATTERNS = [
    ("email", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("phone", re.compile(r"(\+?\d{1,2}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("long_digit_run", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
]


def check_guardrails(query: str, guardrails_config) -> None:
    if guardrails_config is None:
        return
    text = str(query or "")

    max_len = getattr(guardrails_config, "max_query_length", None)
    if max_len and len(text) > max_len:
        raise ValueError(f"GuardrailViolation: query exceeds max_query_length ({len(text)} > {max_len})")

    if getattr(guardrails_config, "block_pii", False):
        for name, pattern in PII_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"GuardrailViolation: possible PII detected ({name})")
```

- [ ] **Step 4: Run and verify they pass**

Run: `pytest tests/test_guardrails.py -v`
Expected: 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add vectra/guardrails.py tests/test_guardrails.py
git commit -m "feat: enforce max_query_length and block_pii guardrails"
```

---

### Task 4: Guardrails module — content_filter, then wire into query_rag

**Files:**
- Modify: `vectra/guardrails.py`
- Modify: `tests/test_guardrails.py`
- Modify: `vectra/core.py:1` (add import), `vectra/core.py:609` (call at top of `query_rag`)
- Create: `tests/test_core_guardrails.py`

**Interfaces:**
- Consumes: `check_guardrails` from Task 3.
- Produces: `VectraClient.query_rag` now raises synchronously (before any embedding/LLM call) when a guardrail is violated.

- [ ] **Step 1: Write the failing content_filter tests**

Append to `tests/test_guardrails.py`:

```python
class TestContentFilter:
    def test_allows_ordinary_query_when_off(self):
        check_guardrails("how do I make a sandwich", GuardrailConfig(content_filter=False))

    def test_rejects_blocked_term_when_on(self):
        with pytest.raises(ValueError, match="GuardrailViolation: query blocked by content filter"):
            check_guardrails("how to make a bomb at home", GuardrailConfig(content_filter=True))

    def test_is_case_insensitive(self):
        with pytest.raises(ValueError, match="GuardrailViolation: query blocked by content filter"):
            check_guardrails("HOW TO MAKE A BOMB", GuardrailConfig(content_filter=True))

    def test_allows_unrelated_query_when_on(self):
        check_guardrails("what vector stores does this SDK support?", GuardrailConfig(content_filter=True))
```

- [ ] **Step 2: Run and verify they fail**

Run: `pytest tests/test_guardrails.py -v`
Expected: the 4 new tests FAIL (content_filter not yet implemented).

- [ ] **Step 3: Add content_filter to the guardrails module**

Edit `vectra/guardrails.py`, add after the `PII_PATTERNS` list:

```python
# Minimal seed list of clearly harmful query patterns. This is a baseline,
# not exhaustive content moderation — extend DEFAULT_BLOCKED_TERMS for your
# deployment's needs, or replace check_guardrails' content_filter branch
# with an LLM-based classifier if you need semantic (not just keyword)
# coverage. Mirrors vectra-js's src/guardrails.js DEFAULT_BLOCKED_TERMS.
DEFAULT_BLOCKED_TERMS = [
    "how to make a bomb",
    "how to build a bomb",
    "how to make explosives",
    "how to synthesize a bioweapon",
]
```

Then edit `check_guardrails`, adding after the `block_pii` block:

```python
    if getattr(guardrails_config, "content_filter", False):
        lower = text.lower()
        for term in DEFAULT_BLOCKED_TERMS:
            if term in lower:
                raise ValueError("GuardrailViolation: query blocked by content filter")
```

- [ ] **Step 4: Run and verify all guardrails tests pass**

Run: `pytest tests/test_guardrails.py -v`
Expected: 13 tests pass.

- [ ] **Step 5: Write the failing integration test for query_rag**

Create `tests/test_core_guardrails.py`:

```python
import pytest
from unittest.mock import AsyncMock
from vectra.core import VectraClient
from vectra.config import VectraConfig, EmbeddingConfig, LLMConfig, DatabaseConfig, ProviderType, GuardrailConfig


def make_config(guardrails):
    return VectraConfig(
        embedding=EmbeddingConfig(provider=ProviderType.OPENAI, api_key="test-key"),
        llm=LLMConfig(provider=ProviderType.OPENAI, api_key="test-key", model_name="gpt-4o-mini"),
        database=DatabaseConfig(type="chroma", client_instance=object()),
        guardrails=guardrails,
    )


class TestQueryRagGuardrailEnforcement:
    async def test_rejects_over_length_query_before_any_embedding_call(self):
        client = VectraClient(make_config(GuardrailConfig(max_query_length=10)))
        client.embedder.embed_query = AsyncMock()
        with pytest.raises(ValueError, match="GuardrailViolation: query exceeds max_query_length"):
            await client.query_rag("this query is way too long for the limit")
        client.embedder.embed_query.assert_not_called()

    async def test_rejects_pii_query_before_any_embedding_call(self):
        client = VectraClient(make_config(GuardrailConfig(block_pii=True)))
        client.embedder.embed_query = AsyncMock()
        with pytest.raises(ValueError, match="GuardrailViolation: possible PII detected"):
            await client.query_rag("email me at test@example.com")
        client.embedder.embed_query.assert_not_called()
```

*(Note: `VectraClient.__init__` unconditionally constructs a `SQLiteLogger` — this writes a local SQLite file. That's pre-existing behavior unrelated to this task; if it causes test pollution, use `tmp_path`/`monkeypatch` to redirect `config.observability.sqlite_path` to a temp location rather than changing the source.)*

- [ ] **Step 6: Run and verify they fail**

Run: `pytest tests/test_core_guardrails.py -v`
Expected: FAIL — `check_guardrails` not yet called from `query_rag`, so no exception is raised and the mocked `embed_query` gets called instead.

- [ ] **Step 7: Wire check_guardrails into query_rag**

Edit `vectra/core.py` near the top of the file, add to the existing import block (alongside the other `from .config import ...` / `from .telemetry import ...` lines):

```python
from .guardrails import check_guardrails
```

Edit `vectra/core.py:609-611`, change:

```python
    async def query_rag(self, query: str, filter: Optional[Dict] = None, stream: bool = False, session_id: Optional[str] = None) -> Dict[str, Any] | AsyncGenerator[str, None]:
        trace_id = str(uuid.uuid4())
        root_span_id = str(uuid.uuid4())
```

to:

```python
    async def query_rag(self, query: str, filter: Optional[Dict] = None, stream: bool = False, session_id: Optional[str] = None) -> Dict[str, Any] | AsyncGenerator[str, None]:
        check_guardrails(query, self.config.guardrails)
        trace_id = str(uuid.uuid4())
        root_span_id = str(uuid.uuid4())
```

- [ ] **Step 8: Run and verify both pass**

Run: `pytest tests/test_core_guardrails.py -v`
Expected: 2 tests pass.

- [ ] **Step 9: Run the full suite**

Run: `pytest`
Expected: all tests pass (Phase 1's 36, plus this plan's tests so far: 8 identifier + 4 filter + 13 guardrails + 2 core = 27, running total 63).

- [ ] **Step 10: Commit**

```bash
git add vectra/guardrails.py tests/test_guardrails.py vectra/core.py tests/test_core_guardrails.py
git commit -m "feat: enforce content_filter guardrail and wire all guardrails into query_rag"
```

---

### Task 5: Ingestion file-size limit

**Files:**
- Modify: `vectra/config.py:35-37`
- Modify: `vectra/core.py:208-215`
- Create: `tests/test_ingestion_limits.py`

**Interfaces:**
- Produces: `IngestionConfig.max_file_size_bytes` (Pydantic field, default `52428800` = 50MB). The file-processing loop in `ingest_batch` now raises before hashing if a file's size exceeds the configured limit.

- [ ] **Step 1: Add the config field**

Edit `vectra/config.py` lines 35-37, change:

```python
class IngestionConfig(BaseModel):
    rate_limit_enabled: bool = False
    concurrency_limit: int = 5
```

to:

```python
class IngestionConfig(BaseModel):
    rate_limit_enabled: bool = False
    concurrency_limit: int = 5
    max_file_size_bytes: int = 52428800
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_ingestion_limits.py`:

```python
import os
import pytest
from vectra.core import VectraClient
from vectra.config import VectraConfig, EmbeddingConfig, LLMConfig, DatabaseConfig, ProviderType, IngestionConfig


def make_client(max_file_size_bytes=None):
    kwargs = {}
    if max_file_size_bytes is not None:
        kwargs["ingestion"] = IngestionConfig(max_file_size_bytes=max_file_size_bytes)
    return VectraClient(VectraConfig(
        embedding=EmbeddingConfig(provider=ProviderType.OPENAI, api_key="test-key"),
        llm=LLMConfig(provider=ProviderType.OPENAI, api_key="test-key", model_name="gpt-4o-mini"),
        database=DatabaseConfig(type="chroma", client_instance=object()),
        **kwargs,
    ))


class TestIngestionFileSizeLimit:
    async def test_rejects_file_over_configured_limit(self, tmp_path):
        big_file = tmp_path / "big.txt"
        big_file.write_bytes(b"x" * 2000)
        client = make_client(max_file_size_bytes=1000)
        with pytest.raises(Exception, match="File exceeds maximum allowed size"):
            await client.ingest_batch([str(big_file)])

    async def test_accepts_file_within_limit(self, tmp_path, monkeypatch):
        small_file = tmp_path / "small.txt"
        small_file.write_text("hello world")
        client = make_client(max_file_size_bytes=1000)
        # Stub out everything past the size check so this test only exercises the guard.
        client.processor.load_document = lambda *a, **k: "hello world"
        client.processor.process = lambda *a, **k: ["hello world"]
        client.embedder.embed_documents = __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(return_value=[[0.1]])
        client.vector_store.add_documents = __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock()
        await client.ingest_batch([str(small_file)])  # should not raise for size reasons

    async def test_uses_default_50mb_limit_when_not_configured(self, tmp_path):
        big_file = tmp_path / "huge.txt"
        big_file.write_bytes(b"x" * 100)  # small file, but we assert the default is applied via config inspection instead
        client = make_client()
        assert client.config.ingestion.max_file_size_bytes == 52428800
```

- [ ] **Step 3: Run and verify the first test fails**

Run: `pytest tests/test_ingestion_limits.py -v`
Expected: `test_rejects_file_over_configured_limit` FAILS (no size check exists yet, so it proceeds past the point where the size-limit error would be raised and fails somewhere else, e.g. a real embedding API call attempt). `test_uses_default_50mb_limit_when_not_configured` should already PASS (pure config default check, no behavior change needed for it). `test_accepts_file_within_limit` may fail or pass depending on mocking completeness — the important signal here is the first test.

- [ ] **Step 4: Add the size check**

Edit `vectra/core.py`, inside the `ingest_batch` file loop (starts around line 208), change:

```python
        for file_path in all_files:
            abs_path = os.path.abspath(file_path)
            try:
                size = int(os.path.getsize(file_path))
                mtime = int(os.path.getmtime(file_path))
            except Exception:
                size = 0
                mtime = 0
```

to:

```python
        for file_path in all_files:
            abs_path = os.path.abspath(file_path)
            try:
                size = int(os.path.getsize(file_path))
                mtime = int(os.path.getmtime(file_path))
            except Exception:
                size = 0
                mtime = 0

            max_size = self.config.ingestion.max_file_size_bytes if self.config.ingestion else 52428800
            if size > max_size:
                raise ValueError(f"File exceeds maximum allowed size: {file_path} ({size} bytes > {max_size} bytes limit)")
```

- [ ] **Step 5: Run and verify all pass**

Run: `pytest tests/test_ingestion_limits.py -v`
Expected: 3 tests pass.

- [ ] **Step 6: Run the full suite**

Run: `pytest`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add vectra/config.py vectra/core.py tests/test_ingestion_limits.py
git commit -m "feat: add configurable max file size limit for ingestion"
```

---

### Task 6: Dependency vulnerability scanning in CI

**Files:**
- Modify: `pyproject.toml` (dev optional-dependencies)
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `pip-audit` added to the `dev` extras; a GitHub Actions workflow running `pytest` and `pip-audit` on every push/PR against `master`.

- [ ] **Step 1: Add pip-audit to dev dependencies**

Edit `pyproject.toml`'s `[project.optional-dependencies]` section, change:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24"]
```

to:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "pip-audit>=2.7"]
```

- [ ] **Step 2: Create the CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -e ".[dev]"
      - run: pytest
      - run: pip-audit
```

- [ ] **Step 3: Verify locally**

Run: `pip install -e ".[dev]"`
Run: `pytest`
Expected: all tests pass, confirming the dependency addition didn't break anything.
Run: `pip-audit`
Expected: runs to completion (it may report pre-existing advisories in third-party dependencies unrelated to this change — that's a finding for a human to triage, not something this task fixes; the task's job is wiring the scan in, not remediating every transitive advisory).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml
git commit -m "ci: run tests and pip-audit on every push and pull request"
```

---

### Task 7: Correct telemetry documentation drift and publish SECURITY.md

**Files:**
- Modify: `README.md:524-525`
- Create: `SECURITY.md`

**Interfaces:**
- Produces: README's telemetry event list matches the actual fields sent (verified against every `telemetry.track(...)` call site in `vectra/core.py`, `vectra/cli.py`, and `vectra/webconfig_server.py`). `SECURITY.md` gives a responsible-disclosure contact.

- [ ] **Step 1: Fix the telemetry documentation drift**

Edit `README.md` lines 524-525, change:

```
    * `ingest_started/completed`: Source type, chunking strategy, duration bucket, chunk count bucket.
    * `query_executed`: Retrieval strategy, query mode (rag), result count, latency bucket.
```

to:

```
    * `ingest_batch_started`: File count, ingestion mode.
    * `ingest_batch_completed`: File count, chunk count, duration in milliseconds.
    * `query_executed`: Retrieval strategy, query mode (rag), reranking enabled, streaming, memory used, result count. No latency is currently tracked on this event.
```

- [ ] **Step 2: Create SECURITY.md**

Create `SECURITY.md`:

```markdown
# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in vectra-py (published as `vectra-rag-py`), please report it privately rather than opening a public GitHub issue.

Email: astroabhi.abhi@gmail.com

Please include:
- A description of the vulnerability and its potential impact
- Steps to reproduce (a minimal example is ideal)
- The version of `vectra-rag-py` affected

We aim to acknowledge reports within 5 business days. Once a fix is available, we'll coordinate on a disclosure timeline with you before making details public.

## Supported Versions

Only the latest published version on PyPI receives security fixes.

## Scope

This SDK orchestrates calls to external vector databases and LLM providers that you configure and supply credentials for — it does not host or store your data itself. Vulnerabilities in this repo's own code (e.g. SQL/expression construction, input validation, dependency issues) are in scope. Misconfiguration of the external services you connect it to is not.
```

- [ ] **Step 3: Run the full suite one last time**

Run: `pytest`
Expected: all tests pass (this task only touches documentation).

- [ ] **Step 4: Commit**

```bash
git add README.md SECURITY.md
git commit -m "docs: correct telemetry event documentation, add SECURITY.md"
```

---

## Self-Review Notes

- **Spec coverage:** the two named security gaps found during Phase 1 (Postgres identifier sanitization built from scratch, Milvus filter-expression escaping) are covered in Tasks 1-2; guardrails enforcement in Tasks 3-4; ingestion limits in Task 5; dependency scanning in Task 6; telemetry docs + SECURITY.md in Task 7.
- **Placeholder scan:** no TBD/TODO; every step has runnable code.
- **Type consistency:** `check_guardrails(query, guardrails_config)` signature identical across Tasks 3, 4, and its call site. `is_safe_identifier`/`assert_safe_identifier` names match vectra-js's equivalents for cross-SDK consistency, adapted to Python naming (snake_case).
- **Parity note:** this plan deliberately mirrors vectra-js's Phase 2 plan's design decisions (same PII patterns, same seed content-filter terms, same 50MB default ingestion limit, same GuardrailViolation-prefixed error convention) so the two SDKs stay behaviorally consistent — a stated design-spec goal.
