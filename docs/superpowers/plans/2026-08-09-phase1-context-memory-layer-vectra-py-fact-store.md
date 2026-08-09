# Context & Memory Layer Phase 1 — vectra-py Fact Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mirror vectra-js's Phase 1 fact-store implementation in Python — same architecture, same task shape, same public API, from `docs/superpowers/specs/2026-08-09-context-memory-layer-design.md` (Component 1). Executed in lockstep with the vectra-js plan (`vectra-js/docs/superpowers/plans/2026-08-09-phase1-context-memory-layer-vectra-js-fact-store.md`) — same 7 tasks, same order, same review cadence.

**Architecture:** New `FactStore` class in `vectra/memory/fact_store.py`, following the exact structural pattern of `PostgresHistory` in `vectra/memory.py` (`_safe_ident` validation, `client.execute`/`client.fetch` via `asyncpg`, the `_get_connection()`/`DummyContext` pattern already established in `vectra/backends/postgres_store.py` — reuse `assert_safe_identifier` from that file rather than duplicating identifier-safety logic). Postgres/pgvector only in this phase.

**Tech Stack:** `asyncpg`, pytest (`pytest-asyncio` auto mode, already configured), the existing embedder/LLM backend abstractions (`self.embedder.embed_query`/`embed_documents`, backend `generate(prompt, sys)`), `uuid.uuid4()` (Python stdlib, no new dependency).

## Global Constraints

- Postgres/pgvector only in this phase. No Neo4j/SurrealDB/graph-native backends — future work per the spec.
- Never delete a fact row. Superseding sets `invalid_at = NOW()` on the old row and inserts a new one.
- All identifiers go through `assert_safe_identifier` (already defined in `vectra/backends/postgres_store.py` — import and reuse it, do not duplicate).
- All queries parameterized (`$1`, `$2`, ... via asyncpg's `*args` positional style) — no string-interpolated user-controlled values, only validated identifiers.
- Fact extraction (write path's LLM call) must be fail-soft: malformed/unparseable LLM output must not raise or crash the caller's turn.
- Read path must be a single batched query — no per-fact/per-entity round trips.
- Every task ends with `pytest` passing before moving to the next task.
- Do not touch `vectra/memory.py`'s existing `InMemoryHistory`/`RedisHistory`/`PostgresHistory` classes, `query_rag`'s existing history wiring, or any Phase 3/4/5 work already merged — purely additive.

---

### Task 1: FactStore schema — table creation, indexes

**Files:**
- Create: `vectra/memory/fact_store.py`
- Create: `vectra/memory/__init__.py` (if `vectra/memory` doesn't already exist as a package — check first; `vectra/memory.py` is currently a single module file, not a package, so this task converts the layout: create `vectra/memory/` directory, move the existing `vectra/memory.py` content into `vectra/memory/__init__.py` so `from vectra.memory import InMemoryHistory, ...` keeps working unchanged, then add `fact_store.py` as a sibling module inside the new package. Confirm this via a quick check of every existing `from .memory import` / `from vectra.memory import` call site across the codebase and `tests/`, and verify they all still pass after the conversion, before writing any new FactStore code.)
- Create: `tests/test_memory/test_fact_store_ensure_indexes.py`

**Interfaces:**
- Produces: `class FactStore` with `__init__(self, config)` where `config` has `client_instance`, `table_name` (default `'VectraFact'`), and `async def ensure_indexes(self, dimensions: int = 1536)` creating the fact table and indexes if they don't exist.

- [ ] **Step 1: Convert `vectra/memory.py` to a package**

Run `mkdir vectra/memory_pkg_tmp` is NOT the approach — instead: `git mv vectra/memory.py vectra/memory/__init__.py` requires `vectra/memory/` to not already exist as a file path conflict; do this via: create directory `vectra/memory/`, then `git mv vectra/memory.py vectra/memory/__init__.py`. Run the full test suite immediately after (`pytest`) to confirm every existing `from .memory import ...` / `from vectra.memory import ...` reference across `vectra/core.py` and all test files still resolves — Python package `__init__.py` re-exports work transparently for `from vectra.memory import X` callers with zero code changes needed at call sites, but confirm this is actually true in this codebase before proceeding (some codebases have `from vectra import memory; memory.something()` style access that could behave differently — grep for both import styles).

- [ ] **Step 2: Write the failing test**

Create `tests/test_memory/test_fact_store_ensure_indexes.py`:

```python
import pytest
from unittest.mock import AsyncMock
from vectra.memory.fact_store import FactStore


class FakeConn:
    def __init__(self):
        self.queries = []
        self.execute = AsyncMock(side_effect=self._record)

    async def _record(self, q, *args):
        self.queries.append(q)
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeClient:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return self._conn


def make_config(client):
    return type("Cfg", (), {"client_instance": client, "table_name": "VectraFact"})()


class TestFactStoreEnsureIndexes:
    async def test_creates_table_with_given_dimension_and_indexes(self):
        conn = FakeConn()
        store = FactStore(make_config(FakeClient(conn)))

        await store.ensure_indexes(768)

        create_table = next((q for q in conn.queries if "CREATE TABLE IF NOT EXISTS" in q), None)
        assert create_table is not None
        assert "vector(768)" in create_table
        assert '"subject"' in create_table
        assert '"predicate"' in create_table
        assert '"object"' in create_table
        assert '"valid_at"' in create_table
        assert '"invalid_at"' in create_table
        assert '"session_id"' in create_table

        vec_index = next((q for q in conn.queries if "USING hnsw" in q or "USING ivfflat" in q), None)
        assert vec_index is not None

        temporal_index = next((q for q in conn.queries if "index" in q.lower() and '"session_id"' in q and '"valid_at"' in q), None)
        assert temporal_index is not None

    def test_rejects_unsafe_table_name(self):
        with pytest.raises(ValueError, match="[Uu]nsafe SQL identifier"):
            FactStore(make_config(FakeClient(FakeConn())).__class__ if False else type("Cfg", (), {"client_instance": FakeClient(FakeConn()), "table_name": "Fact; DROP TABLE users;--"})())
```

- [ ] **Step 2b: Fix the deliberately-awkward second test**

The `test_rejects_unsafe_table_name` snippet above has an intentionally convoluted config-construction one-liner to avoid a second `make_config` call needing a different table name — before running, simplify it to a clean two-line form:

```python
    def test_rejects_unsafe_table_name(self):
        bad_config = type("Cfg", (), {"client_instance": FakeClient(FakeConn()), "table_name": "Fact; DROP TABLE users;--"})()
        with pytest.raises(ValueError, match="[Uu]nsafe SQL identifier"):
            FactStore(bad_config)
```

- [ ] **Step 3: Run and verify it fails**

Run: `pytest tests/test_memory/test_fact_store_ensure_indexes.py -v`
Expected: FAIL — `vectra/memory/fact_store.py` doesn't exist yet.

- [ ] **Step 4: Implement**

Create `vectra/memory/fact_store.py`:

```python
from typing import Any, Dict, List, Optional
from ..backends.postgres_store import assert_safe_identifier


class FactStore:
    def __init__(self, config: Any):
        self.client = config.client_instance
        self.table_name = assert_safe_identifier(getattr(config, 'table_name', 'VectraFact') or 'VectraFact', 'table_name')
        self.llm = getattr(config, 'llm', None)
        self.embedder = getattr(config, 'embedder', None)

    def _get_connection(self):
        if hasattr(self.client, 'acquire'):
            return self.client.acquire()

        class DummyContext:
            def __init__(self, conn): self.conn = conn
            async def __aenter__(self): return self.conn
            async def __aexit__(self, *args): pass
        return DummyContext(self.client)

    async def ensure_indexes(self, dimensions: int = 1536):
        t = self.table_name
        dim = dimensions or 1536
        async with self._get_connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f'''CREATE TABLE IF NOT EXISTS "{t}" (
                "id" TEXT PRIMARY KEY,
                "session_id" TEXT NOT NULL,
                "subject" TEXT NOT NULL,
                "predicate" TEXT NOT NULL,
                "object" TEXT NOT NULL,
                "embedding" vector({dim}),
                "valid_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                "invalid_at" TIMESTAMP WITH TIME ZONE,
                "source_message_id" TEXT,
                "created_at" TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )''')
            try:
                await conn.execute(f'CREATE INDEX IF NOT EXISTS "{t}_vec_idx" ON "{t}" USING hnsw ("embedding" vector_cosine_ops)')
            except Exception:
                try:
                    await conn.execute(f'CREATE INDEX IF NOT EXISTS "{t}_vec_idx" ON "{t}" USING ivfflat ("embedding" vector_cosine_ops)')
                except Exception:
                    pass
            await conn.execute(f'CREATE INDEX IF NOT EXISTS "{t}_session_temporal_idx" ON "{t}" ("session_id", "valid_at", "invalid_at")')
            await conn.execute(f'CREATE INDEX IF NOT EXISTS "{t}_subject_predicate_idx" ON "{t}" ("session_id", "subject", "predicate")')
```

- [ ] **Step 5: Run and verify it passes**

Run: `pytest tests/test_memory/test_fact_store_ensure_indexes.py -v`
Expected: 2 tests pass.

- [ ] **Step 6: Run the full suite**

Run: `pytest`
Expected: all tests pass — this is the step that proves Step 1's package conversion didn't break any existing `memory` import across the codebase.

- [ ] **Step 7: Commit**

```bash
git add vectra/memory/ tests/test_memory/test_fact_store_ensure_indexes.py
git rm vectra/memory.py 2>/dev/null || true
git commit -m "feat: convert memory.py to a package, add FactStore schema and index creation"
```
(The `git rm` is a safety net only — `git mv` in Step 1 should have already staged the rename correctly; confirm with `git status` before committing that there's no leftover untracked `vectra/memory.py`.)

---

### Task 2: Fact extraction on write (LLM-driven, fail-soft)

**Files:**
- Modify: `vectra/memory/fact_store.py`
- Create: `tests/test_memory/test_fact_store_write.py`

**Interfaces:**
- Consumes: `FactStore` from Task 1, plus `llm` (object with `async def generate(self, prompt, sys)`) and `embedder` (object with `async def embed_documents(self, texts)`) passed via `config`.
- Produces: `async def write(self, session_id: str, turn: Dict[str, str])` where `turn = {"user_message": ..., "assistant_message": ...}`. Extracts triples via one `generate()` call, embeds each, inserts as new facts (invalidation is Task 3).

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory/test_fact_store_write.py`:

```python
import json
import pytest
from unittest.mock import AsyncMock
from vectra.memory.fact_store import FactStore


class FakeConn:
    def __init__(self):
        self.inserted = []
        self.execute = AsyncMock(side_effect=self._record)

    async def _record(self, q, *args):
        if "INSERT INTO" in q:
            self.inserted.append(args)
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeClient:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return self._conn


def make_store(llm_response, embed_response=None):
    conn = FakeConn()
    llm = type("LLM", (), {"generate": AsyncMock(return_value=llm_response)})()
    embedder = type("Embedder", (), {"embed_documents": AsyncMock(side_effect=lambda texts: [embed_response or [0.1, 0.2] for _ in texts])})()
    config = type("Cfg", (), {
        "client_instance": FakeClient(conn), "table_name": "VectraFact",
        "llm": llm, "embedder": embedder,
    })()
    return FactStore(config), conn, llm, embedder


class TestFactStoreWrite:
    async def test_extracts_triples_and_inserts_as_new_facts(self):
        store, conn, llm, embedder = make_store(
            json.dumps({"facts": [{"subject": "user", "predicate": "likes", "object": "coffee"}]})
        )

        await store.write("session-1", {"user_message": "I really like coffee", "assistant_message": "Noted!"})

        llm.generate.assert_called_once()
        embedder.embed_documents.assert_called_once()
        assert len(conn.inserted) == 1
        args = conn.inserted[0]
        assert args[1] == "session-1"
        assert args[2] == "user"
        assert args[3] == "likes"
        assert args[4] == "coffee"

    async def test_extracts_multiple_triples_from_one_turn(self):
        store, conn, _, _ = make_store(
            json.dumps({"facts": [
                {"subject": "user", "predicate": "likes", "object": "coffee"},
                {"subject": "user", "predicate": "lives_in", "object": "Berlin"},
            ]})
        )

        await store.write("session-1", {"user_message": "I like coffee and live in Berlin", "assistant_message": "Cool!"})

        assert len(conn.inserted) == 2

    async def test_does_not_raise_on_malformed_json(self):
        store, conn, _, _ = make_store("not json at all {{{")

        await store.write("session-1", {"user_message": "hi", "assistant_message": "hello"})

        assert len(conn.inserted) == 0

    async def test_does_not_raise_on_empty_facts_array(self):
        store, conn, _, _ = make_store(json.dumps({"facts": []}))

        await store.write("session-1", {"user_message": "hi", "assistant_message": "hello"})

        assert len(conn.inserted) == 0

    async def test_no_op_when_session_id_missing(self):
        store, conn, llm, _ = make_store(json.dumps({"facts": []}))

        await store.write(None, {"user_message": "hi", "assistant_message": "hello"})

        llm.generate.assert_not_called()
        assert len(conn.inserted) == 0
```

- [ ] **Step 2: Run and verify it fails**

Run: `pytest tests/test_memory/test_fact_store_write.py -v`
Expected: FAIL — `write` not implemented.

- [ ] **Step 3: Implement**

Add to `vectra/memory/fact_store.py` (add `import uuid` to imports):

```python
EXTRACTION_PROMPT = """Extract factual (subject, predicate, object) triples from the conversation turn below. Only extract clear, stated facts about the user or entities discussed — not questions, greetings, or the assistant's own commentary. Return strict JSON only, no prose: {{"facts": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}. If there are no clear facts, return {{"facts": []}}.

User: {user}
Assistant: {assistant}"""


class FactStore:
    # ... __init__/_get_connection/ensure_indexes from Task 1 unchanged ...

    async def write(self, session_id: str, turn: Dict[str, str]):
        if not session_id or not self.llm or not self.embedder:
            return

        prompt = EXTRACTION_PROMPT.format(
            user=turn.get("user_message", ""),
            assistant=turn.get("assistant_message", ""),
        )

        try:
            raw = await self.llm.generate(prompt, "You extract structured facts as strict JSON.")
            parsed = json.loads(raw)
            facts = parsed.get("facts", []) if isinstance(parsed, dict) else []
        except Exception:
            return

        facts = [f for f in facts if f.get("subject") and f.get("predicate") and f.get("object")]
        if not facts:
            return

        texts = [f"{f['subject']} {f['predicate']} {f['object']}" for f in facts]
        try:
            embeddings = await self.embedder.embed_documents(texts)
        except Exception:
            return

        t = self.table_name
        async with self._get_connection() as conn:
            for i, f in enumerate(facts):
                vec = f"[{','.join(map(str, embeddings[i]))}]"
                fact_id = str(uuid.uuid4())
                try:
                    await conn.execute(
                        f'INSERT INTO "{t}" ("id","session_id","subject","predicate","object","embedding","source_message_id") VALUES ($1,$2,$3,$4,$5,$6,$7)',
                        fact_id, session_id, f["subject"], f["predicate"], f["object"], vec, turn.get("source_message_id"),
                    )
                except Exception:
                    pass
```

Add `import json` to the top imports alongside `uuid` if not already present.

- [ ] **Step 4: Run and verify it passes**

Run: `pytest tests/test_memory/test_fact_store_write.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Run the full suite**

Run: `pytest`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add vectra/memory/fact_store.py tests/test_memory/test_fact_store_write.py
git commit -m "feat: LLM-driven fact extraction on FactStore.write, fail-soft on malformed output"
```

---

### Task 3: Contradiction detection and invalidation

**Files:**
- Modify: `vectra/memory/fact_store.py`
- Create: `tests/test_memory/test_fact_store_invalidation.py`

**Interfaces:**
- Produces: `write()` now checks, per extracted triple, for a currently-valid fact with the same `session_id`+`subject`+`predicate` but a different `object`. If found: invalidate old (`invalid_at = NOW()`), insert new. If an identical valid fact already exists: skip (no duplicate).

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory/test_fact_store_invalidation.py`:

```python
import json
import pytest
from unittest.mock import AsyncMock
from vectra.memory.fact_store import FactStore


class FakeConn:
    def __init__(self, existing_facts=None):
        self.existing_facts = existing_facts or []
        self.inserted = []
        self.invalidated = []
        self.fetchrow = AsyncMock(side_effect=self._fetchrow)
        self.execute = AsyncMock(side_effect=self._execute)

    async def _fetchrow(self, q, *args):
        if "SELECT" in q and "invalid_at" in q and "IS NULL" in q:
            session_id, subject, predicate = args
            for f in self.existing_facts:
                if f["subject"] == subject and f["predicate"] == predicate:
                    return {"id": f["id"], "object": f["object"]}
        return None

    async def _execute(self, q, *args):
        if "UPDATE" in q and "invalid_at" in q:
            self.invalidated.append(args[0])
        elif "INSERT INTO" in q:
            self.inserted.append(args)
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeClient:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return self._conn


def make_store(existing_facts, llm_response):
    conn = FakeConn(existing_facts)
    llm = type("LLM", (), {"generate": AsyncMock(return_value=llm_response)})()
    embedder = type("Embedder", (), {"embed_documents": AsyncMock(side_effect=lambda texts: [[0.1, 0.2] for _ in texts])})()
    config = type("Cfg", (), {
        "client_instance": FakeClient(conn), "table_name": "VectraFact",
        "llm": llm, "embedder": embedder,
    })()
    return FactStore(config), conn


class TestFactStoreInvalidation:
    async def test_invalidates_old_fact_and_inserts_new_when_object_changes(self):
        store, conn = make_store(
            [{"id": "old-1", "subject": "user", "predicate": "lives_in", "object": "Berlin"}],
            json.dumps({"facts": [{"subject": "user", "predicate": "lives_in", "object": "Tokyo"}]}),
        )

        await store.write("session-1", {"user_message": "I moved to Tokyo", "assistant_message": "Nice!"})

        assert "old-1" in conn.invalidated
        assert len(conn.inserted) == 1
        assert conn.inserted[0][4] == "Tokyo"

    async def test_no_duplicate_when_identical_fact_already_valid(self):
        store, conn = make_store(
            [{"id": "old-1", "subject": "user", "predicate": "likes", "object": "coffee"}],
            json.dumps({"facts": [{"subject": "user", "predicate": "likes", "object": "coffee"}]}),
        )

        await store.write("session-1", {"user_message": "I still like coffee", "assistant_message": "Great!"})

        assert len(conn.invalidated) == 0
        assert len(conn.inserted) == 0

    async def test_inserts_as_new_when_no_existing_subject_predicate(self):
        store, conn = make_store([], json.dumps({"facts": [{"subject": "user", "predicate": "likes", "object": "tea"}]}))

        await store.write("session-1", {"user_message": "I like tea", "assistant_message": "Noted!"})

        assert len(conn.invalidated) == 0
        assert len(conn.inserted) == 1
```

- [ ] **Step 2: Run and verify it fails**

Run: `pytest tests/test_memory/test_fact_store_invalidation.py -v`
Expected: FAIL — no contradiction check yet, every fact inserted unconditionally.

- [ ] **Step 3: Implement**

Modify `write()`'s per-fact loop:

```python
        t = self.table_name
        async with self._get_connection() as conn:
            for i, f in enumerate(facts):
                vec = f"[{','.join(map(str, embeddings[i]))}]"

                existing = await conn.fetchrow(
                    f'SELECT "id","object" FROM "{t}" WHERE "session_id" = $1 AND "subject" = $2 AND "predicate" = $3 AND "invalid_at" IS NULL',
                    session_id, f["subject"], f["predicate"],
                )

                if existing and existing["object"] == f["object"]:
                    continue
                if existing:
                    try:
                        await conn.execute(f'UPDATE "{t}" SET "invalid_at" = NOW() WHERE "id" = $1', existing["id"])
                    except Exception:
                        pass

                fact_id = str(uuid.uuid4())
                try:
                    await conn.execute(
                        f'INSERT INTO "{t}" ("id","session_id","subject","predicate","object","embedding","source_message_id") VALUES ($1,$2,$3,$4,$5,$6,$7)',
                        fact_id, session_id, f["subject"], f["predicate"], f["object"], vec, turn.get("source_message_id"),
                    )
                except Exception:
                    pass
```

Note: real `asyncpg` connections return `Record` objects supporting `row["col"]` access, matching `postgres_store.py`'s existing `similarity_search` convention (`row['id']`) — the fake test double above uses plain dicts, which support the same `[...]` access pattern, so no special-casing needed in the implementation.

- [ ] **Step 4: Run and verify it passes**

Run: `pytest tests/test_memory/test_fact_store_invalidation.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Run the full suite, including Task 2's tests**

Run: `pytest tests/test_memory/ -v && pytest`
Expected: all pass — confirm Task 2's tests still pass (empty `existing_facts` list means `fetchrow` returns `None`, identical to pre-Task-3 behavior).

- [ ] **Step 6: Commit**

```bash
git add vectra/memory/fact_store.py tests/test_memory/test_fact_store_invalidation.py
git commit -m "feat: contradiction detection and bi-temporal invalidation on FactStore.write"
```

---

### Task 4: Read path — indexed vector + temporal retrieval

**Files:**
- Modify: `vectra/memory/fact_store.py`
- Create: `tests/test_memory/test_fact_store_read.py`

**Interfaces:**
- Produces: `async def read(self, session_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]` — embeds `query`, runs ONE batched query combining vector-similarity ordering with temporal-validity filtering (`invalid_at IS NULL`), scoped to `session_id`. Returns fact dicts without the `embedding` field.

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory/test_fact_store_read.py`:

```python
import pytest
from unittest.mock import AsyncMock
from vectra.memory.fact_store import FactStore


class FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.last_query = None
        self.last_args = None
        self.fetch = AsyncMock(side_effect=self._fetch)

    async def _fetch(self, q, *args):
        self.last_query = q
        self.last_args = args
        return self.rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeClient:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return self._conn


class TestFactStoreRead:
    async def test_single_query_combining_vector_and_temporal_validity(self):
        rows = [{"id": "1", "subject": "user", "predicate": "likes", "object": "coffee", "valid_at": "2026-01-01", "invalid_at": None}]
        conn = FakeConn(rows)
        embedder = type("Embedder", (), {"embed_query": AsyncMock(return_value=[0.1, 0.2])})()
        config = type("Cfg", (), {"client_instance": FakeClient(conn), "table_name": "VectraFact", "embedder": embedder})()
        store = FactStore(config)

        facts = await store.read("session-1", "what does the user like?", limit=5)

        conn.fetch.assert_called_once()
        assert '"invalid_at" IS NULL' in conn.last_query
        assert '"session_id" = $1' in conn.last_query
        assert "order by" in conn.last_query.lower()
        assert conn.last_args[0] == "session-1"

        assert len(facts) == 1
        assert facts[0]["object"] == "coffee"
        assert "embedding" not in facts[0]

    async def test_returns_empty_and_does_not_query_when_session_id_missing(self):
        conn = FakeConn([])
        embedder = type("Embedder", (), {"embed_query": AsyncMock(return_value=[0.1])})()
        config = type("Cfg", (), {"client_instance": FakeClient(conn), "table_name": "VectraFact", "embedder": embedder})()
        store = FactStore(config)

        facts = await store.read(None, "query")

        assert facts == []
        conn.fetch.assert_not_called()

    async def test_respects_limit(self):
        conn = FakeConn([])
        embedder = type("Embedder", (), {"embed_query": AsyncMock(return_value=[0.1])})()
        config = type("Cfg", (), {"client_instance": FakeClient(conn), "table_name": "VectraFact", "embedder": embedder})()
        store = FactStore(config)

        await store.read("session-1", "query", limit=3)

        assert "limit" in conn.last_query.lower()
        assert 3 in conn.last_args
```

- [ ] **Step 2: Run and verify it fails**

Run: `pytest tests/test_memory/test_fact_store_read.py -v`
Expected: FAIL — `read` not implemented.

- [ ] **Step 3: Implement**

Add to `vectra/memory/fact_store.py`:

```python
    async def read(self, session_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not session_id or not self.embedder:
            return []
        vector = await self.embedder.embed_query(query)
        vec = f"[{','.join(map(str, vector))}]"
        t = self.table_name

        async with self._get_connection() as conn:
            rows = await conn.fetch(
                f'''SELECT "id","subject","predicate","object","valid_at","invalid_at"
                    FROM "{t}"
                    WHERE "session_id" = $1 AND "invalid_at" IS NULL
                    ORDER BY "embedding" <=> $2
                    LIMIT $3''',
                session_id, vec, max(1, limit),
            )
        return [dict(r) for r in rows]
```

Note: `dict(r)` converts asyncpg `Record` objects to plain dicts, matching the test double's plain-dict rows and giving callers a stable, embedding-free shape.

- [ ] **Step 4: Run and verify it passes**

Run: `pytest tests/test_memory/test_fact_store_read.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Run the full suite**

Run: `pytest`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add vectra/memory/fact_store.py tests/test_memory/test_fact_store_read.py
git commit -m "feat: indexed vector+temporal read path for FactStore, single batched query"
```

---

### Task 5: Config wiring — `memory.facts`, VectraClient instantiation

**Files:**
- Modify: `vectra/config.py`
- Modify: `vectra/core.py`
- Create: `tests/test_config_memory_facts.py`

**Interfaces:**
- Produces: `VectraClient` instantiates `self.fact_store = FactStore(...)` when `config.memory.get('facts', {}).get('enabled')` is true (Python's `memory` config is an untyped `Dict[str, Any]`, per existing convention in `vectra/config.py` — confirmed via research, matches how `query_planning` already works; do NOT introduce a typed Pydantic sub-model here, stay consistent with the existing untyped-dict pattern for this field).

- [ ] **Step 1: Read the current memory config and VectraClient constructor**

Read `vectra/config.py`'s `memory` field (around line 135) and `vectra/core.py`'s constructor (around lines 90-126, where `self.embedder`/`self.llm`/`self.history` are built) in full before editing.

- [ ] **Step 2: Write the failing test**

Create `tests/test_config_memory_facts.py`:

```python
from vectra.core import VectraClient
from vectra.config import VectraConfig, EmbeddingConfig, LLMConfig, DatabaseConfig


def make_config(memory=None):
    return VectraConfig(
        embedding=EmbeddingConfig(provider="openai", model_name="text-embedding-3-small"),
        llm=LLMConfig(provider="openai", model_name="gpt-4o-mini"),
        database=DatabaseConfig(type="postgres", client_instance=object()),
        memory=memory,
    )


class TestMemoryFactsConfig:
    def test_fact_store_not_created_when_facts_not_enabled(self):
        client = VectraClient(make_config(memory={"enabled": False}))
        assert getattr(client, "fact_store", None) is None

    def test_fact_store_created_when_facts_enabled(self):
        client = VectraClient(make_config(memory={"enabled": True, "facts": {"enabled": True, "client_instance": object(), "table_name": "MyFacts"}}))
        assert client.fact_store is not None
        assert client.fact_store.table_name == "MyFacts"
```

- [ ] **Step 3: Run and verify it fails**

Run: `pytest tests/test_config_memory_facts.py -v`
Expected: FAIL — `VectraClient` has no `fact_store` attribute yet.

- [ ] **Step 4: Wire instantiation in core.py**

In `vectra/core.py`'s constructor, after `self.embedder`/`self.llm` are constructed (confirmed line ~90-91 from Step 1's read) and after the existing `self.history` block, add:

```python
        facts_cfg = (config.memory or {}).get('facts') if config.memory else None
        if facts_cfg and facts_cfg.get('enabled'):
            from .memory.fact_store import FactStore
            fact_config = type("FactCfg", (), {
                'client_instance': facts_cfg.get('client_instance'),
                'table_name': facts_cfg.get('table_name', 'VectraFact'),
                'llm': self.llm,
                'embedder': self.embedder,
            })()
            self.fact_store = FactStore(fact_config)
        else:
            self.fact_store = None
```

No `config.py` schema change is needed for this task specifically (the `memory` field is already an untyped `Optional[Dict[str, Any]]`, so a `facts` key is accepted without a schema edit) — confirm this by re-reading `config.py`'s `memory` field definition before skipping the "modify config.py" file listed above; if research turns up any validation that WOULD reject an unknown `facts` key (e.g. a stricter model elsewhere), address that specifically rather than assuming.

- [ ] **Step 5: Run and verify it passes**

Run: `pytest tests/test_config_memory_facts.py -v`
Expected: 2 tests pass.

- [ ] **Step 6: Run the full suite**

Run: `pytest`
Expected: all pass, no regressions to existing `VectraClient` construction tests.

- [ ] **Step 7: Commit**

```bash
git add vectra/core.py tests/test_config_memory_facts.py
git commit -m "feat: wire FactStore into VectraClient via memory.facts config"
```
(Omit `vectra/config.py` from the commit if Step 4's research confirms no schema change was actually needed — don't stage a file with no diff.)

---

### Task 6: Short-lived per-session read cache

**Files:**
- Modify: `vectra/memory/fact_store.py`
- Create: `tests/test_memory/test_fact_store_cache.py`

**Interfaces:**
- Produces: `read()` checks an in-process dict-based cache keyed by `f"{session_id}:{query}"` before querying, with TTL-based expiry (default 30s, via `config.cache_ttl_ms` — note: Python uses seconds internally is fine, but keep the config field name `cache_ttl_ms` for naming consistency with the JS sibling's `cacheTtlMs`, converting to seconds internally in this file).

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory/test_fact_store_cache.py`:

```python
import asyncio
import pytest
from unittest.mock import AsyncMock
from vectra.memory.fact_store import FactStore


class FakeConn:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.fetch = AsyncMock(return_value=self.rows)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeClient:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return self._conn


def make_store(cache_ttl_ms=30000):
    conn = FakeConn([{"id": "1", "subject": "a", "predicate": "b", "object": "c"}])
    embedder = type("Embedder", (), {"embed_query": AsyncMock(return_value=[0.1])})()
    config = type("Cfg", (), {
        "client_instance": FakeClient(conn), "table_name": "VectraFact",
        "embedder": embedder, "cache_ttl_ms": cache_ttl_ms,
    })()
    return FactStore(config), conn


class TestFactStoreCache:
    async def test_does_not_requery_within_ttl(self):
        store, conn = make_store(cache_ttl_ms=30000)

        await store.read("session-1", "same query")
        await store.read("session-1", "same query")

        conn.fetch.assert_called_once()

    async def test_requeries_for_different_query_text(self):
        store, conn = make_store(cache_ttl_ms=30000)

        await store.read("session-1", "query A")
        await store.read("session-1", "query B")

        assert conn.fetch.call_count == 2

    async def test_requeries_after_expiry(self):
        store, conn = make_store(cache_ttl_ms=1)

        await store.read("session-1", "query")
        await asyncio.sleep(0.01)
        await store.read("session-1", "query")

        assert conn.fetch.call_count == 2
```

- [ ] **Step 2: Run and verify it fails**

Run: `pytest tests/test_memory/test_fact_store_cache.py -v`
Expected: FAIL — no caching exists yet.

- [ ] **Step 3: Implement**

In `__init__`, add: `self.cache_ttl_ms = getattr(config, 'cache_ttl_ms', 30000); self._read_cache: Dict[str, Any] = {}`. Add `import time` to the top imports. At the top of `read()`:

```python
        cache_key = f"{session_id}:{query}"
        cached = self._read_cache.get(cache_key)
        if cached and (time.time() * 1000 - cached['ts']) < self.cache_ttl_ms:
            return cached['value']
```

Before the final `return [dict(r) for r in rows]`, replace with:

```python
        result = [dict(r) for r in rows]
        self._read_cache[cache_key] = {'ts': time.time() * 1000, 'value': result}
        return result
```

- [ ] **Step 4: Run and verify it passes**

Run: `pytest tests/test_memory/test_fact_store_cache.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Run the full suite, including all prior FactStore tests**

Run: `pytest tests/test_memory/ -v && pytest`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add vectra/memory/fact_store.py tests/test_memory/test_fact_store_cache.py
git commit -m "perf: add short-lived per-session read cache to FactStore"
```

---

### Task 7: End-to-end integration test

**Files:**
- Create: `tests/test_memory/test_fact_store_integration.py`

**Interfaces:**
- Consumes: everything from Tasks 1-6. No new production code.

- [ ] **Step 1: Write the integration test**

Create `tests/test_memory/test_fact_store_integration.py`:

```python
import json
import uuid
from unittest.mock import AsyncMock
from vectra.memory.fact_store import FactStore


class InMemoryFakeConn:
    """Actually stores/updates/filters rows, unlike the narrower per-task mocks,
    to catch integration bugs the unit tests' shape-specific mocks could miss."""
    def __init__(self):
        self.rows = []

    async def execute(self, q, *args):
        if "UPDATE" in q and "invalid_at" in q:
            fact_id = args[0]
            for r in self.rows:
                if r["id"] == fact_id:
                    r["invalid_at"] = "now"
        elif "INSERT INTO" in q:
            fact_id, session_id, subject, predicate, obj = args[0], args[1], args[2], args[3], args[4]
            self.rows.append({"id": fact_id, "session_id": session_id, "subject": subject, "predicate": predicate, "object": obj, "valid_at": "then", "invalid_at": None})
        return None

    async def fetchrow(self, q, *args):
        session_id, subject, predicate = args
        for r in self.rows:
            if r["session_id"] == session_id and r["subject"] == subject and r["predicate"] == predicate and not r["invalid_at"]:
                return {"id": r["id"], "object": r["object"]}
        return None

    async def fetch(self, q, *args):
        session_id = args[0]
        return [
            {"id": r["id"], "subject": r["subject"], "predicate": r["predicate"], "object": r["object"], "valid_at": r["valid_at"], "invalid_at": r["invalid_at"]}
            for r in self.rows if r["session_id"] == session_id and not r["invalid_at"]
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeClient:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return self._conn


def make_store(conn, llm_responses):
    responses = iter(llm_responses)
    llm = type("LLM", (), {"generate": AsyncMock(side_effect=lambda *a, **k: next(responses))})()
    embedder = type("Embedder", (), {
        "embed_documents": AsyncMock(side_effect=lambda texts: [[0.1] for _ in texts]),
        "embed_query": AsyncMock(return_value=[0.1]),
    })()
    config = type("Cfg", (), {"client_instance": FakeClient(conn), "table_name": "VectraFact", "llm": llm, "embedder": embedder})()
    return FactStore(config)


class TestFactStoreEndToEnd:
    async def test_write_then_read_round_trips_a_fact(self):
        conn = InMemoryFakeConn()
        store = make_store(conn, [json.dumps({"facts": [{"subject": "user", "predicate": "likes", "object": "coffee"}]})])

        await store.write("session-1", {"user_message": "I like coffee", "assistant_message": "Noted!"})
        facts = await store.read("session-1", "what does the user like")

        assert len(facts) == 1
        assert facts[0]["object"] == "coffee"

    async def test_superseding_fact_replaces_old_one_never_both(self):
        conn = InMemoryFakeConn()
        store = make_store(conn, [
            json.dumps({"facts": [{"subject": "user", "predicate": "lives_in", "object": "Berlin"}]}),
            json.dumps({"facts": [{"subject": "user", "predicate": "lives_in", "object": "Tokyo"}]}),
        ])

        await store.write("session-1", {"user_message": "I live in Berlin", "assistant_message": "Cool!"})
        await store.write("session-1", {"user_message": "I moved to Tokyo", "assistant_message": "Wow!"})
        facts = await store.read("session-1", "where does the user live")

        lives_in = [f for f in facts if f["predicate"] == "lives_in"]
        assert len(lives_in) == 1
        assert lives_in[0]["object"] == "Tokyo"

    async def test_facts_never_leak_across_sessions(self):
        conn = InMemoryFakeConn()
        store = make_store(conn, [json.dumps({"facts": [{"subject": "user", "predicate": "likes", "object": "coffee"}]})])

        await store.write("session-A", {"user_message": "I like coffee", "assistant_message": "Noted!"})
        facts_b = await store.read("session-B", "what does the user like")

        assert len(facts_b) == 0
```

- [ ] **Step 2: Run and verify it passes**

Run: `pytest tests/test_memory/test_fact_store_integration.py -v`
Expected: 3 tests pass. If any fail, it reveals an integration bug the narrower unit tests missed — fix the underlying implementation, do not weaken these assertions.

- [ ] **Step 3: Run the full suite**

Run: `pytest`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_memory/test_fact_store_integration.py
git commit -m "test: add end-to-end integration coverage for FactStore write/invalidate/read"
```

---

## Self-Review Notes

- **Spec coverage:** mirrors vectra-js's Phase 1 plan exactly — schema (Task 1), extraction (Task 2), invalidation (Task 3), read (Task 4), config wiring (Task 5), caching (Task 6), integration (Task 7). Multi-hop graph walk deferred to a later phase, same as JS.
- **Placeholder scan:** no TBD/TODO.
- **Package-conversion risk flagged explicitly**: Task 1 converts `vectra/memory.py` from a module to a package (`vectra/memory/__init__.py`) to make room for the new `fact_store.py` sibling module — this is the one structurally risky step in this plan (existing import call sites must keep working), so Task 1 explicitly requires running the full suite immediately after the conversion, before any FactStore code is written, to catch breakage at the cheapest possible point.
- **Type/interface consistency:** `FactStore.__init__`, `write(session_id, turn)`, `read(session_id, query, limit)` signatures match the JS sibling's shape 1:1 (adjusted for snake_case), verified no drift across Tasks 2-7.
- **Naming parity with JS**: `cache_ttl_ms` (not `cache_ttl_seconds`) deliberately chosen to match the JS sibling's `cacheTtlMs` field name across the language boundary, even though Python could idiomatically use seconds — consistency with the dual-language parity goal from the spec outweighs local idiom here.
