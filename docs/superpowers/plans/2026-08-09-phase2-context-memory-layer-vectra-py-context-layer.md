# Context & Memory Layer Phase 2 — vectra-py Context Layer Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mirror vectra-js's Phase 2 in Python — `build_context()` (Tier 1) and `context.ask()` (Tier 2), then a surgical `query_rag` fix reserving a budgeted slice for history via `build_context`, fixing the same unbounded history/token-budget bug. Same architecture, same task shape as the JS sibling (`vectra-js/docs/superpowers/plans/2026-08-09-phase2-context-memory-layer-vectra-js-context-layer.md`), already reviewed and merged there — this plan bakes in every fix found during that review from the start, rather than rediscovering them:

- Token cache is bounded/LRU-evicted from Task 1 (not an unbounded `dict`).
- `context.ask()` calls `check_guardrails` and `_run_middlewares('on_before_retrieve', ...)` from Task 6 (an automated security review of the JS sibling found these were silently bypassable — fix it here before it ships, not after).
- `query_rag`'s refactor (Task 7) reserves a budget slice for history via `build_context` and passes the REMAINING budget into the existing, unmodified `_build_context_parts` — deliberately NOT routing docs through `build_context`'s own generic packing, to avoid reconstructing `doc_map` and risking citation-index misalignment (the JS sibling's implementer made this same judgment call after finding the literal original plan text riskier than necessary).

**Architecture:** New `vectra/context_layer.py` module, exporting `async def build_context(input)`. Single vector store only in this phase.

**Tech Stack:** Reuses `_get_token_encoder()`/`_token_estimate` already in `vectra/core.py` (lazy tiktoken singleton with offline fallback). pytest, `pytest-asyncio` auto mode.

## Global Constraints

- Single vector store only in this phase — multi-db fan-out is Phase 3.
- Tool-results accepted as pre-computed input, never executed.
- Token counts cached, bounded (LRU, matching JS's `TOKEN_CACHE_MAX_SIZE = 10000` for consistency across the two SDKs).
- `dropped` in the output must be explicit and honest.
- `context.ask()` must call `check_guardrails` and `_run_middlewares('on_before_retrieve', ...)` — same as `query_rag` — from its first commit in this plan, not as a follow-up fix.
- The `query_rag` refactor task must not change its public signature or break any existing test. Run the FULL suite after that task.
- No direct-to-master commits, no force-push, no skipped hooks.
- Every task ends with `pytest` passing before moving to the next task.

---

### Task 1: Cached, bounded token counting helper

**Files:**
- Create: `vectra/context_layer.py`
- Create: `tests/test_context_layer_token_cache.py`

**Interfaces:**
- Produces: `estimate_tokens_cached(text)` — same token count as `VectraClient._token_estimate`, memoized by exact string content, LRU-bounded at 10000 entries.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_context_layer_token_cache.py`:

```python
from unittest.mock import patch
from vectra.context_layer import estimate_tokens_cached, _clear_token_cache, _encode_for_test


class TestEstimateTokensCached:
    def setup_method(self):
        _clear_token_cache()

    def test_returns_a_real_token_count(self):
        assert estimate_tokens_cached("Hello, world!") == 4

    def test_does_not_retokenize_identical_content_on_a_second_call(self):
        with patch("vectra.context_layer._encode_for_test", wraps=_encode_for_test) as spy:
            estimate_tokens_cached("the quick brown fox")
            estimate_tokens_cached("the quick brown fox")
            calls_for_this_string = sum(1 for c in spy.call_args_list if c.args[0] == "the quick brown fox")
            assert calls_for_this_string <= 1

    def test_returns_zero_for_empty_or_falsy_input(self):
        assert estimate_tokens_cached("") == 0
        assert estimate_tokens_cached(None) == 0

    def test_evicts_the_oldest_entry_once_the_cache_exceeds_its_bound(self):
        with patch("vectra.context_layer._encode_for_test", wraps=_encode_for_test) as spy:
            for i in range(10001):
                estimate_tokens_cached(f"unique string number {i}")
            spy.reset_mock()
            estimate_tokens_cached("unique string number 0")
            assert any(c.args[0] == "unique string number 0" for c in spy.call_args_list)
```

IMPORTANT: `patch(..., wraps=_encode_for_test)` patches the module-level `_encode_for_test` NAME, which Python's attribute-lookup semantics mean IS observed by internal calls, unlike JS's `jest.spyOn` on a CommonJS export (the JS sibling hit a real gap here — a closure-local call bypassed the spy entirely, silently making its first version of this exact test vacuous). Confirm this Python pattern genuinely works by deliberately breaking the cache (comment out the cache-hit branch temporarily during Step 4's verification) and confirming test 2 actually fails — do not skip this check.

- [ ] **Step 2: Run and verify they fail**

Run: `pytest tests/test_context_layer_token_cache.py -v`
Expected: FAIL — `vectra/context_layer.py` doesn't exist yet.

- [ ] **Step 3: Implement**

Create `vectra/context_layer.py`:

```python
from typing import Any, Dict, List, Optional
import tiktoken

_token_encoder = None


def _get_token_encoder():
    global _token_encoder
    if _token_encoder is None:
        try:
            _token_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None
    return _token_encoder


def _encode_for_test(text: str):
    encoder = _get_token_encoder()
    if encoder is None:
        return []
    return encoder.encode(str(text))


_TOKEN_CACHE_MAX_SIZE = 10000
_token_cache: Dict[str, int] = {}


def estimate_tokens_cached(text: Optional[str]) -> int:
    if not text:
        return 0
    key = str(text)
    if key in _token_cache:
        count = _token_cache.pop(key)
        _token_cache[key] = count  # refresh recency (dict preserves insertion order in Python 3.7+)
        return count
    encoded = _encode_for_test(key)
    if encoded:
        count = len(encoded)
    else:
        # Offline fallback, same heuristic as VectraClient._token_estimate's fallback.
        ascii_chars = sum(1 for c in key if ord(c) < 128)
        non_ascii = len(key) - ascii_chars
        count = max(1, (ascii_chars + 3) // 4 + non_ascii)
    _token_cache[key] = count
    if len(_token_cache) > _TOKEN_CACHE_MAX_SIZE:
        oldest_key = next(iter(_token_cache))
        del _token_cache[oldest_key]
    return count


def _clear_token_cache():
    _token_cache.clear()
```

- [ ] **Step 4: Run and verify all pass**

Run: `pytest tests/test_context_layer_token_cache.py -v`
Expected: 4 tests pass. Also perform the deliberate-break verification described in Step 1's note before moving on.

- [ ] **Step 5: Run the full suite**

Run: `pytest`
Expected: all pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add vectra/context_layer.py tests/test_context_layer_token_cache.py
git commit -m "feat: add cached, bounded token-count helper for the new context layer"
```

---

### Task 2: `build_context` — docs source, budget trimming, honest `dropped`

**Files:**
- Modify: `vectra/context_layer.py`
- Create: `tests/test_context_layer_build_context_docs.py`

**Interfaces:**
- Produces: `async def build_context(input: Dict) -> Dict` where `input = {"query": ..., "budget": {"max_tokens": ...}, "sources": [{"type": "docs", "items": [{"content": ..., "metadata": {...}}, ...]}]}`. Returns `{"parts": [...], "text": ..., "tokens_used": ..., "tokens_budget": ..., "dropped": [...], "warnings": []}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_context_layer_build_context_docs.py`:

```python
from vectra.context_layer import build_context, _clear_token_cache


class TestBuildContextDocs:
    def setup_method(self):
        _clear_token_cache()

    async def test_packs_doc_content_into_parts_within_budget(self):
        result = await build_context({
            "query": "what is vectra?",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "items": [
                {"content": "Vectra is a RAG orchestration SDK.", "metadata": {"source": "readme.md"}},
            ]}],
        })

        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "docs"
        assert "Vectra is a RAG orchestration SDK." in result["parts"][0]["content"]
        assert result["parts"][0]["tokens"] > 0
        assert "Vectra is a RAG orchestration SDK." in result["text"]
        assert result["tokens_used"] > 0
        assert result["tokens_budget"] == 1000
        assert result["dropped"] == []

    async def test_honestly_reports_dropped_items_when_budget_exceeded(self):
        long_content = "word " * 200
        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 10},
            "sources": [{"type": "docs", "items": [
                {"content": long_content, "metadata": {"source": "a.md"}},
                {"content": "short", "metadata": {"source": "b.md"}},
            ]}],
        })

        assert len(result["dropped"]) > 0
        assert result["dropped"][0]["source"] == "docs"
        assert len(result["parts"]) + len(result["dropped"]) == 2

    async def test_returns_empty_result_for_no_sources(self):
        result = await build_context({"query": "q", "budget": {"max_tokens": 100}, "sources": []})
        assert result["parts"] == []
        assert result["text"] == ""
        assert result["tokens_used"] == 0
```

- [ ] **Step 2: Run and verify they fail**

Run: `pytest tests/test_context_layer_build_context_docs.py -v`
Expected: FAIL — `build_context` not defined yet.

- [ ] **Step 3: Implement**

Add to `vectra/context_layer.py`:

```python
async def build_context(input: Dict[str, Any]) -> Dict[str, Any]:
    query = input.get("query")
    budget = input.get("budget") or {}
    sources = input.get("sources") or []
    priority = input.get("priority")

    max_tokens = budget.get("max_tokens", 2048)
    parts: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    used = 0

    if priority:
        def rank(source):
            t = source.get("type")
            return priority.index(t) if t in priority else len(priority)
        ordered_sources = sorted(sources, key=rank)
    else:
        ordered_sources = sources

    for source in ordered_sources:
        if source.get("type") == "docs":
            for item in source.get("items", []):
                content = item.get("content", "")
                tokens = estimate_tokens_cached(content)
                if used + tokens > max_tokens:
                    dropped.append({"source": "docs", "metadata": item.get("metadata", {})})
                    continue
                parts.append({"source": "docs", "type": "docs", "content": content, "tokens": tokens})
                used += tokens

    return {
        "parts": parts,
        "text": "\n---\n".join(p["content"] for p in parts),
        "tokens_used": used,
        "tokens_budget": max_tokens,
        "dropped": dropped,
        "warnings": [],
    }
```

- [ ] **Step 4: Run and verify all pass**

Run: `pytest tests/test_context_layer_build_context_docs.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Run the full suite**

Run: `pytest`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add vectra/context_layer.py tests/test_context_layer_build_context_docs.py
git commit -m "feat: build_context docs source with budget-aware packing and honest dropped reporting"
```

---

### Task 3: `build_context` — memory source (FactStore integration)

**Files:**
- Modify: `vectra/context_layer.py`
- Create: `tests/test_context_layer_build_context_memory.py`

**Interfaces:**
- Consumes: `FactStore.read(session_id, query, limit=10)` from Phase 1.
- Produces: `sources` accepts `{"type": "memory", "fact_store": ..., "session_id": ...}`. Packs each returned fact, counted against the same budget.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_context_layer_build_context_memory.py`:

```python
from unittest.mock import AsyncMock
from vectra.context_layer import build_context, _clear_token_cache


def fake_fact_store(facts):
    store = type("FS", (), {})()
    store.read = AsyncMock(return_value=facts)
    return store


class TestBuildContextMemory:
    def setup_method(self):
        _clear_token_cache()

    async def test_packs_facts_from_fact_store_read_counted_against_budget(self):
        fact_store = fake_fact_store([{"subject": "user", "predicate": "likes", "object": "coffee"}])

        result = await build_context({
            "query": "what does the user like",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "memory", "fact_store": fact_store, "session_id": "session-1"}],
        })

        fact_store.read.assert_called_once_with("session-1", "what does the user like")
        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "memory"
        assert "coffee" in result["parts"][0]["content"]
        assert result["tokens_used"] > 0

    async def test_memory_competes_for_budget_with_docs_real_regression_test(self):
        long_doc = "word " * 50
        fact_store = fake_fact_store([{"subject": "user", "predicate": "likes", "object": "coffee"}])

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 5},
            "sources": [
                {"type": "docs", "items": [{"content": long_doc, "metadata": {}}]},
                {"type": "memory", "fact_store": fact_store, "session_id": "session-1"},
            ],
        })

        assert any(d["source"] == "docs" for d in result["dropped"])

    async def test_skips_memory_cleanly_when_no_fact_store_or_session_id(self):
        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 100},
            "sources": [{"type": "memory", "fact_store": None, "session_id": "session-1"}],
        })
        assert result["parts"] == []
```

- [ ] **Step 2: Run and verify they fail**

Run: `pytest tests/test_context_layer_build_context_memory.py -v`
Expected: FAIL — memory source type not handled yet.

- [ ] **Step 3: Implement**

Add a `memory` branch to `build_context`'s loop:

```python
        if source.get("type") == "memory":
            fact_store = source.get("fact_store")
            session_id = source.get("session_id")
            if not fact_store or not session_id:
                continue
            facts = await fact_store.read(session_id, query) or []
            for fact in facts:
                content = f"{fact['subject']} {fact['predicate']} {fact['object']}"
                tokens = estimate_tokens_cached(content)
                if used + tokens > max_tokens:
                    dropped.append({"source": "memory", "fact": fact})
                    continue
                parts.append({"source": "memory", "type": "memory", "content": content, "tokens": tokens})
                used += tokens
```

- [ ] **Step 4: Run and verify all pass**

Run: `pytest tests/test_context_layer_build_context_memory.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Run the full suite, including Task 2's tests**

Run: `pytest tests/test_context_layer_build_context_docs.py tests/test_context_layer_build_context_memory.py && pytest`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add vectra/context_layer.py tests/test_context_layer_build_context_memory.py
git commit -m "feat: build_context memory source via FactStore.read, counted against the shared token budget"
```

---

### Task 4: `build_context` — tools source (pre-computed results only)

**Files:**
- Modify: `vectra/context_layer.py`
- Create: `tests/test_context_layer_build_context_tools.py`

**Interfaces:**
- Produces: `sources` accepts `{"type": "tools", "results": [{"name": ..., "output": ...}, ...]}`. No execution.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_context_layer_build_context_tools.py`:

```python
import json
from vectra.context_layer import build_context, _clear_token_cache


class TestBuildContextTools:
    def setup_method(self):
        _clear_token_cache()

    async def test_packs_precomputed_tool_results_into_parts(self):
        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "tools", "results": [{"name": "get_weather", "output": "Sunny, 22C"}]}],
        })

        assert len(result["parts"]) == 1
        assert result["parts"][0]["type"] == "tools"
        assert "get_weather" in result["parts"][0]["content"]
        assert "Sunny, 22C" in result["parts"][0]["content"]

    async def test_does_not_execute_anything_results_used_verbatim(self):
        output = json.dumps({"note": "this is data, not a function"})
        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "tools", "results": [{"name": "x", "output": output}]}],
        })
        assert "this is data, not a function" in result["parts"][0]["content"]

    async def test_honestly_drops_tool_results_that_dont_fit(self):
        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 2},
            "sources": [{"type": "tools", "results": [{"name": "x", "output": "word " * 50}]}],
        })
        assert result["parts"] == []
        assert len(result["dropped"]) == 1
        assert result["dropped"][0]["source"] == "tools"
```

- [ ] **Step 2: Run and verify they fail**

Run: `pytest tests/test_context_layer_build_context_tools.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add a `tools` branch:

```python
        if source.get("type") == "tools":
            for result_item in source.get("results", []):
                content = f"Tool: {result_item['name']}\nResult: {result_item['output']}"
                tokens = estimate_tokens_cached(content)
                if used + tokens > max_tokens:
                    dropped.append({"source": "tools", "name": result_item["name"]})
                    continue
                parts.append({"source": "tools", "type": "tools", "content": content, "tokens": tokens})
                used += tokens
```

- [ ] **Step 4: Run and verify all pass**

Run: `pytest tests/test_context_layer_build_context_tools.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Run the full suite**

Run: `pytest`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add vectra/context_layer.py tests/test_context_layer_build_context_tools.py
git commit -m "feat: build_context tools source, pre-computed results only, no execution"
```

---

### Task 5: Priority-based trim order

**Files:**
- Modify: `vectra/context_layer.py` (verify — the sort implemented in Task 2 already ranks unlisted types to `len(priority)`, i.e. last, which is the CORRECT behavior from the start, unlike JS's first attempt which used a naive `indexOf` difference that sorted unlisted types first — a real bug the JS sibling found and fixed. Confirm this Python version is already correct by writing the discriminating test below; if it somehow isn't, fix it.)
- Create: `tests/test_context_layer_priority.py`

**Interfaces:**
- Verifies: `priority` list determines trim order across source types; unlisted types fall to the end, not silently dropped and not given undeserved first-claim priority.

- [ ] **Step 1: Write the tests**

Create `tests/test_context_layer_priority.py`:

```python
from unittest.mock import AsyncMock
from vectra.context_layer import build_context, _clear_token_cache


def fake_fact_store(facts):
    store = type("FS", (), {})()
    store.read = AsyncMock(return_value=facts)
    return store


class TestBuildContextPriority:
    def setup_method(self):
        _clear_token_cache()

    async def test_gives_earlier_priority_sources_first_claim_on_tight_budget(self):
        fact_store = fake_fact_store([{"subject": "user", "predicate": "likes", "object": "coffee"}])
        big_doc = "word " * 50

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 10},
            "sources": [
                {"type": "docs", "items": [{"content": big_doc, "metadata": {}}]},
                {"type": "memory", "fact_store": fact_store, "session_id": "s1"},
            ],
            "priority": ["memory", "docs"],
        })

        assert any(p["type"] == "memory" for p in result["parts"])
        assert not any(p["type"] == "docs" for p in result["parts"])
        assert any(d["source"] == "docs" for d in result["dropped"])

    async def test_unlisted_source_type_falls_after_listed_ones_not_before(self):
        fact_store = fake_fact_store([{"subject": "user", "predicate": "likes", "object": "coffee"}])

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [
                {"type": "tools", "results": [{"name": "x", "output": "y"}]},  # unlisted
                {"type": "memory", "fact_store": fact_store, "session_id": "s1"},  # listed, should win ordering
            ],
            "priority": ["memory", "docs"],
        })

        assert result["parts"][0]["type"] == "memory"
        assert result["parts"][1]["type"] == "tools"

    async def test_no_priority_processes_in_supplied_order(self):
        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [
                {"type": "tools", "results": [{"name": "x", "output": "y"}]},
                {"type": "docs", "items": [{"content": "doc content", "metadata": {}}]},
            ],
        })
        assert result["parts"][0]["type"] == "tools"
        assert result["parts"][1]["type"] == "docs"
```

- [ ] **Step 2: Run and verify they pass**

Run: `pytest tests/test_context_layer_priority.py -v`
Expected: all 3 pass, since Task 2's `rank()` helper already uses `priority.index(t) if t in priority else len(priority)` — correctly ranking unlisted types last. If the second test unexpectedly fails, it means the implementation drifted from Task 2's code — fix the `rank()` function to match, do not weaken the test.

- [ ] **Step 3: Run the full suite**

Run: `pytest`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_context_layer_priority.py
git commit -m "test: add dedicated priority-order coverage for build_context (already correct from Task 2, unlike the JS sibling's first attempt)"
```

---

### Task 6: Tier 2 — `context.ask()` on `VectraClient`, with guardrails/middleware enforcement from the start

**Files:**
- Modify: `vectra/core.py`
- Create: `tests/test_core_context_ask.py`

**Interfaces:**
- Consumes: `build_context` from `vectra/context_layer.py`, `self.vector_store.similarity_search`, `self.embedder.embed_query`, `self.fact_store` (Phase 1, may be `None`), `check_guardrails`, `self._run_middlewares`.
- Produces: `VectraClient` gains `self.context` — an object/namespace with an `ask(query, session_id=None, tools=None)` async method. MUST call `check_guardrails(query, self.config.guardrails)` and `await self._run_middlewares('on_before_retrieve', query, query_vector)` — same as `query_rag` — BEFORE embedding/retrieval. This requirement is in this task from the start (an automated security review of the JS sibling found these were bypassable when omitted).

- [ ] **Step 1: Read `core.py`'s constructor and `query_rag`'s guardrails/middleware call pattern**

Confirm exact current property names (`self.vector_store`, `self.embedder`, `self.fact_store` from Phase 1, `self.config`, `self._run_middlewares`) and the exact `check_guardrails`/`_run_middlewares` call shape at the top of `query_rag` before wiring.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_core_context_ask.py`:

```python
import pytest
from unittest.mock import AsyncMock
from vectra.core import VectraClient
from vectra.config import VectraConfig, EmbeddingConfig, LLMConfig, DatabaseConfig


def make_config(memory=None, guardrails=None):
    return VectraConfig(
        embedding=EmbeddingConfig(provider="openai", api_key="test-key", model_name="text-embedding-3-small"),
        llm=LLMConfig(provider="openai", api_key="test-key", model_name="gpt-4o-mini"),
        database=DatabaseConfig(type="postgres", client_instance=object()),
        memory=memory,
        guardrails=guardrails,
    )


class TestContextAsk:
    async def test_embeds_query_retrieves_docs_returns_packed_context(self):
        client = VectraClient(make_config())
        client.embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
        client.vector_store.similarity_search = AsyncMock(return_value=[
            {"content": "Vectra is a RAG SDK.", "metadata": {"source": "readme.md"}},
        ])

        result = await client.context.ask("what is vectra?")

        client.embedder.embed_query.assert_called_with("what is vectra?")
        client.vector_store.similarity_search.assert_called()
        assert any(p["type"] == "docs" for p in result["parts"])
        assert "Vectra is a RAG SDK." in result["text"]

    async def test_includes_memory_when_session_id_given_and_fact_store_configured(self):
        client = VectraClient(make_config(memory={"enabled": True, "facts": {"enabled": True, "client_instance": object(), "table_name": "F"}}))
        client.embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
        client.vector_store.similarity_search = AsyncMock(return_value=[])
        client.fact_store.read = AsyncMock(return_value=[{"subject": "user", "predicate": "likes", "object": "coffee"}])

        result = await client.context.ask("what does the user like", session_id="session-1")

        client.fact_store.read.assert_called_with("session-1", "what does the user like")
        assert any(p["type"] == "memory" for p in result["parts"])

    async def test_skips_memory_cleanly_when_no_session_id(self):
        client = VectraClient(make_config(memory={"enabled": True, "facts": {"enabled": True, "client_instance": object(), "table_name": "F"}}))
        client.embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
        client.vector_store.similarity_search = AsyncMock(return_value=[])
        client.fact_store.read = AsyncMock()

        await client.context.ask("q")

        client.fact_store.read.assert_not_called()

    async def test_includes_tool_results_when_given(self):
        client = VectraClient(make_config())
        client.embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
        client.vector_store.similarity_search = AsyncMock(return_value=[])

        result = await client.context.ask("q", tools=[{"name": "get_weather", "output": "Sunny"}])

        assert any(p["type"] == "tools" for p in result["parts"])

    async def test_enforces_guardrails_before_any_embedding_call(self):
        client = VectraClient(make_config(guardrails={"max_query_length": 10}))
        client.embedder.embed_query = AsyncMock()

        with pytest.raises(Exception, match="GuardrailViolation"):
            await client.context.ask("this query is way too long for the configured limit")
        client.embedder.embed_query.assert_not_called()

    async def test_runs_on_before_retrieve_middleware(self):
        client = VectraClient(make_config())
        client.embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
        client.vector_store.similarity_search = AsyncMock(return_value=[])
        client._run_middlewares = AsyncMock(side_effect=lambda name, *args: args)

        await client.context.ask("q")

        client._run_middlewares.assert_any_call("on_before_retrieve", "q", [0.1, 0.2])
```

- [ ] **Step 3: Run and verify they fail**

Run: `pytest tests/test_core_context_ask.py -v`
Expected: FAIL — `client.context` doesn't exist yet.

- [ ] **Step 4: Implement**

Add `from .context_layer import build_context` to `core.py`'s imports. In `VectraClient.__init__`, after `self.fact_store` is set (Phase 1's wiring), add:

```python
        async def _context_ask(query: str, session_id: Optional[str] = None, tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
            check_guardrails(query, self.config.guardrails)
            query_vector = await self.embedder.embed_query(query)
            query, query_vector = await self._run_middlewares('on_before_retrieve', query, query_vector)
            docs = await self.vector_store.similarity_search(query_vector, 5)
            sources = [{"type": "docs", "items": [{"content": d["content"], "metadata": d.get("metadata", {})} for d in docs]}]
            if session_id and self.fact_store:
                sources.append({"type": "memory", "fact_store": self.fact_store, "session_id": session_id})
            if tools:
                sources.append({"type": "tools", "results": tools})
            context_layer_cfg = getattr(self.config, 'context_layer', None) or {}
            budget = context_layer_cfg.get('budget', {"max_tokens": 2048})
            return await build_context({"query": query, "budget": budget, "sources": sources, "priority": context_layer_cfg.get('priority')})

        self.context = type("ContextNamespace", (), {"ask": staticmethod(_context_ask)})()
```

Note: check whether `VectraConfig` currently has a `context_layer` field at all (likely not, since this is new) — if `getattr(self.config, 'context_layer', None)` would raise on a Pydantic model without that field (rather than returning `None`), either add `context_layer: Optional[Dict[str, Any]] = None` to `VectraConfig` in `vectra/config.py` (matching the existing untyped-dict pattern used for `memory`/`query_planning`/etc.), or confirm Pydantic's `getattr` behavior on unknown fields first — do not guess, verify directly.

- [ ] **Step 5: Run and verify all pass**

Run: `pytest tests/test_core_context_ask.py -v`
Expected: 6 tests pass.

- [ ] **Step 6: Run the full suite**

Run: `pytest`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add vectra/core.py vectra/config.py tests/test_core_context_ask.py
git commit -m "feat: add VectraClient.context.ask (tier-2 wrapper over build_context), enforcing guardrails/middleware from the start"
```
(Omit `vectra/config.py` if Step 4's research found no schema change was actually needed.)

---

### Task 7: Refactor `query_rag` — reserve a budgeted slice for history via `build_context`

**Files:**
- Modify: `vectra/core.py`
- Create: `tests/test_query_rag_context_layer_refactor.py`

**Interfaces:**
- `query_rag`'s public signature is UNCHANGED.
- `_build_context_parts` gains an optional 3rd parameter `budget_override` (defaulting to reading from `config.query_planning` as before, zero behavior change when omitted).
- History is now fetched, then routed through `build_context` with a RESERVED budget slice (half of the total, matching the JS sibling's design), and the REMAINING budget is passed to `_build_context_parts` for docs. `_build_context_parts`'s doc-formatting/`doc_map`-building logic is otherwise completely untouched — this deliberately avoids reconstructing `doc_map` from `build_context`'s generic output, which would risk breaking citation-index alignment (the same judgment call the JS sibling's implementer made after finding the naive approach riskier than necessary).
- Add a `history` source type to `build_context` in this task (small, only needed here):

```python
        if source.get("type") == "history":
            for m in source.get("messages", []):
                content = f"{str(m['role']).upper()}: {m['content']}"
                tokens = estimate_tokens_cached(content)
                if used + tokens > max_tokens:
                    dropped.append({"source": "history"})
                    continue
                parts.append({"source": "history", "type": "history", "content": content, "tokens": tokens})
                used += tokens
```

- [ ] **Step 1: Read the CURRENT exact state of `query_rag`'s context-building block**

Read `vectra/core.py` around `_build_context_parts`'s call site and the `history_text` block (confirmed at research time near lines 789-815, but re-confirm exact current line numbers after Tasks 1-6). Read the full surrounding `query_rag` function to understand every existing behavior that must be preserved: citations numbering, grounding snippet injection (strict vs. non-strict), the custom `config.prompts['query']` template path, default prompt construction for citations-enabled/disabled.

- [ ] **Step 2: Write the discriminating regression test FIRST**

Create `tests/test_query_rag_context_layer_refactor.py` — must fail against the CURRENT unfixed code:

```python
from unittest.mock import AsyncMock
from vectra.core import VectraClient
from vectra.config import VectraConfig, EmbeddingConfig, LLMConfig, DatabaseConfig


def make_config():
    return VectraConfig(
        embedding=EmbeddingConfig(provider="openai", api_key="test-key", model_name="text-embedding-3-small"),
        llm=LLMConfig(provider="openai", api_key="test-key", model_name="gpt-4o-mini"),
        database=DatabaseConfig(type="postgres", client_instance=object()),
        memory={"enabled": True, "type": "in-memory"},
        query_planning={"token_budget": 15},
    )


class TestQueryRagHistoryBudget:
    async def test_long_history_no_longer_bypasses_the_budget_uncounted(self):
        client = VectraClient(make_config())
        client.embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
        client.vector_store.similarity_search = AsyncMock(return_value=[{"content": "short doc", "metadata": {}}])
        client.llm.generate = AsyncMock(return_value="an answer")

        for i in range(20):
            client.history.add_message("session-1", "user", f"This is a fairly long historical message number {i} with a lot of extra real words padded in to make it substantial and unmistakably large for the purposes of this specific regression test.")

        await client.query_rag("what is this?", None, False, "session-1")

        prompt_sent = client.llm.generate.call_args[0][0]
        prompt_token_estimate = client._token_estimate(prompt_sent)

        assert prompt_token_estimate < 150
```

- [ ] **Step 3: Run and verify it fails against the CURRENT code**

Run: `pytest tests/test_query_rag_context_layer_refactor.py -v`
Expected: FAIL. If it does NOT fail, increase seeded message count/length until it genuinely discriminates — verify before proceeding (the JS sibling's first attempt at this exact test needed strengthening for the same reason).

- [ ] **Step 4: Implement the refactor**

Replace `_build_context_parts`'s call and the separate `history_text` block with: fetch history first, route through `build_context` with `max_tokens = total_budget // 2`, then call `_build_context_parts(boosted, query, total_budget - history_tokens_used)`. Preserve every existing citations/grounding/prompt-construction line verbatim — only the budget passed to `_build_context_parts` and where `history_text` originates change.

Add `history_source_type` support to `build_context` per this task's Interfaces section above before writing this code.

- [ ] **Step 5: Run the regression test and verify it now passes**

Run: `pytest tests/test_query_rag_context_layer_refactor.py -v`
Expected: PASS.

- [ ] **Step 6: Run the ENTIRE existing test suite**

Run: `pytest`
Expected: ALL tests pass, including every existing `query_rag`-related test. If anything fails, find and fix the real gap — do not weaken any existing assertion.

- [ ] **Step 7: Commit**

```bash
git add vectra/core.py vectra/context_layer.py tests/test_query_rag_context_layer_refactor.py
git commit -m "refactor: query_rag reserves a budgeted slice for history via build_context, fixing the unbounded history/token-budget bug (doc_map/citation alignment untouched)"
```

---

## Self-Review Notes

- **Spec coverage**: mirrors vectra-js's Phase 2 exactly — Tier 1 `build_context` (Tasks 1-5), Tier 2 `context.ask` (Task 6, WITH the security fix baked in from the start), backward-compatible `query_rag` refactor (Task 7, using the same lower-risk budget-reservation design the JS sibling's implementer converged on).
- **Placeholder scan**: no TBD/TODO.
- **Lessons carried over from the JS sibling's self-review, applied from Task 1 rather than discovered later**: bounded/LRU token cache (not unbounded), a verified-working spy-observability pattern for the cache test (Python's `patch(..., wraps=...)` on a module-level name is directly observable by internal calls, unlike JS's `jest.spyOn` on a CommonJS export — confirmed this actually holds via the deliberate-break verification in Task 1, not assumed), `context.ask` guardrails/middleware enforcement (Task 6), and the surgical (not full-reconstruction) `query_rag` refactor approach (Task 7).
- **Type/interface consistency**: `build_context`'s dict shape (`parts`/`text`/`tokens_used`/`tokens_budget`/`dropped`/`warnings`) matches the JS sibling's `PackedContext` shape 1:1 (snake_case), verified no drift across tasks.
