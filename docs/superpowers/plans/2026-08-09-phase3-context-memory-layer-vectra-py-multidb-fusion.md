# Context & Memory Layer Phase 3 — vectra-py Multi-DB Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Mirror vectra-js's Phase 3 — extend `build_context`'s `docs` source to fan out across multiple vector store instances concurrently (`asyncio.gather`) and RRF-fuse the results, with per-store timeout/circuit-breaker. Same architecture, same task shape as the JS sibling (already reviewed and merged there).

**Architecture:** New docs-source shape alongside the existing pre-fetched-`items` shape: `{"type": "docs", "stores": [store1, store2], "vector": ..., "query": ..., "limit": ..., "strategy": ..., "timeout_s": ...}`. Fan out via `asyncio.gather(..., return_exceptions=True)` wrapped per-store with `asyncio.wait_for` for the timeout, RRF-fuse successful results (reusing vectra-py's existing `1/(k+rank+1)`, k=60 formula from `VectraClient._reciprocal_rank_fusion`), report per-store failures in `warnings`.

**Tech Stack:** No new dependencies (`asyncio` is stdlib).

## Global Constraints

- Concurrent fan-out only (`asyncio.gather`), never sequential.
- Per-store timeout (default 5.0s, configurable via `source["timeout_s"]`) + circuit-breaker: a failed/timed-out store produces a `warnings` entry, fan-out proceeds with survivors.
- RRF fusion constant k=60, matching every existing fusion site in this codebase.
- The existing `items`-based docs source (Phase 2) is unchanged — additive only.
- No direct-to-master commits, no force-push, no skipped hooks.
- Every task ends with `pytest` passing.

---

### Task 1: Concurrent multi-store fan-out with RRF fusion

**Files:**
- Modify: `vectra/context_layer.py`
- Create: `tests/test_context_layer_multidb.py`

**Interfaces:**
- Produces: `build_context`'s `docs` branch recognizes `source["stores"]` (a list of objects each exposing `async similarity_search(vector, limit, filter)` and optionally `async hybrid_search(text, vector, limit, filter)`). Concurrent fan-out via `asyncio.gather(..., return_exceptions=True)`, RRF-fused, packed like the `items` path.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_context_layer_multidb.py`:

```python
from unittest.mock import AsyncMock
from vectra.context_layer import build_context, _clear_token_cache


def make_store(results):
    store = type("Store", (), {})()
    store.similarity_search = AsyncMock(return_value=results)
    return store


class TestBuildContextMultiDb:
    def setup_method(self):
        _clear_token_cache()

    async def test_fans_out_to_all_stores_concurrently_and_fuses_results(self):
        store_a = make_store([{"content": "doc from A", "metadata": {}, "score": 0.9}])
        store_b = make_store([{"content": "doc from B", "metadata": {}, "score": 0.8}])

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "stores": [store_a, store_b], "vector": [0.1, 0.2], "limit": 5}],
        })

        store_a.similarity_search.assert_called_with([0.1, 0.2], 5, None)
        store_b.similarity_search.assert_called_with([0.1, 0.2], 5, None)
        contents = [p["content"] for p in result["parts"]]
        assert "doc from A" in contents
        assert "doc from B" in contents

    async def test_doc_findable_only_in_one_store_survives_fusion(self):
        store_a = make_store([
            {"content": "shared doc", "metadata": {}, "score": 0.5},
            {"content": "only in A", "metadata": {}, "score": 0.4},
        ])
        store_b = make_store([{"content": "shared doc", "metadata": {}, "score": 0.5}])

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "stores": [store_a, store_b], "vector": [0.1], "limit": 5}],
        })

        contents = [p["content"] for p in result["parts"]]
        assert "only in A" in contents
        assert "shared doc" in contents

    async def test_dedupes_by_content_across_stores(self):
        store_a = make_store([{"content": "dup", "metadata": {}, "score": 0.9}])
        store_b = make_store([{"content": "dup", "metadata": {}, "score": 0.9}])

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "stores": [store_a, store_b], "vector": [0.1], "limit": 5}],
        })

        assert len([p for p in result["parts"] if p["content"] == "dup"]) == 1

    async def test_uses_hybrid_search_when_strategy_is_hybrid(self):
        store = type("Store", (), {})()
        store.similarity_search = AsyncMock()
        store.hybrid_search = AsyncMock(return_value=[{"content": "hybrid result", "metadata": {}, "score": 0.9}])

        await build_context({
            "query": "the query text",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "stores": [store], "vector": [0.1], "limit": 5, "strategy": "hybrid"}],
        })

        store.hybrid_search.assert_called_with("the query text", [0.1], 5, None)
        store.similarity_search.assert_not_called()
```

- [ ] **Step 2: Run and verify they fail**

Run: `pytest tests/test_context_layer_multidb.py -v`
Expected: FAIL — `source["stores"]` not recognized yet.

- [ ] **Step 3: Implement**

Add to `vectra/context_layer.py`:

```python
def _reciprocal_rank_fusion(result_lists, k=60):
    scores = {}
    content_map = {}
    for lst in result_lists:
        for rank, doc in enumerate(lst):
            content = doc["content"]
            if content not in content_map:
                content_map[content] = doc
            scores[content] = scores.get(content, 0) + 1 / (k + rank + 1)
    ordered = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    return [content_map[c] for c in ordered]
```

Inside `build_context`'s loop, before the existing `items`-based `docs` handling, add (and change the existing `if source.get("type") == "docs":` to `elif` so only one branch runs per source):

```python
        if source.get("type") == "docs" and source.get("stores"):
            stores = source["stores"]
            vector = source.get("vector")
            limit = source.get("limit", 5)
            filt = source.get("filter")
            strategy = source.get("strategy")
            timeout_s = source.get("timeout_s", 5.0)

            async def _call_one(store):
                if strategy == "hybrid" and hasattr(store, "hybrid_search"):
                    coro = store.hybrid_search(query, vector, limit, filt)
                else:
                    coro = store.similarity_search(vector, limit, filt)
                return await asyncio.wait_for(coro, timeout=timeout_s)

            results = await asyncio.gather(*[_call_one(s) for s in stores], return_exceptions=True)
            successful_lists = []
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    warnings.append({"store": i, "error": str(res)})
                else:
                    successful_lists.append(res)

            fused = _reciprocal_rank_fusion(successful_lists)
            for doc in fused:
                content = doc.get("content", "")
                tokens = estimate_tokens_cached(content)
                if used + tokens > max_tokens:
                    dropped.append({"source": "docs", "metadata": doc.get("metadata", {})})
                    continue
                parts.append({"source": "docs", "type": "docs", "content": content, "tokens": tokens})
                used += tokens
        elif source.get("type") == "docs":
            for item in source.get("items", []):
```

(The `elif` swap means the rest of the original `items` for-loop body stays exactly as-is, just re-indented under `elif` instead of `if`.) Add `import asyncio` to the top of the file if not present, and add a mutable `warnings = []` list near `parts`/`dropped` (declared once, before the loop, used at the end of the function instead of the current hardcoded `"warnings": []`).

- [ ] **Step 4: Run and verify all pass**

Run: `pytest tests/test_context_layer_multidb.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Run the full suite**

Run: `pytest`
Expected: all pass, no regressions to Phase 2's `items`-based docs-source tests.

- [ ] **Step 6: Commit**

```bash
git add vectra/context_layer.py tests/test_context_layer_multidb.py
git commit -m "feat: multi-db fan-out for build_context's docs source, concurrent + RRF-fused"
```

---

### Task 2: Per-store timeout + circuit-breaker

**Files:**
- Modify: `vectra/context_layer.py` (verify — Task 1's `asyncio.wait_for` + `return_exceptions=True` already implements this; this task is a dedicated proof, same as JS sibling's Task 2)
- Create: `tests/test_context_layer_multidb_timeout.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_context_layer_multidb_timeout.py`:

```python
import asyncio
import time
from unittest.mock import AsyncMock
from vectra.context_layer import build_context, _clear_token_cache


def slow_store(delay_s, result):
    async def _search(*args, **kwargs):
        await asyncio.sleep(delay_s)
        return result
    store = type("Store", (), {})()
    store.similarity_search = _search
    return store


def throwing_store():
    store = type("Store", (), {})()
    store.similarity_search = AsyncMock(side_effect=Exception("connection refused"))
    return store


class TestBuildContextMultiDbTimeout:
    def setup_method(self):
        _clear_token_cache()

    async def test_a_store_that_throws_produces_a_warning_not_a_failed_call(self):
        good_store = type("Store", (), {})()
        good_store.similarity_search = AsyncMock(return_value=[{"content": "good doc", "metadata": {}, "score": 0.9}])
        bad_store = throwing_store()

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "stores": [good_store, bad_store], "vector": [0.1], "limit": 5}],
        })

        assert any(p["content"] == "good doc" for p in result["parts"])
        assert len(result["warnings"]) > 0
        assert "connection refused" in result["warnings"][0]["error"]

    async def test_a_store_that_exceeds_timeout_produces_a_warning_does_not_hang(self):
        fast_store = type("Store", (), {})()
        fast_store.similarity_search = AsyncMock(return_value=[{"content": "fast doc", "metadata": {}, "score": 0.9}])
        hanging_store = slow_store(10, [{"content": "never arrives", "metadata": {}, "score": 0.9}])

        result = await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "stores": [fast_store, hanging_store], "vector": [0.1], "limit": 5, "timeout_s": 0.05}],
        })

        assert any(p["content"] == "fast doc" for p in result["parts"])
        assert not any(p["content"] == "never arrives" for p in result["parts"])
        assert any("timeout" in str(w["error"]).lower() for w in result["warnings"])

    async def test_total_wallclock_bounded_by_slowest_allowed_store_not_the_sum(self):
        store_a = slow_store(0.03, [{"content": "a", "metadata": {}, "score": 0.9}])
        store_b = slow_store(0.03, [{"content": "b", "metadata": {}, "score": 0.9}])
        store_c = slow_store(0.03, [{"content": "c", "metadata": {}, "score": 0.9}])

        start = time.monotonic()
        await build_context({
            "query": "q",
            "budget": {"max_tokens": 1000},
            "sources": [{"type": "docs", "stores": [store_a, store_b, store_c], "vector": [0.1], "limit": 5}],
        })
        elapsed = time.monotonic() - start

        assert elapsed < 0.08
```

- [ ] **Step 2: Run and verify they pass**

Run: `pytest tests/test_context_layer_multidb_timeout.py -v`
Expected: all 3 pass already, since Task 1's `asyncio.wait_for` + `return_exceptions=True` already implements timeout/circuit-breaker. If any fail, fix `_call_one`/the gather call to match — do not weaken the tests.

- [ ] **Step 3: Run the full suite**

Run: `pytest`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_context_layer_multidb_timeout.py
git commit -m "test: add dedicated timeout/circuit-breaker coverage for multi-db fan-out (already correct from Task 1)"
```

---

## Self-Review Notes

- **Spec coverage**: mirrors vectra-js's Phase 3 exactly — concurrent fan-out + RRF fusion (Task 1, using `asyncio.gather`/`asyncio.wait_for` idiomatically instead of JS's `Promise.allSettled`/manual timeout race), dedicated timeout/circuit-breaker proof (Task 2).
- **Placeholder scan**: no TBD/TODO.
- **Idiomatic difference from JS, noted deliberately**: Python's `asyncio.wait_for` natively raises `asyncio.TimeoutError` on timeout, so no hand-rolled `_withTimeout` race helper (like JS's) is needed — `asyncio.gather(..., return_exceptions=True)` combined with `asyncio.wait_for` per-call is the idiomatic equivalent, not a divergence in behavior.
