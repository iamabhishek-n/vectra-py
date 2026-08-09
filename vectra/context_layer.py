from typing import Any, Dict, List, Optional
import asyncio
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
        _token_cache[key] = count  # refresh recency (dict preserves insertion order)
        return count
    encoded = _encode_for_test(key)
    if encoded:
        count = len(encoded)
    else:
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


async def build_context(input: Dict[str, Any]) -> Dict[str, Any]:
    query = input.get("query")
    budget = input.get("budget") or {}
    sources = input.get("sources") or []
    priority = input.get("priority")

    max_tokens = budget.get("max_tokens", 2048)
    parts: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    used = 0

    if priority:
        def rank(source):
            t = source.get("type")
            return priority.index(t) if t in priority else len(priority)
        ordered_sources = sorted(sources, key=rank)
    else:
        ordered_sources = sources

    for source in ordered_sources:
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
                try:
                    return await asyncio.wait_for(coro, timeout=timeout_s)
                except asyncio.TimeoutError:
                    # asyncio.TimeoutError's str() is empty by default -- give it a
                    # real, descriptive message so callers/warnings can identify it.
                    raise TimeoutError(f"timeout after {timeout_s}s")

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
                content = item.get("content", "")
                tokens = estimate_tokens_cached(content)
                if used + tokens > max_tokens:
                    dropped.append({"source": "docs", "metadata": item.get("metadata", {})})
                    continue
                parts.append({"source": "docs", "type": "docs", "content": content, "tokens": tokens})
                used += tokens

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

        if source.get("type") == "history":
            for m in source.get("messages", []):
                content = f"{str(m['role']).upper()}: {m['content']}"
                tokens = estimate_tokens_cached(content)
                if used + tokens > max_tokens:
                    dropped.append({"source": "history"})
                    continue
                parts.append({"source": "history", "type": "history", "content": content, "tokens": tokens})
                used += tokens

        if source.get("type") == "tools":
            for result_item in source.get("results", []):
                content = f"Tool: {result_item['name']}\nResult: {result_item['output']}"
                tokens = estimate_tokens_cached(content)
                if used + tokens > max_tokens:
                    dropped.append({"source": "tools", "name": result_item["name"]})
                    continue
                parts.append({"source": "tools", "type": "tools", "content": content, "tokens": tokens})
                used += tokens

    return {
        "parts": parts,
        "text": "\n---\n".join(p["content"] for p in parts),
        "tokens_used": used,
        "tokens_budget": max_tokens,
        "dropped": dropped,
        "warnings": warnings,
    }
