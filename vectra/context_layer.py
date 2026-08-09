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
