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
