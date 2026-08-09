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
