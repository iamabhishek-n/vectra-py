from unittest.mock import patch
import vectra.core as core
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
        text = "a" * 400  # BPE efficiently handles repeated characters; 400 chars tokenize to ~50 tokens vs 100 from heuristic
        old_heuristic = max(1, (len(text) + 3) // 4)
        real = VectraClient._token_estimate(None, text)
        assert real < old_heuristic


class TestTokenEstimateOfflineFallback:
    def setup_method(self):
        # Reset the module-level lazy-encoder cache/sentinel so each test controls
        # whether tiktoken.get_encoding succeeds or fails, independent of test order.
        core._token_encoder = None
        core._token_encoder_unavailable = False

    def teardown_method(self):
        core._token_encoder = None
        core._token_encoder_unavailable = False

    def test_falls_back_to_char_heuristic_when_tiktoken_get_encoding_raises(self):
        text = "Hello, world! éè"  # mix of ascii + non-ascii to exercise both branches
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        non_ascii = len(text) - ascii_chars
        expected = max(1, (ascii_chars + 3) // 4 + non_ascii)

        with patch("vectra.core.tiktoken.get_encoding", side_effect=Exception("network error")):
            result = VectraClient._token_estimate(None, text)

        assert result == expected

    def test_does_not_retry_the_network_call_after_first_failure(self):
        with patch("vectra.core.tiktoken.get_encoding", side_effect=Exception("network error")) as mock_get_encoding:
            VectraClient._token_estimate(None, "first call")
            VectraClient._token_estimate(None, "second call")

        assert mock_get_encoding.call_count == 1
        assert core._token_encoder_unavailable is True
