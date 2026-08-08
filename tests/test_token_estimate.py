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
