import time

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

    def test_rejects_when_max_query_length_is_zero(self):
        # Regression test: max_query_length=0 must not be silently treated as
        # "no limit" due to Python's falsy-zero. A plain `if max_len and ...`
        # check would skip enforcement entirely when max_len == 0.
        with pytest.raises(ValueError, match="GuardrailViolation: query exceeds max_query_length"):
            check_guardrails("a" * 10, GuardrailConfig(max_query_length=0))


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

    def test_no_redos_on_long_text_without_at_sign(self):
        # Regression test: the email regex exhibits catastrophic backtracking
        # on long text with no '@' character (measured 70+ seconds on 200k
        # chars in the vectra-js sibling). check_guardrails must short-circuit
        # the email pattern behind a cheap `"@" in text` pre-check so this
        # completes quickly. This also doubles as a functional test: no PII
        # is present, so no violation should be raised.
        text = "a" * 50000
        start = time.monotonic()
        check_guardrails(text, GuardrailConfig(block_pii=True, max_query_length=len(text)))
        elapsed = time.monotonic() - start
        assert elapsed < 1.0


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

    def test_custom_blocked_terms_combine_with_defaults(self):
        # Custom blocked_terms must extend, not replace, DEFAULT_BLOCKED_TERMS.
        config = GuardrailConfig(content_filter=True, blocked_terms=["forbidden custom phrase"])
        with pytest.raises(ValueError, match="GuardrailViolation: query blocked by content filter"):
            check_guardrails("this contains a forbidden custom phrase right here", config)
        with pytest.raises(ValueError, match="GuardrailViolation: query blocked by content filter"):
            check_guardrails("how to make a bomb at home", config)
