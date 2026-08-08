import re

# Best-effort, regex-based PII detection — not a comprehensive moderation
# system. Flags emails, phone numbers, SSN-shaped numbers, and long digit
# runs (credit-card-shaped). False positives are possible on legitimate
# long numeric identifiers; that's an accepted tradeoff of a fast, local,
# no-external-dependency check. Mirrors vectra-js's src/guardrails.js for
# feature parity between the two SDKs.
#
# NOTE: PII_PATTERNS[0] must remain the email pattern. It can only ever
# match text containing '@', so check_guardrails skips it entirely on text
# with no '@' via a cheap O(n) pre-check — without that pre-check, the
# email pattern exhibits catastrophic backtracking (ReDoS) on long text
# with no '@' character. The email pattern's quantifiers are also bounded
# ({1,64} / {1,255} / {2,24}, matching RFC 5321's practical limits) rather
# than unbounded (`+`) — without that bound, text that DOES contain '@' but
# no valid match (e.g. a long run of non-domain characters) still triggers
# catastrophic backtracking, defeating the pre-check above.
PII_PATTERNS = [
    ("email", re.compile(r"[a-zA-Z0-9._%+-]{1,64}@[a-zA-Z0-9.-]{1,255}\.[a-zA-Z]{2,24}")),
    ("phone", re.compile(r"(\+?\d{1,2}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("long_digit_run", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
]

# Minimal seed list of clearly harmful query patterns. This is a baseline,
# not exhaustive content moderation — extend DEFAULT_BLOCKED_TERMS for your
# deployment's needs, or replace check_guardrails' content_filter branch
# with an LLM-based classifier if you need semantic (not just keyword)
# coverage. Mirrors vectra-js's src/guardrails.js DEFAULT_BLOCKED_TERMS.
DEFAULT_BLOCKED_TERMS = [
    "how to make a bomb",
    "how to build a bomb",
    "how to make explosives",
    "how to synthesize a bioweapon",
]


def check_guardrails(query: str, guardrails_config) -> None:
    if guardrails_config is None:
        return
    text = str(query or "")

    max_len = getattr(guardrails_config, "max_query_length", None)
    # Explicit type check rather than `if max_len and ...`: Python's `0` is
    # falsy, so a truthiness check would silently disable the length check
    # when max_query_length=0. `bool` is excluded because it is a subclass
    # of `int` in Python (isinstance(True, int) is True), so a wrongly
    # typed max_query_length=True would otherwise be treated as a limit of 1.
    if isinstance(max_len, int) and not isinstance(max_len, bool) and len(text) > max_len:
        raise ValueError(f"GuardrailViolation: query exceeds max_query_length ({len(text)} > {max_len})")

    if getattr(guardrails_config, "block_pii", False):
        # The email pattern can only ever match text containing '@'; skip it
        # entirely on text with no '@' to avoid catastrophic backtracking on
        # long inputs with no match (see ReDoS fix — cheap O(n) pre-check).
        if "@" in text and PII_PATTERNS[0][1].search(text):
            raise ValueError(f"GuardrailViolation: possible PII detected ({PII_PATTERNS[0][0]})")
        for name, pattern in PII_PATTERNS[1:]:
            if pattern.search(text):
                raise ValueError(f"GuardrailViolation: possible PII detected ({name})")

    if getattr(guardrails_config, "content_filter", False):
        lower = text.lower()
        # Combine the built-in seed list with any user-supplied blocked_terms
        # from GuardrailConfig, so deployments can extend content filtering
        # from the public API instead of editing DEFAULT_BLOCKED_TERMS directly.
        terms = list(DEFAULT_BLOCKED_TERMS) + list(getattr(guardrails_config, "blocked_terms", None) or [])
        for term in terms:
            if term.lower() in lower:
                raise ValueError("GuardrailViolation: query blocked by content filter")
