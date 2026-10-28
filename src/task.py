"""Frozen eight-intent task interface and conservative text normalization."""

import re

LABELS = (
    "card_arrival", "card_not_working", "change_pin", "exchange_rate",
    "lost_or_stolen_card", "pending_transfer", "transfer_timing", "verify_my_identity",
)
PROMPT_TEMPLATE = (
    "Classify the banking customer's message into exactly one intent.\n"
    "Allowed intents: " + ", ".join(LABELS) + ".\n"
    "Reply with only the exact intent name. Do not explain.\n"
    "Customer message:\n{text}\nIntent:"
)


def messages(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected nonempty text")
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}]


def normalize_label(raw):
    """Accept only a whole canonical label, optionally quoted/cased/punctuated."""
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower().strip('"\'`').strip()
    value = re.sub(r"[.!]$", "", value).strip()
    return value if value in LABELS else None
