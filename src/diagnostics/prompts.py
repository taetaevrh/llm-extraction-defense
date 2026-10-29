"""Prompt family fixed before validation inference; no outcome-based edits."""

import hashlib
from src.task import LABELS, PROMPT_TEMPLATE

DEFINITIONS = {
    "card_arrival": "The customer is waiting for a bank card to arrive or asks about its delivery.",
    "card_not_working": "The customer's existing bank card does not work or is declined.",
    "change_pin": "The customer wants to change or reset the card PIN.",
    "exchange_rate": "The customer asks which currency exchange rate applies or how it is calculated.",
    "lost_or_stolen_card": "The customer's bank card has been lost or stolen.",
    "pending_transfer": "An already initiated bank transfer is pending or has not completed.",
    "transfer_timing": "The customer asks how long a bank transfer normally takes.",
    "verify_my_identity": "The customer asks how or why to verify their identity or which identity documents to provide.",
}


def prompt_variants():
    variants = []
    for offset in range(8):
        order = list(LABELS[offset:] + LABELS[:offset])
        template = PROMPT_TEMPLATE.replace(", ".join(LABELS), ", ".join(order))
        variants.append({"id": f"rotation_{offset}", "family": "rotation", "order": order,
                         "template": template, "max_input_tokens": 256,
                         "alias": "original" if offset == 0 else None})
    order = list(reversed(LABELS))
    variants.append({"id": "reverse", "family": "reverse", "order": order,
                     "template": PROMPT_TEMPLATE.replace(", ".join(LABELS), ", ".join(order)),
                     "max_input_tokens": 256, "alias": None})
    template = (
        "Classify the banking customer's message by the help they are requesting.\n"
        "Choose exactly one of these intent names, using the definitions below:\n"
        + "\n".join(f"{label}: {DEFINITIONS[label]}" for label in LABELS)
        + "\nReply with only the exact intent name. Do not explain.\n"
        "Customer message:\n{text}\nIntent:"
    )
    variants.append({"id": "definitions", "family": "definitions", "order": list(LABELS),
                     "template": template, "max_input_tokens": 512, "alias": None})
    for variant in variants:
        variant["sha256"] = hashlib.sha256(variant["template"].encode()).hexdigest()
    return variants
