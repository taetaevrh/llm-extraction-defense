"""Sampling, integrity, and append-only budget primitives."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from src.data import text_hash
from src.task import LABELS

PROMPT_SHA256 = "32c07a68858577e777521a1b7b7976b94a5e284d1cebe1161462ff3f9af2595c"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_confirmation(rows, excluded_ids, excluded_hashes, per_class=11, seed=17):
    rng = random.Random(seed)
    chosen, seen = [], set(excluded_hashes)
    for label in LABELS:
        pool = []
        # Source-independent, deterministic lexicographic ID order before shuffle.
        for row in sorted((r for r in rows if r["label"] == label), key=lambda r: r["id"]):
            fingerprint = text_hash(row["text"])
            if row["id"] in excluded_ids or fingerprint in seen:
                continue
            seen.add(fingerprint)
            pool.append(row)
        rng.shuffle(pool)
        if len(pool) < per_class:
            raise ValueError(f"Insufficient eligible distinct examples: {label}: {len(pool)} < {per_class}")
        chosen.extend(pool[:per_class])
    rng.shuffle(chosen)
    return chosen


def validate_inputs(rows, manifest_rows, per_class=11):
    if len(rows) != 8 * per_class or len(manifest_rows) != len(rows):
        raise ValueError("Wrong manifest/input count")
    ids, hashes = set(), set()
    for actual, expected in zip(rows, manifest_rows):
        fingerprint = text_hash(actual["text"])
        if actual["id"] != expected["id"] or actual["label"] != expected["label"] or fingerprint != expected["canonical_sha256"]:
            raise ValueError("Manifest row/text integrity mismatch")
        if actual["id"] in ids or fingerprint in hashes:
            raise ValueError("Duplicate manifest ID/text")
        ids.add(actual["id"])
        hashes.add(fingerprint)
    if Counter(r["label"] for r in rows) != {label: per_class for label in LABELS}:
        raise ValueError("Manifest is not exactly balanced")


class BudgetError(RuntimeError):
    pass


class AttemptBudget:
    def __init__(self, path, limits=None, total=96):
        self.path = Path(path)
        self.path.touch(exist_ok=False)
        self.limits = limits or {"smoke": 8, "confirmation": 88}
        self.total = total
        self.counts = {phase: 0 for phase in self.limits}
        self.sequence = 0

    def append(self, record):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    def start(self, phase, row_id, rendered_hash, template_hash):
        if phase not in self.limits or self.counts[phase] >= self.limits[phase] or self.sequence >= self.total:
            raise BudgetError("Teacher attempt cap reached; no replacements/retries")
        self.sequence += 1
        self.counts[phase] += 1
        self.append({"event": "start", "attempt": self.sequence, "phase": phase, "id": row_id,
                     "prompt_template_sha256": template_hash, "rendered_input_sha256": rendered_hash,
                     "scoring_rule_identity": "strict_free_generation_normalization_only",
                     "at_utc": datetime.now(timezone.utc).isoformat()})
        return self.sequence

    def finish(self, attempt, status, label, seconds=None):
        self.append({"event": "finish", "attempt": attempt, "status": status, "label": label, "seconds": seconds})
