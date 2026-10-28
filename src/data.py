"""Trusted curator only. Ground truth never crosses the attacker boundary."""

from collections import Counter, defaultdict
import hashlib
import random
import re
import unicodedata

from src.task import LABELS


def canonical_text(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def text_hash(text):
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def validate_disjoint(subsets):
    seen_texts, seen_ids = set(), set()
    for name, rows in subsets.items():
        for row in rows:
            fingerprint = text_hash(row["text"])
            if fingerprint in seen_texts or row["id"] in seen_ids:
                raise ValueError(f"Duplicate/leakage in subset {name}")
            seen_texts.add(fingerprint)
            seen_ids.add(row["id"])


def freeze_splits(source, sizes, seed):
    """Balanced deterministic quotas. Test source has priority during dedup."""
    grouped = {key: defaultdict(list) for key in ("train", "test")}
    seen, removed = set(), Counter()
    for split in ("test", "train"):
        for row in source[split]:
            if row["label"] not in LABELS:
                continue
            fingerprint = text_hash(row["text"])
            if not canonical_text(row["text"]) or fingerprint in seen:
                removed[split] += 1
                continue
            seen.add(fingerprint)
            grouped[split][row["label"]].append(dict(row))
    available = {split: {label: len(grouped[split][label]) for label in LABELS} for split in grouped}
    rng = random.Random(seed)
    for split in ("train", "test"):
        for label in LABELS:
            rng.shuffle(grouped[split][label])
    result = {}
    for name in ("test", "extraction", "validation", "legitimate"):
        count = sizes[name]
        pool = grouped["test" if name in ("test", "legitimate") else "train"]
        rows = []
        for index, label in enumerate(LABELS):
            quota = count // len(LABELS) + int(index < count % len(LABELS))
            if quota < 1 or len(pool[label]) < quota:
                raise ValueError(f"Insufficient distinct examples for {name}/{label}: need {quota}, available {len(pool[label])}")
            rows.extend(pool[label][:quota])
            del pool[label][:quota]
        rng.shuffle(rows)
        result[name] = rows
    validate_disjoint(result)
    return result, {"unique_available_before_sampling": available, "normalized_duplicates_removed": dict(removed),
                    "source_splits": {"extraction": "train", "validation": "train", "test": "test", "legitimate": "test"}}


def attacker_texts(rows):
    return [{"id": row["id"], "text": row["text"]} for row in rows]


def split_manifest(subsets, audit, dataset_revision, seed):
    """Only row IDs, labels used by curator, and hashes; never raw dataset text."""
    return {
        "schema_version": 1, "dataset_revision": dataset_revision, "seed": seed,
        "canonicalization": "NFKC, casefold, remove punctuation, collapse whitespace",
        "audit": audit,
        "subsets": {name: [{"id": r["id"], "label": r["label"], "text_sha256": text_hash(r["text"])} for r in rows] for name, rows in subsets.items()},
        "class_counts": {name: dict(Counter(row["label"] for row in rows)) for name, rows in subsets.items()},
    }


def load_banking77(revision):
    from datasets import load_dataset
    dataset = load_dataset("PolyAI/banking77", revision=revision)
    names = dataset["train"].features["label"].names
    if not set(LABELS).issubset(names):
        raise ValueError("Approved labels missing from pinned dataset")
    return {
        split: [{"id": f"{split}:{index}", "text": row["text"], "label": names[row["label"]]} for index, row in enumerate(dataset[split])]
        for split in ("train", "test")
    }
