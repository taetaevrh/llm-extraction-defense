"""Phase A: conservative usage inventory and reserved-only validation selection."""

from collections import Counter, defaultdict
from pathlib import Path
import json
import random

from src.data import text_hash
from src.task import LABELS


def choose_validation(reserved, excluded_ids, excluded_hashes, per_class, seed):
    rng = random.Random(seed)
    selected = []
    for label in LABELS:
        pool = sorted((r for r in reserved if r["label"] == label and r["id"] not in excluded_ids
                       and r["text_sha256"] not in excluded_hashes), key=lambda r: r["id"])
        rng.shuffle(pool)
        if len(pool) < per_class:
            raise ValueError(f"Insufficient untouched reserved validation: {label}: {len(pool)} < {per_class}")
        selected.extend(pool[:per_class])
    rng.shuffle(selected)
    return selected


def unused_inventory(source, reserved_hashes):
    labels = sorted({r["label"] for rows in source.values() for r in rows})
    counts = {label: {"raw_train": 0, "raw_test": 0, "remaining_train": 0, "remaining_test": 0,
                      "remaining_total": 0, "duplicates_removed": 0, "excluded_prior_text_rows": 0,
                      "in_eight_intent_scope": label in LABELS} for label in labels}
    remaining, seen = [], set()
    for split in ("test", "train"):
        for row in source[split]:
            record = counts[row["label"]]
            record[f"raw_{split}"] += 1
            fingerprint = text_hash(row["text"])
            if fingerprint in reserved_hashes:
                record["excluded_prior_text_rows"] += 1
                continue
            if fingerprint in seen:
                record["duplicates_removed"] += 1
                continue
            seen.add(fingerprint)
            record[f"remaining_{split}"] += 1
            record["remaining_total"] += 1
            remaining.append({"id": row["id"], "label": row["label"], "text_sha256": fingerprint})
    return counts, remaining


def audit_prior_usage(root, source, config):
    root = Path(root)
    split_path = root / "docs/banking77_split_manifest.json"
    splits = json.loads(split_path.read_text())["subsets"]
    rows_by_id = {r["id"]: r for rows in source.values() for r in rows}
    assigned = {r["id"]: r for rows in splits.values() for r in rows}
    for row in assigned.values():
        if text_hash(rows_by_id[row["id"]]["text"]) != row["text_sha256"]:
            raise ValueError("Prior split hash mismatches pinned dataset")
    raw_run = root / config["prior_run"]
    required = ["extraction_responses.jsonl", "teacher_test.jsonl", "teacher_legitimate.jsonl", "smoke/responses.jsonl"]
    observed_ids, smoke_ids = set(), set()
    artifacts = []
    for directory in sorted((root / "experiments/runs").glob("*banking77*")):
        for filename in required:
            path = directory / filename
            if not path.exists():
                if directory == raw_run:
                    raise FileNotFoundError(f"Missing historical artifact: {path}")
                continue
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            ids = {r["id"] for r in records}
            if not ids.issubset(assigned):
                raise ValueError("Uninventoried prior BANKING77 query rows found")
            observed_ids.update(ids)
            if filename.startswith("smoke/"):
                smoke_ids.update(ids)
            artifacts.append({"path": path.relative_to(root).as_posix(), "records": len(records), "row_ids": sorted(ids)})
    reserved_hashes = {r["text_sha256"] for r in assigned.values()}
    # All historical validation reservations remain excluded from future tests,
    # including rows not yet queried. B reuses only non-smoke reserved rows.
    counts, remaining = unused_inventory(source, reserved_hashes)
    forbidden_ids = {r["id"] for name in ("extraction", "test", "legitimate") for r in splits[name]} | smoke_ids
    forbidden_hashes = {assigned[i]["text_sha256"] for i in forbidden_ids}
    validation = choose_validation(splits["validation"], forbidden_ids, forbidden_hashes,
                                   config["validation_per_class"], config["seed"])
    min_all = min(counts[label]["remaining_total"] for label in LABELS)
    min_test = min(counts[label]["remaining_test"] for label in LABELS)
    inventory = {
        "schema_version": 1, "dataset_revision": config["dataset_revision"],
        "canonicalization": "Task 02 NFKC/casefold/punctuation removal/whitespace collapse",
        "per_class": counts, "prior_assigned_unique": len(assigned), "prior_observed_unique": len(observed_ids),
        "prior_smoke_unique": len(smoke_ids), "prior_artifacts": artifacts,
        "prior_subsets": splits, "task03_validation": validation,
        "task03_validation_class_counts": dict(Counter(r["label"] for r in validation)),
        "remaining_eight_intent_balanced_capacity": {"mixed_official_sources": min_all * 8,
            "per_class_mixed": min_all, "official_test_only": min_test * 8, "per_class_official_test": min_test},
        "remaining_unused_row_ids_and_hashes": remaining,
        "future_test_exclusion": "All 900 prior assigned text hashes excluded; B uses only reserved validation, so no previously untouched pool is consumed.",
        "independence_limit": "Canonical distinctness cannot rule out paraphrase overlap or unknown model-pretraining exposure.",
    }
    return inventory, [rows_by_id[row["id"]] for row in validation]
