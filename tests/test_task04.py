"""Synthetic fixtures for the frozen Task04 confirmation boundary."""

import json
import pytest

from src.task import LABELS
from src.data import text_hash
from src.task04.core import select_confirmation, validate_inputs, AttemptBudget, BudgetError
from src.task04.metrics import metrics_with_uncertainty, validate_aggregate
from src.task04.rebuild import reconstruct, build_summary
from src.task04.core import digest


def fixture_rows(n=3):
    return [{"id": f"train:{label}:{i}", "label": label, "text": f"Example {label} {i}"}
            for label in LABELS for i in range(n)]


def test_sampling_is_deterministic_and_excludes_all_known_texts_and_ids():
    rows = fixture_rows()
    ids = {f"train:{label}:0" for label in LABELS}
    hashes = {text_hash(r["text"]) for r in rows if r["id"] in ids}
    first = select_confirmation(rows, ids, hashes, per_class=2, seed=17)
    assert first == select_confirmation(list(reversed(rows)), ids, hashes, per_class=2, seed=17)
    assert len(first) == 16
    assert not ids.intersection(r["id"] for r in first)
    with pytest.raises(ValueError, match="Insufficient"):
        select_confirmation(rows, ids, hashes, per_class=3, seed=17)


def test_manifest_integrity_rejects_changed_text_and_duplicate_ids():
    rows = fixture_rows(1)
    manifest = [{"id": r["id"], "label": r["label"], "canonical_sha256": text_hash(r["text"])} for r in rows]
    validate_inputs(rows, manifest, per_class=1)
    rows[0]["text"] = "tampered"
    with pytest.raises(ValueError):
        validate_inputs(rows, manifest, per_class=1)
    rows = fixture_rows(1)
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError):
        validate_inputs(rows, manifest, per_class=1)


def test_budget_persists_attempt_before_backend_and_counts_failures(tmp_path):
    path = tmp_path / "attempts.jsonl"
    budget = AttemptBudget(path, {"smoke": 1, "confirmation": 2}, total=3)
    first = budget.start("smoke", "known", "rendered", "template")
    assert json.loads(path.read_text().splitlines()[0])["event"] == "start"
    budget.finish(first, "error", None)
    for index in range(2):
        attempt = budget.start("confirmation", str(index), "rendered", "template")
        budget.finish(attempt, "invalid", None)
    with pytest.raises(BudgetError):
        budget.start("confirmation", "extra", "rendered", "template")
    assert budget.counts == {"smoke": 1, "confirmation": 2}


def test_invalids_stay_in_denominator_and_confusion_recomputes_metrics():
    truth = list(LABELS)
    pred = list(LABELS)
    pred[0] = None
    result = metrics_with_uncertainty(pred, truth, resamples=30, seed=17)
    assert result["accuracy"] == 7 / 8
    assert result["macro_f1"] == 7 / 8
    assert result["per_class"][LABELS[0]]["recall"] == 0
    assert result["invalid"] == 1
    assert result["decision"]["passes_point_criteria"] is False
    validate_aggregate(result)
    result["correct"] = 8
    with pytest.raises(ValueError):
        validate_aggregate(result)


def test_offline_reconstruction_is_deterministic_and_requires_complete_rows():
    truth = list(LABELS)
    manifest = {"rows": [{"id": str(i), "label": label} for i, label in enumerate(truth)], "expected_samples": 8}
    predictions = [{"id": str(i), "raw": label, "label": label} for i, label in enumerate(truth)]
    config = {"statistics": {"bootstrap_resamples": 30, "bootstrap_seed": 17}}
    first = reconstruct(manifest, predictions, config)
    assert first == reconstruct(manifest, predictions, config)
    assert first["accuracy"] == 1
    with pytest.raises(ValueError):
        reconstruct(manifest, predictions[:-1], config)


def test_sampling_excludes_canonical_duplicates_under_new_ids():
    rows = fixture_rows(2)
    duplicate = {**rows[0], "id": "other", "text": "EXAMPLE card_arrival 0!!!"}
    hashes = {text_hash(rows[0]["text"])}
    chosen = select_confirmation([*rows, duplicate], set(), hashes, per_class=1)
    assert all(text_hash(r["text"]) not in hashes for r in chosen)


def test_blocked_rebuild_preserves_null_and_checks_local_hashes(tmp_path):
    manifest = {"run_id": "fixture", "status": "blocked", "failure": "resource gate", "executed_source_commit": "fixture", "frozen_hashes": {}}
    (tmp_path / "input_manifest.json").write_text(json.dumps({"source_composition": {}, "rows": [], "expected_samples": 88}))
    (tmp_path / "configuration.yaml").write_text("statistics: {}\n")
    (tmp_path / "attempts.jsonl").write_text("")
    manifest["raw_files_sha256"] = {"attempts.jsonl": digest(tmp_path / "attempts.jsonl")}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = build_summary(tmp_path)
    assert result["metrics"] is None
    assert result["unmeasured_reason"] == "resource gate"
    assert result["total_teacher_attempts"] == 0
    (tmp_path / "attempts.jsonl").write_text("tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_summary(tmp_path)
