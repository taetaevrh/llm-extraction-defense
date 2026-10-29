"""Task 03 diagnostics on synthetic data, without model downloads."""

import math
import pytest

from src.task import LABELS, PROMPT_TEMPLATE
from src.diagnostics.prompts import prompt_variants
from src.diagnostics.metrics import diagnostic_metrics, meets_quality_targets
from src.diagnostics.inventory import choose_validation, unused_inventory
from src.diagnostics.scoring import label_log_scores


def test_rotations_expose_every_label_at_every_position_once():
    variants = prompt_variants()
    rotations = [v for v in variants if v["family"] == "rotation"]
    assert len(rotations) == 8
    assert rotations[0]["template"] == PROMPT_TEMPLATE
    assert len({v["sha256"] for v in rotations}) == 8
    for label in LABELS:
        assert sorted(v["order"].index(label) for v in rotations) == list(range(8))
    assert next(v for v in variants if v["id"] == "reverse")["order"] == list(reversed(LABELS))
    assert len(variants) == 10
    assert len({v["sha256"] for v in variants}) == 10


def test_macro_f1_counts_invalid_as_a_false_negative_and_keeps_zero_classes():
    truth = list(LABELS)
    pred = list(LABELS)
    pred[0] = LABELS[1]
    pred[2] = None
    result = diagnostic_metrics(pred, truth, LABELS[-1])
    assert result["correct"] == 6
    assert result["invalid_rate"] == 1 / 8
    assert result["per_class"][LABELS[0]]["recall"] == 0
    assert result["per_class"][LABELS[1]]["precision"] == 1 / 2
    assert result["macro_f1"] == pytest.approx((5 + 2 / 3) / 8)
    assert result["final_option_rate"] == 1 / 8
    assert result["confusion"][LABELS[2]]["invalid"] == 1
    assert not meets_quality_targets(result)
    perfect = diagnostic_metrics(truth, truth, LABELS[-1])
    assert meets_quality_targets(perfect)


def test_dominant_class_fails_quality_even_with_good_accuracy():
    truth = [LABELS[0]] * 8 + list(LABELS[1:])
    result = diagnostic_metrics(truth, truth, LABELS[-1])
    assert result["accuracy"] == 1
    assert not meets_quality_targets(result)


def test_validation_excludes_training_smoke_and_known_test_hashes():
    records = [{"id": f"train:{label}:{i}", "label": label, "text_sha256": f"hash-{label}-{i}"}
               for label in LABELS for i in range(3)]
    used_ids = {f"train:{label}:0" for label in LABELS}
    chosen = choose_validation(records, used_ids, set(), per_class=2, seed=17)
    assert len(chosen) == 16
    assert not used_ids.intersection(r["id"] for r in chosen)
    with pytest.raises(ValueError, match="Insufficient"):
        choose_validation(records, used_ids, {records[1]["text_sha256"]}, per_class=2, seed=17)


def test_inventory_excludes_canonical_duplicates_of_known_rows():
    source = {"train": [{"id": "train:1", "text": "Known!", "label": LABELS[0]},
                         {"id": "train:2", "text": "Unseen", "label": LABELS[0]}],
              "test": [{"id": "test:1", "text": "known", "label": LABELS[0]}]}
    from src.data import text_hash
    counts, rows = unused_inventory(source, {text_hash("Known!")})
    assert counts[LABELS[0]]["remaining_total"] == 1
    assert [row["id"] for row in rows] == ["train:2"]


def test_forced_choice_scores_only_candidate_tokens_and_averages_length():
    import torch
    # prompt length two, candidate positions start at logit index one.
    logits = torch.zeros(2, 4, 5)
    ids = torch.tensor([[0, 0, 1, 2], [0, 0, 3, 0]])
    logits[0, 1, 1] = 2
    logits[0, 2, 2] = 2
    logits[1, 1, 3] = 2
    sums, means = label_log_scores(logits, ids, prompt_length=2, candidate_lengths=[2, 1])
    assert means[0] == pytest.approx(means[1])
    assert sums[0] == pytest.approx(2 * sums[1])
    logits[:, 0, :] = 100  # Prompt-only scores must not affect the answer.
    assert label_log_scores(logits, ids, 2, [2, 1])[1] == pytest.approx(means)


def test_position_bootstrap_keeps_rotations_paired_and_constant_classifier_at_null():
    from src.diagnostics.analysis import position_summary
    orders = [list(LABELS[i:] + LABELS[:i]) for i in range(8)]
    constant = [[LABELS[-1]] * 8 for _ in range(8)]
    result = position_summary(constant, list(LABELS), orders, bootstrap_resamples=20)
    assert result["final_option_rate"] == 0.125
    assert result["excess_over_valid_adjusted_reference_95_interval"] == [0.0, 0.0]
    always_final = [[order[-1]] * 8 for order in orders]
    assert position_summary(always_final, list(LABELS), orders, bootstrap_resamples=20)["final_option_rate"] == 1.0


def test_ledger_records_start_before_result_and_distinguishes_prompt_and_rule(tmp_path):
    import json
    from src.diagnostics.worker import AttemptLedger
    path = tmp_path / "attempts.jsonl"
    ledger = AttemptLedger(path)
    attempt = ledger.start("row:1", "auxiliary", "prompt-hash", 8, "rule-hash")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1 and records[0]["event"] == "start"
    assert records[0]["prompt_sha256"] == "prompt-hash"
    assert records[0]["scoring_rule_sha256"] == "rule-hash"
    ledger.finish(attempt, "error:RuntimeError", 0.1)
    assert len(path.read_text().splitlines()) == 2
