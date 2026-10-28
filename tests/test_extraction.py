"""Behavioral checks using synthetic examples; no model downloads."""

import json

import pytest

from src.data import canonical_text, freeze_splits, validate_disjoint, attacker_texts
from src.evaluation.metrics import classification_metrics, agreement
from src.task import LABELS, normalize_label
from src.teacher.client import BudgetedTeacher, BudgetExceeded
from src.attacker.training import response_only_example, TeacherPair


@pytest.mark.parametrize("label", LABELS)
def test_normalizes_exact_label_with_case_quotes_and_terminal_punctuation(label):
    assert normalize_label(' "' + label.upper() + '." \n') == label


@pytest.mark.parametrize("raw", ["", "positive", "card_arrival or change_pin", "The label is card_arrival", "not card_arrival", "card_arrival\nchange_pin"])
def test_rejects_ambiguous_explanatory_and_out_of_scope_outputs(raw):
    assert normalize_label(raw) is None


def synthetic_rows():
    return {
        split: [{"id": f"{split}:{label}:{i}", "text": f"{split} {label} sample {i}", "label": label}
                for label in LABELS for i in range(20)]
        for split in ["train", "test"]
    }


def test_seeded_stratified_splits_are_disjoint_and_repeatable():
    sizes = {"extraction": 17, "validation": 8, "legitimate": 9, "test": 16}
    a, audit = freeze_splits(synthetic_rows(), sizes, 17)
    b, _ = freeze_splits(synthetic_rows(), sizes, 17)
    assert a == b
    assert {key: len(value) for key, value in a.items()} == sizes
    validate_disjoint(a)
    for subset in a.values():
        counts = [sum(row["label"] == label for row in subset) for label in LABELS]
        assert min(counts) > 0 and max(counts) - min(counts) <= 1
    assert set(attacker_texts(a["extraction"])[0]) == {"id", "text"}
    assert all(row["id"].startswith("test:") for row in a["legitimate"])


def test_duplicate_text_even_with_punctuation_case_or_space_is_leakage():
    assert canonical_text('  CARD arrived! ') == canonical_text('card arrived')
    with pytest.raises(ValueError, match="leakage"):
        validate_disjoint({"a": [{"id": "1", "text": "Same text!"}], "b": [{"id": "2", "text": "same TEXT"}]})


def test_insufficient_unique_examples_stops_split_construction():
    with pytest.raises(ValueError, match="Insufficient"):
        freeze_splits(synthetic_rows(), {"extraction": 160, "validation": 8, "legitimate": 8, "test": 8}, 17)


def test_all_attempts_including_errors_and_invalid_outputs_consume_budget(tmp_path):
    outputs = iter(["card_arrival", "unparseable", RuntimeError("transport failed")])
    def backend(text):
        value = next(outputs)
        if isinstance(value, Exception):
            raise value
        return value
    teacher = BudgetedTeacher(backend, tmp_path / "queries.jsonl", {"extraction": 3, "test": 1})
    assert teacher.query("query one", "extraction") == "card_arrival"
    assert teacher.query("query two", "extraction") == "unparseable"
    with pytest.raises(RuntimeError, match="transport failed"):
        teacher.query("query three", "extraction")
    with pytest.raises(BudgetExceeded):
        teacher.query("not sent", "extraction")
    assert teacher.attempts["extraction"] == 3
    records = [json.loads(line) for line in (tmp_path / "queries.jsonl").read_text().splitlines()]
    assert [r["status"] for r in records] == ["valid", "invalid", "error"]
    assert all("prompt" not in row and "text" not in row for row in records)


def test_metrics_do_not_count_two_invalid_outputs_as_agreement():
    a, b = ["card_arrival", None, "change_pin"], ["card_arrival", None, "card_arrival"]
    assert agreement(a, b) == {"matches": 1, "examples": 3, "fraction": 1 / 3}
    metrics = classification_metrics(a, ["card_arrival", "change_pin", "change_pin"])
    assert metrics["correct"] == 2 and metrics["accuracy"] == 2 / 3
    assert metrics["invalid"] == 1
    assert sum(metrics["predicted_distribution"].values()) == 3
    with pytest.raises(ValueError):
        agreement(["change_pin"], [])


def test_distillation_record_has_no_ground_truth_channel():
    with pytest.raises(TypeError):
        TeacherPair(text="example", teacher_label="card_arrival", ground_truth="change_pin")


def test_response_loss_masks_prompt_and_supervises_response_plus_eos():
    class Tokenizer:
        eos_token_id = 99
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "PROMPT"
        def encode(self, text, add_special_tokens=False):
            return [10, 11, 12] if text == "PROMPT" else [20, 21]
    item = response_only_example(Tokenizer(), TeacherPair("query", "change_pin"), max_length=32)
    assert item["input_ids"] == [10, 11, 12, 20, 21, 99]
    assert item["labels"] == [-100, -100, -100, 20, 21, 99]
    with pytest.raises(ValueError, match="length"):
        response_only_example(Tokenizer(), TeacherPair("query", "change_pin"), max_length=5)


def test_real_causal_lm_loss_uses_only_shifted_response_targets():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(17)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        attn_implementation="eager",
    )).eval()
    ids = torch.tensor([[10, 11, 12, 20, 21, 99]])
    labels = torch.tensor([[-100, -100, -100, 20, 21, 99]])
    result = model(input_ids=ids, labels=labels)
    manual = torch.nn.functional.cross_entropy(result.logits[0, 2:5], ids[0, 3:6])
    torch.testing.assert_close(result.loss, manual)


def test_local_model_directory_is_repository_scoped_and_revision_specific():
    from pathlib import Path
    from src.models import local_model_path
    spec = {"model_id": "publisher/model", "revision": "a" * 40}
    path = local_model_path(spec)
    assert path.relative_to(Path(__file__).resolve().parents[1]).as_posix() == (
        "models/publisher--model/" + "a" * 40
    )
