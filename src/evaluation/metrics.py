"""Metrics keep invalid outputs in denominators and outside label matches."""

from collections import Counter
from src.task import LABELS


def _check(a, b):
    if not a or len(a) != len(b):
        raise ValueError("Expected equal nonempty prediction/reference sequences")


def agreement(predictions, teacher):
    _check(predictions, teacher)
    matches = sum(p in LABELS and t in LABELS and p == t for p, t in zip(predictions, teacher))
    return {"matches": matches, "examples": len(predictions), "fraction": matches / len(predictions)}


def classification_metrics(predictions, truth):
    _check(predictions, truth)
    if not all(label in LABELS for label in truth):
        raise ValueError("Invalid ground-truth label")
    correct = sum(p in LABELS and p == t for p, t in zip(predictions, truth))
    counts = Counter(p if p in LABELS else "invalid" for p in predictions)
    return {
        "correct": correct, "examples": len(truth), "accuracy": correct / len(truth),
        "invalid": sum(p not in LABELS for p in predictions),
        "predicted_distribution": {label: counts[label] for label in (*LABELS, "invalid")},
        "ground_truth_distribution": {label: truth.count(label) for label in LABELS},
        "confusion": {label: dict(Counter(p if p in LABELS else "invalid" for p, t in zip(predictions, truth) if t == label)) for label in LABELS},
    }
