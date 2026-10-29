"""Fixed-label macro metrics; invalid predictions are never dropped."""

from collections import Counter
from src.task import LABELS


def diagnostic_metrics(predictions, truth, final_label=None):
    if not truth or len(predictions) != len(truth) or any(t not in LABELS for t in truth):
        raise ValueError("Expected equal nonempty arrays and valid ground truth")
    confusion = {t: {p: 0 for p in (*LABELS, "invalid")} for t in LABELS}
    for pred, target in zip(predictions, truth):
        confusion[target][pred if pred in LABELS else "invalid"] += 1
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        actual = sum(confusion[label].values())
        predicted = sum(confusion[t][label] for t in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": actual, "predicted": predicted, "true_positive": tp}
    count = len(truth)
    distribution = Counter(p if p in LABELS else "invalid" for p in predictions)
    correct = sum(confusion[label][label] for label in LABELS)
    return {
        "examples": count, "correct": correct, "accuracy": correct / count,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(LABELS),
        "per_class": per_class, "confusion": confusion,
        "invalid": distribution["invalid"], "invalid_rate": distribution["invalid"] / count,
        "predicted_distribution": {label: distribution[label] for label in (*LABELS, "invalid")},
        "maximum_predicted_label_fraction": max(distribution[label] for label in LABELS) / count,
        "final_listed_label": final_label,
        "final_option_count": distribution[final_label] if final_label else None,
        "final_option_rate": distribution[final_label] / count if final_label else None,
    }


def meets_quality_targets(metrics):
    return (metrics["accuracy"] >= 0.70 and metrics["macro_f1"] >= 0.65
            and all(v["recall"] > 0 for v in metrics["per_class"].values())
            and metrics["maximum_predicted_label_fraction"] <= 0.35)
