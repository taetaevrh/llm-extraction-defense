"""Descriptive Wilson intervals and stratified macro-F1 bootstrap."""

import math
from src.diagnostics.metrics import diagnostic_metrics, meets_quality_targets
from src.task import LABELS


def wilson(successes, trials):
    if trials <= 0:
        return None
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def validate_aggregate(metrics):
    matrix = metrics["confusion"]
    count = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix.get(label, {}).get(label, 0) for label in LABELS)
    invalid = sum(matrix.get(label, {}).get("invalid", 0) for label in LABELS)
    if count != metrics["examples"] or correct != metrics["correct"] or not math.isclose(correct / count, metrics["accuracy"], abs_tol=1e-12):
        raise ValueError("Confusion total/correct/accuracy mismatch")
    f1s = []
    for label in LABELS:
        support = sum(matrix.get(label, {}).values())
        predicted = sum(matrix.get(t, {}).get(label, 0) for t in LABELS)
        tp = matrix.get(label, {}).get(label, 0)
        f1s.append(2 * tp / (support + predicted) if support + predicted else 0.0)
    if "macro_f1" in metrics and not math.isclose(sum(f1s) / 8, metrics["macro_f1"], abs_tol=1e-12):
        raise ValueError("Confusion macro-F1 mismatch")
    if "invalid" in metrics and metrics["invalid"] != invalid:
        raise ValueError("Confusion invalid count mismatch")
    return {"examples": count, "correct": correct, "accuracy": correct / count, "macro_f1": sum(f1s) / 8, "invalid": invalid}


def metrics_with_uncertainty(predictions, truth, resamples=2000, seed=17):
    import numpy as np
    result = diagnostic_metrics(predictions, truth)
    result["accuracy_wilson_95"] = wilson(result["correct"], result["examples"])
    for label, values in result["per_class"].items():
        values["recall_wilson_95"] = wilson(values["true_positive"], values["support"])
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(np.array(truth) == label) for label in LABELS]
    if any(not len(group) for group in groups):
        raise ValueError("Stratified bootstrap requires every class")
    values = []
    for _ in range(resamples):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        values.append(diagnostic_metrics([predictions[i] for i in indices], [truth[i] for i in indices])["macro_f1"])
    result["macro_f1_stratified_bootstrap_95"] = np.percentile(values, [2.5, 97.5], method="linear").tolist()
    result["statistics"] = {"wilson_z": 1.959963984540054, "bootstrap_resamples": resamples, "bootstrap_seed": seed,
                            "bootstrap_rng": "NumPy default_rng/PCG64", "bootstrap_percentile_method": "linear",
                            "sampling_unit": "input row within true class", "simultaneous_coverage_claim": False}
    passed = meets_quality_targets(result)
    result["decision"] = {"passes_point_criteria": passed,
        "accuracy_ge_0_70": result["accuracy"] >= 0.70,
        "macro_f1_ge_0_65": result["macro_f1"] >= 0.65,
        "every_class_recall_positive": all(v["recall"] > 0 for v in result["per_class"].values()),
        "maximum_label_share_le_0_35": result["maximum_predicted_label_fraction"] <= 0.35,
        "population_threshold_confidence_claim": False}
    validate_aggregate(result)
    return result
