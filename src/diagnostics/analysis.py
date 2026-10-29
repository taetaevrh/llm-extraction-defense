"""Predefined paired position summaries from saved outputs; no inference."""

from src.task import LABELS


def position_summary(predictions, truth, orders, bootstrap_resamples=2000, seed=17):
    """Arrays are eight rotations by N inputs, retaining within-input pairing."""
    import numpy as np
    pred = np.array([[p if p in LABELS else "invalid" for p in row] for row in predictions])
    if pred.shape != (8, len(truth)) or any(len(order) != 8 for order in orders):
        raise ValueError("Expected eight complete rotations on the same examples")
    final = pred == np.array([order[-1] for order in orders])[:, None]
    valid = pred != "invalid"
    reference = float(valid.mean() / 8)
    rng = np.random.default_rng(seed)
    grouped = [np.flatnonzero(np.asarray(truth) == label) for label in LABELS]
    if any(not len(group) for group in grouped):
        raise ValueError("All eight true classes required")
    boot_rate, boot_excess = [], []
    for _ in range(bootstrap_resamples):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in grouped])
        value = float(final[:, indices].mean())
        boot_rate.append(value)
        boot_excess.append(value - float(valid[:, indices].mean() / 8))
    matrix = {}
    for label in LABELS:
        counts = [0] * 8
        for index, order in enumerate(orders):
            counts[order.index(label)] = int((pred[index] == label).sum())
        matrix[label] = counts
    return {
        "examples": len(truth), "rotation_responses": int(pred.size),
        "final_option_count": int(final.sum()), "final_option_rate": float(final.mean()),
        "valid_fraction": float(valid.mean()),
        "order_invariant_reference_rate": reference,
        "raw_uniform_position_reference": 0.125,
        "final_rate_95_percentile_interval": np.percentile(boot_rate, [2.5, 97.5]).tolist(),
        "excess_over_valid_adjusted_reference_95_interval": np.percentile(boot_excess, [2.5, 97.5]).tolist(),
        "method": f"{bootstrap_resamples} seed-{seed} bootstrap resamples of input rows within true class; keep eight paired rotations together",
        "selection_counts_by_label_position_zero_based": matrix,
        "changed_prediction_counts_vs_rotation_0": {f"rotation_{i}": int((pred[i] != pred[0]).sum()) for i in range(8)},
    }
