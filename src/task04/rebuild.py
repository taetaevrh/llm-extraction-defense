"""Offline reconstruction: predictions + frozen manifest, no model imports."""

import argparse
import hashlib
import json
from pathlib import Path
import yaml

from src.task import normalize_label
from src.task04.metrics import metrics_with_uncertainty
from src.task04.core import digest


def reconstruct(manifest, predictions, config):
    rows = manifest["rows"]
    if len(predictions) != manifest["expected_samples"] or [r["id"] for r in predictions] != [r["id"] for r in rows]:
        raise ValueError("Confirmation predictions incomplete or misaligned")
    normalized = [normalize_label(r["raw"]) for r in predictions]
    if normalized != [r["label"] for r in predictions]:
        raise ValueError("Recorded labels differ from frozen normalization")
    settings = config["statistics"]
    return metrics_with_uncertainty(normalized, [r["label"] for r in rows], settings["bootstrap_resamples"], settings["bootstrap_seed"])


def build_summary(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    for name, expected in manifest.get("raw_files_sha256", {}).items():
        if digest(run / name) != expected:
            raise ValueError(f"Local raw artifact hash mismatch: {name}")
    data = json.loads((run / "input_manifest.json").read_text())
    config = yaml.safe_load((run / "configuration.yaml").read_text())
    records = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()] if (run / "predictions.jsonl").exists() else []
    events = [json.loads(line) for line in (run / "attempts.jsonl").read_text().splitlines()] if (run / "attempts.jsonl").exists() else []
    counts = {phase: sum(e["event"] == "start" and e["phase"] == phase for e in events) for phase in ("smoke", "confirmation")}
    starts = [e for e in events if e["event"] == "start"]
    finishes = [e for e in events if e["event"] == "finish"]
    if counts["smoke"] > 8 or counts["confirmation"] > 88 or len(starts) > 96:
        raise ValueError("Recorded attempt budget exceeded")
    if len({e["attempt"] for e in starts}) != len(starts) or len({e["attempt"] for e in finishes}) != len(finishes):
        raise ValueError("Duplicate attempt identity")
    if manifest["status"] == "completed" and (counts != {"smoke": 8, "confirmation": 88} or {e["attempt"] for e in starts} != {e["attempt"] for e in finishes}):
        raise ValueError("Completed run has missing attempts/finishes")
    metric = reconstruct(data, records, config) if manifest["status"] == "completed" else None
    return {"schema_version": 1, "run_id": manifest["run_id"], "status": manifest["status"],
        "metrics": metric, "unmeasured_reason": None if metric is not None else manifest.get("failure", "Incomplete confirmation"),
        "attempts": counts, "total_teacher_attempts": sum(counts.values()), "completed_confirmation_records": len(records),
        "source_composition": data["source_composition"], "remaining_inventory_after_run": manifest.get("remaining_inventory_after_run"),
        "resources": manifest.get("resources"), "wall_seconds": manifest.get("wall_seconds"),
        "resource_preflight": manifest.get("resource_preflight"),
        "executed_source_commit": manifest["executed_source_commit"], "frozen_hashes": manifest["frozen_hashes"],
        "environment": manifest.get("environment"), "model_verification": manifest.get("model_verification"),
        "failures": manifest.get("failures", []),
        "prediction_file_sha256": hashlib.sha256((run / "predictions.jsonl").read_bytes()).hexdigest() if (run / "predictions.jsonl").exists() else None,
        "training_steps": 0, "defense_evaluations": 0,
        "interpretation": "Limited pipeline-untouched confirmation; not proof of pretraining absence, a final defense test, or 95% population threshold attainment"}


def render_report(summary):
    m = summary["metrics"]
    lines = ["# Task 04 frozen Teacher confirmation", "", f"Run: `{summary['run_id']}`. Status: **{summary['status']}**.", "",
             f"Executed source: `{summary['executed_source_commit']}`.", ""]
    if m is None:
        lines += ["Metrics: null.", f"Reason: {summary['unmeasured_reason']}"]
    else:
        pct = lambda x: f"{100*x:.2f}%"
        ci = lambda values: "–".join(pct(v) for v in values)
        lines += [f"Accuracy: **{m['correct']}/{m['examples']} = {pct(m['accuracy'])}**, descriptive 95% Wilson interval **{ci(m['accuracy_wilson_95'])}**.",
            f"Macro-F1: **{m['macro_f1']:.4f}**, class-stratified 95% bootstrap interval **{m['macro_f1_stratified_bootstrap_95'][0]:.4f}–{m['macro_f1_stratified_bootstrap_95'][1]:.4f}** (2,000 resamples, seed 17).",
            f"Invalids: {m['invalid']}/{m['examples']}; maximum predicted-label share: {pct(m['maximum_predicted_label_fraction'])}.",
            f"Point-estimate criteria: **{'PASS' if m['decision']['passes_point_criteria'] else 'FAIL'}**. Passing does not establish that population performance exceeds the thresholds with 95% confidence.", "",
            "| Intent | Precision | Recall | Recall Wilson 95% | F1 | Support | Predicted |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
        for label, v in m["per_class"].items():
            lines.append(f"| {label} | {v['precision']:.4f} | {v['recall']:.4f} | {ci(v['recall_wilson_95'])} | {v['f1']:.4f} | {v['support']} | {v['predicted']} |")
        lines += ["", "Confusion matrix (rows: true class; columns: predicted class, plus invalid):", "", "```json", json.dumps(m["confusion"], indent=2), "```"]
    lines += ["", f"Attempts: {json.dumps(summary['attempts'])}; total {summary['total_teacher_attempts']}. Invalid/error attempts consume budget; no replacement or retry.",
        f"Wall time (seconds): {summary['wall_seconds']}.", "", "Resource records:", "", "```json", json.dumps(summary["resources"], indent=2), "```", "",
        "Official source composition by class:", "", "```json", json.dumps(summary["source_composition"], indent=2), "```", "",
        "Unused inventory after this run:", "", "```json", json.dumps(summary["remaining_inventory_after_run"], indent=2), "```", "",
        "These rows were unused by this research pipeline at freezing, not proven absent from pretraining. The mixed official train/test source, canonical rather than semantic deduplication, eleven rows per class, and one frozen configuration limit inference. Attempted rows are observed; unqueried reserved rows remain unobserved. A blocked run supplies no confirmation evidence. Reserved/observed rows cannot silently become an untouched future model-selection test.",
        "The 79/88 Task03 result was a selection-set score. Small-Teacher order sensitivity does not establish a general last-option explanation. Historical Student forced-choice accuracy rose 19% to 35% and generation agreement to 82%; majority/error matches overlap and do not identify causal shares. No defense has been evaluated.", "",
        "Offline rebuild (no model loading): `python -m src.task04.rebuild --run experiments/runs/" + summary["run_id"] + " --output .cache/task04_rebuilt/" + summary["run_id"] + "`.",
        "See task04_reproducibility_audit.md for checks rerun here, historical-only claims, hashes, raw-record locations and missing-artifact boundaries. No Student training or defense work followed this run."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    summary = build_summary(args.run)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(f"Offline summary/report rebuilt at {output}")


if __name__ == "__main__":
    main()
