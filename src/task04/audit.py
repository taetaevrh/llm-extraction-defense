"""Task04 rerun verification, kept separate from historical verification claims."""

import json
import math
import hashlib
from pathlib import Path

from src.task import LABELS
from src.task04.core import digest
from src.task04.metrics import validate_aggregate
from src.diagnostics.metrics import diagnostic_metrics
from src.task import normalize_label


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def same_confusion(first, second):
    return all(first.get(t, {}).get(p, 0) == second.get(t, {}).get(p, 0)
               for t in LABELS for p in (*LABELS, "invalid"))


def historical_audit(root):
    root = Path(root)
    task2 = json.loads((root / "results/banking77_pilot_summary.json").read_text())
    task3 = json.loads((root / "results/task03_results_summary.json").read_text())
    manifest = json.loads((root / "experiments/task03_diagnostic_manifest.json").read_text())
    records, unavailable = [], []
    def check_metrics(name, metrics):
        recomputed = validate_aggregate(metrics)
        records.append({"check": name, "status": "rerun_passed", "recomputed": recomputed})
    for subset, values in task2["evaluation"].items():
        for name in ("teacher", "student_before", "student_after"):
            check_metrics(f"Task02/{subset}/{name}", values[name])
    for role, values in task3["teacher_results"].items():
        for variant, metrics in values["variants"].items():
            check_metrics(f"Task03/{role}/{variant}", metrics)
    for role, values in task3["student_results"].items():
        for subset, methods in values["subsets"].items():
            for method in ("forced_choice", "historical_free_generation", "majority_comparator"):
                check_metrics(f"Task03/{role}/{subset}/{method}", methods[method])
    raw2 = root / manifest["config"]["prior_run"]
    raw3 = root / "experiments/runs" / task3["run_id"]
    for relative, expected in manifest["protected_Task01_Task02_files_sha256"].items():
        path = root / relative
        if path.exists():
            if digest(path) != expected:
                raise ValueError(f"Historical protected artifact changed: {relative}")
        else:
            unavailable.append({"artifact": relative, "blocks": "Historical protected hash comparison"})
    records.append({"check": "Task03/protected_Task01_Task02_hashes", "status": "rerun_passed", "recorded_files": len(manifest["protected_Task01_Task02_files_sha256"])})
    for relative, expected in manifest["frozen_files_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"Task03 frozen file changed: {relative}")
    records.append({"check": "Task03/frozen_configuration_prompt_validation_hashes", "status": "rerun_passed"})
    query_path = raw2 / "queries.jsonl"
    if query_path.exists():
        queries = read_jsonl(query_path)
        for phase, filename in (("extraction", "extraction_responses.jsonl"), ("test", "teacher_test.jsonl"), ("legitimate", "teacher_legitimate.jsonl")):
            path = raw2 / filename
            if not path.exists():
                unavailable.append({"artifact": str(path.relative_to(root)), "blocks": "Task02 input/output SHA comparison"})
                continue
            responses = read_jsonl(path)
            events = [r for r in queries if r["phase"] == phase]
            assert len(responses) == len(events)
            for event, response in zip(events, responses):
                assert hashlib.sha256(response["text"].encode()).hexdigest() == event["input_sha256"]
                assert hashlib.sha256(response["raw"].encode()).hexdigest() == event["output_sha256"]
                assert normalize_label(response["raw"]) == event["normalized_label"] == response["teacher_label"]
            records.append({"check": f"Task02/{phase}/raw_query_hashes_and_normalization", "status": "rerun_passed", "rows": len(responses)})
    for relative, expected_counts in (("queries.jsonl", {"extraction": 500, "test": 200, "legitimate": 100}),
                                     ("smoke/queries.jsonl", {"smoke": 8})):
        path = raw2 / relative
        if not path.exists():
            unavailable.append({"artifact": str(path.relative_to(root)), "blocks": "Task02 query accounting"})
            continue
        from collections import Counter
        entries = read_jsonl(path)
        counts = dict(Counter(e["phase"] for e in entries))
        if relative.startswith("smoke"):
            assert len(entries) == 8
        else:
            assert counts == expected_counts
        records.append({"check": f"Task02/{relative}/query_counts", "status": "rerun_passed", "counts": counts, "sha256": digest(path)})
    splits = json.loads((root / "docs/banking77_split_manifest.json").read_text())["subsets"]
    validation = json.loads((root / "docs/task03_validation_manifest.json").read_text())["rows"]
    for role, values in task3["teacher_results"].items():
        for variant, published in values["variants"].items():
            path = raw3 / role / f"{variant}.jsonl"
            if not path.exists():
                unavailable.append({"artifact": str(path.relative_to(root)), "blocks": "Task03 raw Teacher verification"})
                continue
            rows = read_jsonl(path)
            assert [r["id"] for r in rows] == [r["id"] for r in validation]
            assert [normalize_label(r["raw"]) for r in rows] == [r["label"] for r in rows]
            actual = diagnostic_metrics([r["label"] for r in rows], [r["label"] for r in validation])
            assert actual["confusion"] == published["confusion"]
            records.append({"check": f"Task03/{role}/{variant}/raw_predictions", "status": "rerun_passed", "sha256": digest(path)})
    for subset in ("test", "legitimate"):
        expected = [r["id"] for r in splits[subset]]
        path = raw2 / f"teacher_{subset}.jsonl"
        if not path.exists():
            unavailable.append({"artifact": str(path.relative_to(root)), "blocks": "Paired historical predictions/agreement verification"})
            continue
        teachers = read_jsonl(path)
        assert [r["id"] for r in teachers] == expected
        teacher = [r["teacher_label"] for r in teachers]
        assert [normalize_label(r["raw"]) for r in teachers] == teacher
        truth = [r["label"] for r in splits[subset]]
        assert same_confusion(diagnostic_metrics(teacher, truth)["confusion"], task2["evaluation"][subset]["teacher"]["confusion"])
        for phase, metric_name in (("before", "student_before"), ("after", "student_after")):
            path = raw2 / f"{phase}_{subset}.jsonl"
            if not path.exists():
                unavailable.append({"artifact": str(path.relative_to(root)), "blocks": f"Task02 {phase} raw verification"})
                continue
            preds = read_jsonl(path)
            assert [r["id"] for r in preds] == expected
            labels = [r["label"] for r in preds]
            assert [normalize_label(r["raw"]) for r in preds] == labels
            assert same_confusion(diagnostic_metrics(labels, truth)["confusion"], task2["evaluation"][subset][metric_name]["confusion"])
            matches = sum(p in LABELS and p == t for p, t in zip(labels, teacher))
            assert matches == task2["evaluation"][subset][f"agreement_{phase}"]["matches"]
            records.append({"check": f"Task02/{subset}/{phase}/raw_alignment_agreement", "status": "rerun_passed", "matches": matches, "sha256": digest(path)})
        for role in ("student_pretrained", "student_adapted"):
            path = raw3 / role / f"{subset}.jsonl"
            if not path.exists():
                unavailable.append({"artifact": str(path.relative_to(root)), "blocks": "Task03 auxiliary score verification"})
                continue
            predictions = read_jsonl(path)
            assert [r["id"] for r in predictions] == expected
            labels = []
            for prediction in predictions:
                means = prediction["selection_mean_token_log_probability"]
                for label in LABELS:
                    assert math.isclose(means[label], prediction["label_conditional_log_likelihood_sums"][label] / prediction["candidate_token_lengths"][label], abs_tol=1e-10)
                assert prediction["label"] == max(LABELS, key=lambda label: means[label])
                labels.append(prediction["label"])
            matches = sum(p == t for p, t in zip(labels, teacher))
            assert matches == task3["student_results"][role]["subsets"][subset]["forced_choice"]["teacher_agreement"]["matches"]
            records.append({"check": f"Task03/{role}/{subset}/raw_scores_and_alignment", "status": "rerun_passed", "sha256": digest(path), "matches": matches})
    for role, expected in (("small_teacher", 880), ("strong_teacher", 880), ("student_pretrained", 300), ("student_adapted", 300)):
        path = raw3 / role / "attempts.jsonl"
        if path.exists():
            entries = read_jsonl(path)
            starts = [e for e in entries if e["event"] == "start"]
            ends = [e for e in entries if e["event"] == "finish"]
            assert len(starts) == len(ends) == expected
            assert len({e["attempt"] for e in starts}) == expected
            assert {e["attempt"] for e in starts} == {e["attempt"] for e in ends}
            records.append({"check": f"Task03/{role}/query_count", "status": "rerun_passed", "starts": len(starts), "finishes": len(ends), "sha256": digest(path)})
        else:
            unavailable.append({"artifact": str(path.relative_to(root)), "blocks": "Historical inference-attempt verification"})
    for filename, expected in manifest["adapter_files_sha256"].items():
        path = raw2 / "student_adapter" / filename
        if path.exists():
            assert digest(path) == expected
        else:
            unavailable.append({"artifact": str(path.relative_to(root)), "blocks": "Saved adapter hash verification only; no adapter recreation"})
    records.append({"check": "Task03/recorded_adapter_file_hashes", "status": "rerun_passed" if all((raw2 / "student_adapter" / name).exists() for name in manifest["adapter_files_sha256"]) else "unavailable",
                    "recorded_files": len(manifest["adapter_files_sha256"])})
    return {"schema_version": 1, "status": "rerun_checks_passed" if not unavailable else "partially_unavailable",
        "rerun_checks": records, "unavailable": unavailable,
        "historical_only": ["Historical timing and GPU-memory samples are records, not reproduced by CPU verification.",
                            "No historical model outputs or missing adapters were regenerated.",
                            "Agreement is not derivable from individual gold-label confusion matrices; paired raw rows were checked where present."],
        "provenance_correction_respected": "Task03 Student ledger prompt_sha256 is the scoring-rule hash; exact prompt is separately frozen. No raw record rewritten."}


def observed_exclusions(root):
    """Exclude prior assignments plus every existing id/text in raw JSONL records."""
    root = Path(root)
    original = json.loads((root / "docs/banking77_split_manifest.json").read_text())["subsets"]
    excluded_ids = {r["id"] for rows in original.values() for r in rows}
    excluded_hashes = {r["text_sha256"] for rows in original.values() for r in rows}
    from src.data import text_hash
    provenance = []
    for path in sorted((root / "experiments/runs").glob("**/*.jsonl")):
        records = read_jsonl(path)  # malformed records block exclusion verification
        found = 0
        for row in records:
            if isinstance(row.get("id"), str):
                excluded_ids.add(row["id"])
                found += 1
            if isinstance(row.get("text"), str):
                excluded_hashes.add(text_hash(row["text"]))
        provenance.append({"path": path.relative_to(root).as_posix(), "sha256": digest(path), "id_records": found})
    return excluded_ids, excluded_hashes, provenance
