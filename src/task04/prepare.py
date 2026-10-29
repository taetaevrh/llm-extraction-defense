"""CPU-only pre-registration, availability audit, and deterministic input freeze."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import uuid
import yaml

from src.data import load_banking77, text_hash
from src.models import local_model_path
from src.preflight import ROOT, configure_paths
from src.runtime import write_json
from src.task import LABELS
from src.task04.audit import historical_audit, observed_exclusions
from src.task04.core import PROMPT_SHA256, canonical_json_hash, digest, select_confirmation, validate_inputs


def main():
    configure_paths()
    for path in ("docs/task04_confirmation_manifest.json", "configs/task04_confirmation.yaml"):
        if (ROOT / path).exists():
            raise FileExistsError("Task04 freeze already exists; do not replace prior manifest/configuration")
    report = historical_audit(ROOT)
    report["rerun_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(ROOT / "docs/task04_historical_audit.json", report)
    prior = json.loads((ROOT / "experiments/task03_diagnostic_manifest.json").read_text())
    variant = next(v for v in json.loads((ROOT / "configs/task03_prompts.json").read_text())["variants"] if v["id"] == "definitions")
    if hashlib.sha256(variant["template"].encode()).hexdigest() != PROMPT_SHA256 or variant["sha256"] != PROMPT_SHA256:
        raise ValueError("STOP: selected prompt hash mismatch")
    spec = prior["config"]["strong_teacher"]
    if spec["revision"] != "989aa7980e4cf806f80c7fef2b1adb7bc71aa306" or variant["max_input_tokens"] != 512:
        raise ValueError("STOP: selected Teacher/cap mismatch")
    if prior["config"]["decoding"] != {"do_sample": False, "num_beams": 1, "max_new_tokens": 12, "dtype": "bfloat16", "attention": "eager"}:
        raise ValueError("STOP: historical decoding mismatch")
    model_dir = local_model_path(spec)
    files = {}
    for name, expected in prior["stronger_model_artifacts"]["files"].items():
        path = model_dir / name
        if not path.exists() or digest(path) != expected["sha256"]:
            raise ValueError(f"STOP: incompatible/missing model or tokenizer file {name}")
        files[name] = expected
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    dataset_revision = prior["config"]["dataset_revision"]
    source = load_banking77(dataset_revision)
    all_rows = [r for rows in source.values() for r in rows]
    ids, hashes, raw_files = observed_exclusions(ROOT)
    selected = select_confirmation(all_rows, ids, hashes)
    composition = {label: dict(Counter(r["id"].split(":")[0] for r in selected if r["label"] == label)) for label in LABELS}
    statistics = {"accuracy_interval": "descriptive_two_sided_Wilson_95", "recall_intervals": "descriptive_two_sided_Wilson_95_not_simultaneous",
                  "bootstrap_resamples": 2000, "bootstrap_seed": 17, "macro_f1_interval": "within_true_class_percentile_2.5_97.5_linear",
                  "rng": "NumPy_default_rng_PCG64"}
    decision = {"accuracy_min": 0.70, "macro_f1_min": 0.65, "all_recalls_positive": True,
                "maximum_predicted_label_share": 0.35, "confidence_threshold_guarantee": False}
    frozen_rows = []
    for row in selected:
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": variant["template"].format(text=row["text"])}], tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(rendered, add_special_tokens=False)
        if len(tokens) > 512:
            raise ValueError("STOP: confirmation input would require truncation")
        frozen_rows.append({"id": row["id"], "label": row["label"], "canonical_sha256": text_hash(row["text"]),
                            "rendered_input_sha256": hashlib.sha256(rendered.encode()).hexdigest(), "input_tokens": len(tokens)})
    validate_inputs(selected, frozen_rows)
    dev = json.loads((ROOT / "docs/task03_validation_manifest.json").read_text())["rows"]
    by_id = {r["id"]: r for r in all_rows}
    smoke = [by_id[next(r["id"] for r in dev if r["label"] == label)] for label in LABELS]
    manifest = {"schema_version": 1, "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_id": "PolyAI/banking77",
        "dataset_revision": dataset_revision, "expected_samples": 88, "per_class": 11, "seed": 17,
        "sampling": "Group in approved label order; lexicographic source:id sort, canonical dedup excluding prior IDs/hashes, random.Random(17) shuffle each pool, first 11; final shuffle with same RNG",
        "rows": frozen_rows, "source_composition": composition,
        "smoke_observed_development_ids": [r["id"] for r in smoke],
        "prior_excluded_id_count": len(ids), "prior_excluded_canonical_hash_count": len(hashes),
        "prior_raw_ledger_files": raw_files, "statistics_sha256": canonical_json_hash(statistics), "decision_sha256": canonical_json_hash(decision),
        "model_files": files, "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "prompt_template_sha256": PROMPT_SHA256,
        "meaning_of_untouched": "Unused by research pipeline; model-pretraining absence and paraphrase disjointness unproven"}
    write_json(ROOT / "docs/task04_confirmation_manifest.json", manifest)
    config = {"schema_version": 1, "task": "one_frozen_teacher_confirmation", "seed": 17,
        "teacher": {"model_id": spec["model_id"], "revision": spec["revision"], "tokenizer_revision": spec["revision"]},
        "dataset_revision": dataset_revision, "input_manifest": "docs/task04_confirmation_manifest.json",
        "input_manifest_sha256": digest(ROOT / "docs/task04_confirmation_manifest.json"),
        "prompt_source": "configs/task03_prompts.json", "prompt_id": "definitions", "prompt_template_sha256": PROMPT_SHA256,
        "normalization_source_sha256": digest(ROOT / "src/task.py"),
        "decoding": {"do_sample": False, "num_beams": 1, "max_input_tokens": 512, "max_new_tokens": 12, "dtype": "bfloat16", "attention": "eager"},
        "budgets": {"smoke": 8, "confirmation": 88, "total": 96}, "minimum_driver_free_mib": 1024,
        "smoke_gate": "Only operational compatibility, cap/integrity, and memory; no accuracy-based selection or replacement of invalids",
        "statistics": statistics, "decision": decision,
        "no_training": True, "no_defense": True, "no_prompt_or_model_search": True}
    (ROOT / "configs/task04_confirmation.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    prep_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_task04_preparation_" + uuid.uuid4().hex[:8]
    directory = ROOT / "experiments/runs" / prep_id
    directory.mkdir(exist_ok=False)
    write_json(directory / "frozen_input_texts.json", selected)
    write_json(directory / "smoke_texts.json", smoke)
    print(json.dumps({"preparation_id": prep_id, "historical_checks": len(report["rerun_checks"]), "unavailable": report["unavailable"],
                      "source_composition": composition, "manifest_sha256": config["input_manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
