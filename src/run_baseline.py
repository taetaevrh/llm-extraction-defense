"""Smoke-gated single-seed no-defense pilot. Run as a module from repo root."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import yaml

from src.attacker.training import TeacherPair, train_student
from src.data import attacker_texts, freeze_splits, load_banking77, split_manifest
from src.evaluation.metrics import agreement, classification_metrics
from src.preflight import ROOT, configure_paths, verify_cuda
from src.runtime import Phase, release_gpu, seed_everything, write_json
from src.task import LABELS, PROMPT_TEMPLATE, normalize_label
from src.teacher.client import BudgetedTeacher, TeacherProcess


class Tee:
    def __init__(self, original, record):
        self.original, self.record = original, record
        self.encoding = "utf-8"

    def write(self, value):
        self.record.write(value)
        self.record.flush()
        return self.original.write(value)

    def flush(self):
        self.original.flush()
        self.record.flush()

    def isatty(self):
        return False


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare_models(config):
    from huggingface_hub import snapshot_download
    files = {}
    for role in ("teacher", "student"):
        spec = config[role]
        directory = Path(snapshot_download(
            spec["model_id"], revision=spec["revision"],
            allow_patterns=["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "model.safetensors", "model.safetensors.index.json", "model-*.safetensors", "LICENSE", "README.md"],
        ))
        files[role] = {str(p.relative_to(directory)): {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(directory.rglob("*")) if p.is_file()}
    return files


def predict_rows(model, tokenizer, rows, config, path):
    from src.models import predict
    predictions = []
    with Path(path).open("x", encoding="utf-8") as handle:
        for index, row in enumerate(attacker_texts(rows)):
            started = time.perf_counter()
            raw = predict(model, tokenizer, row["text"], config["decoding"])
            record = {"id": row["id"], "raw": raw, "label": normalize_label(raw), "latency_seconds": time.perf_counter() - started}
            predictions.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if (index + 1) % 100 == 0:
                print(f"predicted {index + 1}/{len(rows)}", flush=True)
    return predictions


def collect(teacher, rows, phase, path):
    responses = []
    with Path(path).open("x", encoding="utf-8") as handle:
        for index, row in enumerate(attacker_texts(rows)):
            raw = teacher.query(row["text"], phase)
            result = {"id": row["id"], "text": row["text"], "raw": raw, "teacher_label": normalize_label(raw)}
            responses.append(result)
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            if (index + 1) % 100 == 0:
                print(f"teacher {phase} {index + 1}/{len(rows)}", flush=True)
    return responses


def pairs_from(responses):
    return [TeacherPair(text=row["text"], teacher_label=row["teacher_label"]) for row in responses if row["teacher_label"] in LABELS]


def check_headroom(config):
    import torch
    free, _ = torch.cuda.mem_get_info()
    driver_free = float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0])
    remaining = min(free / 2**20, driver_free)
    if remaining < config["resources"]["minimum_remaining_vram_mib"]:
        raise RuntimeError("Insufficient remaining GPU memory headroom")
    return remaining


def run_smoke(config, config_path, subsets, run_dir):
    from src.models import load_model
    smoke_dir = run_dir / "smoke"
    smoke_dir.mkdir()
    # Trusted curator selects one example per class; labels are then stripped.
    smoke_rows = [next(row for row in subsets["validation"] if row["label"] == label) for label in LABELS]
    resources = {}
    process = TeacherProcess(config_path, smoke_dir / "teacher_resources.json", smoke_dir / "teacher.stderr.log")
    teacher = BudgetedTeacher(process, smoke_dir / "queries.jsonl", {"smoke": config["smoke"]["teacher_queries"]})
    try:
        responses = collect(teacher, smoke_rows, "smoke", smoke_dir / "responses.jsonl")
    finally:
        process.close()
    pairs = pairs_from(responses)
    if len(pairs) < config["smoke"]["minimum_valid_teacher_labels"]:
        raise RuntimeError("Smoke teacher did not produce sufficient valid outputs")
    release_gpu()
    seed_everything(config["seed"])
    with Phase("student_smoke_load", resources):
        model, tokenizer = load_model(config["student"])
    with Phase("student_smoke_before", resources):
        before = predict_rows(model, tokenizer, smoke_rows, config, smoke_dir / "before.jsonl")
    with Phase("student_smoke_training", resources):
        seed_everything(config["seed"])
        model, training = train_student(model, tokenizer, pairs, config["training"], config["seed"], smoke_dir / "loss.jsonl", smoke_steps=config["smoke"]["student_optimizer_steps"])
        free_after_train = check_headroom(config)
    with Phase("student_smoke_after", resources):
        after = predict_rows(model, tokenizer, smoke_rows, config, smoke_dir / "after.jsonl")
    valid_after = sum(row["label"] is not None for row in after)
    result = {
        "status": "passed" if valid_after >= config["smoke"]["minimum_valid_student_outputs_after_training"] else "failed",
        "teacher_queries": dict(teacher.attempts), "valid_teacher_labels": len(pairs),
        "valid_student_outputs_before": sum(row["label"] is not None for row in before),
        "valid_student_outputs_after": valid_after,
        "training": training, "resources": resources,
        "teacher_resources": json.loads((smoke_dir / "teacher_resources.json").read_text()),
        "free_vram_mib_after_training": free_after_train,
        "adapter_discarded_before_pilot": True,
    }
    write_json(smoke_dir / "summary.json", result)
    del model, tokenizer
    release_gpu()
    if result["status"] != "passed":
        raise RuntimeError("Smoke training pipeline could not produce a valid label")
    return result


def evaluate(rows, references, before, after):
    truth = [row["label"] for row in rows]
    teacher = [row["teacher_label"] for row in references]
    initial = [row["label"] for row in before]
    extracted = [row["label"] for row in after]
    if not ([r["id"] for r in rows] == [r["id"] for r in references] == [r["id"] for r in before] == [r["id"] for r in after]):
        raise ValueError("Evaluation row alignment failed")
    return {
        "teacher": classification_metrics(teacher, truth),
        "student_before": classification_metrics(initial, truth),
        "student_after": classification_metrics(extracted, truth),
        "agreement_before": agreement(initial, teacher),
        "agreement_after": agreement(extracted, teacher),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    configure_paths()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["condition"] != "no_defense" or tuple(config["task"]["labels"]) != LABELS:
        raise ValueError("Only the approved no-defense, eight-label pilot is supported")
    provenance = json.loads((ROOT / config["dataset"]["license_provenance"]).read_text())
    if provenance["revision"] != config["dataset"]["revision"] or provenance["additional_conflicting_restrictions_found"]:
        raise ValueError("Dataset license gate not satisfied")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_banking77_seed17_" + uuid.uuid4().hex[:8]
    run_dir = ROOT / "experiments/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    result_dir = ROOT / "results/runs" / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    sys.stdout = Tee(sys.stdout, (run_dir / "stdout.log").open("x", encoding="utf-8"))
    sys.stderr = Tee(sys.stderr, (run_dir / "stderr.log").open("x", encoding="utf-8"))
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema_version": 1, "run_id": run_id, "status": "started", "stage": "cuda_preflight",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "seed": config["seed"],
        "config": config, "config_sha256": sha256_file(config_path),
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
        "normalization": "Whole canonical label only; trim whitespace, paired/unpaired quotes/backticks, case, and one final period/exclamation. Explanations/ambiguous text invalid.",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "code_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
        "command": [sys.executable, "-m", "src.run_baseline", "--config", args.config],
        "provenance": provenance, "failures": [],
        "environment_files_sha256": {name: sha256_file(ROOT / name) for name in ["environment.yml", "docs/banking77_environment.resolved.yml", "docs/banking77_conda_win64.explicit.txt", "docs/banking77_pip_freeze.txt"]},
    }
    resources = {}
    overall_started = time.perf_counter()
    write_json(manifest_path, manifest)
    shutil.copy2(config_path, run_dir / "config.resolved.yaml")
    (run_dir / "pip.freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True), encoding="utf-8")
    print(f"RUN_ID={run_id}", flush=True)
    print(f"Artifacts: {run_dir}", flush=True)
    try:
        cuda = verify_cuda()
        write_json(run_dir / "cuda.json", cuda)
        if min(cuda["free_vram_mib_before_computation"], cuda["nvidia_free_vram_mib_after_computation"]) < config["resources"]["minimum_free_vram_mib"]:
            raise RuntimeError("Insufficient GPU memory before model loading")
        manifest["stage"] = "dataset_preparation"
        write_json(manifest_path, manifest)
        source = load_banking77(config["dataset"]["revision"])
        subsets, audit = freeze_splits(source, config["sizes"], config["seed"])
        split_record = split_manifest(subsets, audit, config["dataset"]["revision"], config["seed"])
        write_json(run_dir / "split_manifest.json", split_record)
        manifest["split_manifest_sha256"] = sha256_file(run_dir / "split_manifest.json")
        if manifest["split_manifest_sha256"] != config["dataset"]["split_manifest_sha256"]:
            raise ValueError("Prepared splits differ from the pre-frozen manifest")
        manifest["stage"] = "model_preparation"
        write_json(manifest_path, manifest)
        manifest["model_files"] = prepare_models(config)
        manifest["preparation_seconds"] = time.perf_counter() - overall_started
        manifest["stage"] = "smoke"
        write_json(manifest_path, manifest)
        smoke_started = time.perf_counter()
        smoke = run_smoke(config, config_path, subsets, run_dir)
        manifest["smoke_seconds"] = time.perf_counter() - smoke_started
        manifest["smoke_passed"] = True
        manifest["stage"] = "pilot_teacher_collection"
        write_json(manifest_path, manifest)
        print("Smoke passed; starting the approved 500-query pilot.", flush=True)
        pilot_started = time.perf_counter()
        with Phase("teacher_collection_controller", resources):
            process = TeacherProcess(config_path, run_dir / "teacher_resources.json", run_dir / "teacher.stderr.log")
            teacher = BudgetedTeacher(process, run_dir / "queries.jsonl", {"extraction": 500, "test": 200, "legitimate": 100})
            try:
                responses = collect(teacher, subsets["extraction"], "extraction", run_dir / "extraction_responses.jsonl")
                references = {name: collect(teacher, subsets[name], name, run_dir / f"teacher_{name}.jsonl") for name in ("test", "legitimate")}
            finally:
                process.close()
        pairs = pairs_from(responses)
        # The training learner receives only these two-field records. The full
        # dataset, truth, test references, and teacher process never enter it.
        with (run_dir / "attacker_training.jsonl").open("x", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps({"text": pair.text, "teacher_label": pair.teacher_label}) + "\n")
        release_gpu()
        from src.models import load_model
        seed_everything(config["seed"])
        manifest["stage"] = "student_before"
        write_json(manifest_path, manifest)
        with Phase("student_load", resources):
            model, tokenizer = load_model(config["student"])
        with Phase("student_before_evaluation", resources):
            before = {name: predict_rows(model, tokenizer, subsets[name], config, run_dir / f"before_{name}.jsonl") for name in ("test", "legitimate")}
        manifest["stage"] = "student_training"
        write_json(manifest_path, manifest)
        with Phase("student_training", resources):
            seed_everything(config["seed"])
            model, training = train_student(model, tokenizer, pairs, config["training"], config["seed"], run_dir / "training_loss.jsonl")
            check_headroom(config)
        model.save_pretrained(run_dir / "student_adapter", safe_serialization=True)
        tokenizer.save_pretrained(run_dir / "student_adapter")
        manifest["stage"] = "student_after"
        write_json(manifest_path, manifest)
        with Phase("student_after_evaluation", resources):
            after = {name: predict_rows(model, tokenizer, subsets[name], config, run_dir / f"after_{name}.jsonl") for name in ("test", "legitimate")}
        teacher_resources = json.loads((run_dir / "teacher_resources.json").read_text())
        evaluation = {name: evaluate(subsets[name], references[name], before[name], after[name]) for name in ("test", "legitimate")}
        import numpy as np
        legitimate_records = [r for r in teacher.records if r["phase"] == "legitimate"]
        valid_teacher = evaluation["legitimate"]["teacher"]["examples"] - evaluation["legitimate"]["teacher"]["invalid"]
        evaluation["legitimate"]["teacher_user_utility"] = {
            "end_to_end_accuracy": evaluation["legitimate"]["teacher"]["accuracy"],
            "coverage": valid_teacher / len(subsets["legitimate"]), "denial_rate": 0.0,
            "served_valid_accuracy": evaluation["legitimate"]["teacher"]["correct"] / valid_teacher if valid_teacher else None,
            "latency_p50_seconds": float(np.percentile([r["latency_seconds"] for r in legitimate_records], 50)),
            "latency_p95_seconds": float(np.percentile([r["latency_seconds"] for r in legitimate_records], 95)),
        }
        result = {
            "schema_version": 1, "run_id": run_id, "status": "completed", "seed": config["seed"],
            "evaluation": evaluation, "training": training,
            "extraction": {"attempts": teacher.attempts["extraction"], "successful_training_labels": len(pairs),
                           "invalid": len(responses) - len(pairs), "teacher_label_distribution": dict(Counter(r["teacher_label"] or "invalid" for r in responses))},
            "teacher_queries": {**dict(teacher.attempts), "smoke": smoke["teacher_queries"]["smoke"], "total_including_smoke": sum(teacher.attempts.values()) + sum(smoke["teacher_queries"].values())},
            "resources": resources, "teacher_worker_resources": teacher_resources,
            "pilot_wall_seconds": time.perf_counter() - pilot_started,
            "preparation_seconds": manifest["preparation_seconds"], "smoke_seconds": manifest["smoke_seconds"],
            "total_wall_seconds_including_preparation_and_smoke": time.perf_counter() - overall_started,
            "cuda": cuda, "failures": [],
            "scope": "Eight-intent task-specific behavioral extraction; not full-LLM/weight recovery",
        }
        write_json(result_dir / "metrics.json", result)
        manifest.update({"status": "completed", "stage": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat(), "results_path": str((result_dir / "metrics.json").relative_to(ROOT))})
        print(json.dumps({"run_id": run_id, "test_agreement_before": evaluation["test"]["agreement_before"], "test_agreement_after": evaluation["test"]["agreement_after"], "teacher_queries": result["teacher_queries"]}), flush=True)
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error), "stage": manifest["stage"], "traceback": traceback.format_exc()}
        manifest.update({"status": "failed", "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        manifest["failures"].append(failure)
        write_json(run_dir / "failure.json", failure)
        write_json(result_dir / "failure.json", failure)
        raise
    finally:
        write_json(run_dir / "controller_resources.json", resources)
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
