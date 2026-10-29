"""One-shot frozen Teacher confirmation; no Student or defense execution."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import uuid
import yaml

from src.data import load_banking77, text_hash
from src.diagnostics.resources import GuardedPhase, driver_memory
from src.diagnostics.worker import generate
from src.models import load_model, local_model_path
from src.preflight import ROOT, configure_paths, verify_cuda
from src.runtime import write_json, seed_everything
from src.task import LABELS, normalize_label
from src.task04.audit import observed_exclusions
from src.task04.core import AttemptBudget, PROMPT_SHA256, canonical_json_hash, digest, validate_inputs
from src.task04.rebuild import build_summary, render_report


def git(*args):
    return subprocess.check_output(["git", *args], text=True, cwd=ROOT).strip()


def remaining_inventory(source, ids, hashes, reserved):
    counts = {label: Counter() for label in LABELS}
    unassigned = {label: 0 for label in LABELS}
    seen = set(hashes)
    for row in sorted(source, key=lambda row: row["id"]):
        fingerprint = text_hash(row["text"])
        if row["label"] in LABELS and row["id"] not in ids and fingerprint not in seen:
            counts[row["label"]][row["id"].split(":")[0]] += 1
            unassigned[row["label"]] += int(row["id"] not in reserved)
            seen.add(fingerprint)
    return {"pipeline_unused_by_class_and_source": {k: dict(v) for k, v in counts.items()},
            "pipeline_unused_total": sum(sum(v.values()) for v in counts.values()),
            "unassigned_after_confirmation_reservation_by_class": unassigned,
            "unassigned_after_confirmation_reservation_total": sum(unassigned.values()),
            "balanced_pipeline_unused_maximum": 8 * min(sum(v.values()) for v in counts.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", required=True, help="Existing local Task04 preparation directory")
    args = parser.parse_args()
    configure_paths()
    if git("branch", "--show-current") != "main" or git("status", "--porcelain"):
        raise RuntimeError("STOP: execution requires committed clean main")
    for path in ("docs/task04_protocol.md", "configs/task04_confirmation.yaml", "src/task04/run.py"):
        git("ls-files", "--error-unmatch", path)
    source_commit = git("rev-parse", "HEAD")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_task04_confirmation_" + uuid.uuid4().hex[:8]
    run = ROOT / "experiments/runs" / run_id
    # Persistent one-shot claim survives failures before the first model call.
    with (ROOT / "experiments/runs/task04_confirmation_claim.json").open("x", encoding="utf-8") as claim:
        json.dump({"run_id": run_id, "source_commit": source_commit}, claim)
    run.mkdir(exist_ok=False)
    config_path = ROOT / "configs/task04_confirmation.yaml"
    config = yaml.safe_load(config_path.read_text())
    input_path = ROOT / config["input_manifest"]
    frozen = json.loads(input_path.read_text())
    shutil.copyfile(config_path, run / "configuration.yaml")
    shutil.copyfile(input_path, run / "input_manifest.json")
    manifest = {"schema_version": 1, "run_id": run_id, "status": "running", "executed_source_commit": source_commit,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "failures": [],
        "frozen_hashes": {"configuration_sha256": digest(config_path), "input_manifest_sha256": digest(input_path),
                          "protocol_sha256": digest(ROOT / "docs/task04_protocol.md"), "prompt_template_sha256": PROMPT_SHA256}}
    write_json(run / "manifest.json", manifest)
    ledger = AttemptBudget(run / "attempts.jsonl")
    resources = {}
    started = time.perf_counter()
    all_rows, rows = [], []
    model = None
    try:
        if digest(input_path) != config["input_manifest_sha256"] or digest(ROOT / "src/task.py") != config["normalization_source_sha256"]:
            raise ValueError("Frozen manifest/normalizer hash mismatch")
        if canonical_json_hash(config["statistics"]) != frozen["statistics_sha256"] or canonical_json_hash(config["decision"]) != frozen["decision_sha256"]:
            raise ValueError("Frozen statistics/decision mismatch")
        prior = json.loads((ROOT / "experiments/task03_diagnostic_manifest.json").read_text())
        spec = prior["config"]["strong_teacher"]
        if config["teacher"]["revision"] != spec["revision"] or config["teacher"]["model_id"] != spec["model_id"]:
            raise ValueError("Frozen model mismatch")
        directory = local_model_path(spec)
        for name, expected in frozen["model_files"].items():
            if digest(directory / name) != expected["sha256"]:
                raise ValueError(f"Pinned artifact mismatch: {name}")
        manifest["model_verification"] = {"status": "all_pinned_files_rehashed", "files": frozen["model_files"], "teacher": config["teacher"]}
        # Inventory checks precede CUDA/model work, so blocked runs can still report data availability.
        source = load_banking77(config["dataset_revision"])
        all_rows = [r for values in source.values() for r in values]
        ids, hashes, _ = observed_exclusions(ROOT)
        by_id = {r["id"]: r for r in all_rows}
        rows = [by_id[r["id"]] for r in frozen["rows"]]
        validate_inputs(rows, frozen["rows"])
        if ids.intersection(r["id"] for r in rows) or hashes.intersection(text_hash(r["text"]) for r in rows):
            raise ValueError("Confirmation no longer pipeline-untouched")
        prep = Path(args.preparation).resolve()
        if not prep.is_relative_to(ROOT / "experiments/runs"):
            raise ValueError("Preparation must stay within this repository")
        prepared = json.loads((prep / "frozen_input_texts.json").read_text())
        validate_inputs(prepared, frozen["rows"])
        smoke = json.loads((prep / "smoke_texts.json").read_text())
        if [r["id"] for r in smoke] != frozen["smoke_observed_development_ids"] or any(r["id"] not in ids or r != by_id[r["id"]] for r in smoke):
            raise ValueError("Smoke inputs must be exact already-observed development inputs")
        write_json(run / "frozen_input_texts.json", rows)
        write_json(run / "smoke_texts.json", smoke)
        manifest["environment"] = verify_cuda()
        manifest["environment"]["installed_packages"] = subprocess.check_output([str(ROOT / ".conda/envs/llm-extraction-defense/python.exe"), "-m", "pip", "freeze"], text=True).splitlines()
        before = driver_memory()
        required = spec["parameters_bf16"] * 2 / 2**20 + spec["runtime_allowance_mib"] + config["minimum_driver_free_mib"]
        manifest["resource_preflight"] = {**before, "required_preload_budget_mib": required,
            "bf16_parameter_bytes_mib": spec["parameters_bf16"] * 2 / 2**20,
            "historical_runtime_allowance_mib": spec["runtime_allowance_mib"], "minimum_driver_free_mib": 1024,
            "policy": "Retain Task03 conservative preload allowance; do not reduce it to bypass current GPU contention"}
        write_json(run / "manifest.json", manifest)
        if before["free_mib"] < required:
            raise MemoryError(f"Conservative preload gate: {before['free_mib']:.0f} MiB driver-free < {required:.2f} MiB required; no Teacher loaded")
        seed_everything(17)
        with GuardedPhase("model_load", resources):
            model, tokenizer = load_model(spec)
        if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != frozen["chat_template_sha256"]:
            raise ValueError("Native tokenizer chat rendering mismatch")
        variant = next(v for v in json.loads((ROOT / config["prompt_source"]).read_text())["variants"] if v["id"] == "definitions")
        template = variant["template"]
        if hashlib.sha256(template.encode()).hexdigest() != PROMPT_SHA256:
            raise ValueError("Prompt template mismatch")
        for phase, inputs in (("smoke", smoke), ("confirmation", rows)):
            filename = "smoke_predictions.jsonl" if phase == "smoke" else "predictions.jsonl"
            with GuardedPhase(phase, resources) as guard, (run / filename).open("x", encoding="utf-8") as output:
                for index, row in enumerate(inputs):
                    guard.check()
                    rendered = tokenizer.apply_chat_template([{"role": "user", "content": template.format(text=row["text"])}], tokenize=False, add_generation_prompt=True)
                    rendered_hash = hashlib.sha256(rendered.encode()).hexdigest()
                    if len(tokenizer.encode(rendered, add_special_tokens=False)) > 512:
                        raise ValueError("Input cap exceeded; no truncation allowed")
                    if phase == "confirmation" and rendered_hash != frozen["rows"][index]["rendered_input_sha256"]:
                        raise ValueError("Rendered-input mismatch")
                    attempt = ledger.start(phase, row["id"], rendered_hash, PROMPT_SHA256)
                    call_started = time.perf_counter()
                    try:
                        raw = generate(model, tokenizer, template, row["text"], 512, 12)
                        label = normalize_label(raw)
                        ledger.finish(attempt, "valid" if label else "invalid", label, time.perf_counter() - call_started)
                    except BaseException as error:
                        ledger.finish(attempt, f"error:{type(error).__name__}", None, time.perf_counter() - call_started)
                        raise
                    output.write(json.dumps({"id": row["id"], "raw": raw, "label": label}) + "\n")
                    output.flush()
                    guard.check()
                    if (index + 1) % 8 == 0:
                        print(f"{phase}: {index + 1}/{len(inputs)} attempts", flush=True)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "blocked" if ledger.sequence == 0 else "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        manifest["failures"].append({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        print(manifest["failure"], flush=True)
    finally:
        manifest["resources"] = resources
        manifest["wall_seconds"] = time.perf_counter() - started
        manifest["resource_preflight"] = manifest.get("resource_preflight")
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if all_rows:
            ids, hashes, _ = observed_exclusions(ROOT)
            manifest["remaining_inventory_after_run"] = remaining_inventory(all_rows, ids, hashes, {r["id"] for r in rows})
        manifest["raw_files_sha256"] = {p.name: digest(p) for p in run.iterdir() if p.is_file() and p.name != "manifest.json"}
        write_json(run / "manifest.json", manifest)
        summary = build_summary(run)
        write_json(run / "rebuilt_summary.json", summary)
        (run / "rebuilt_report.md").write_text(render_report(summary), encoding="utf-8")
        print(json.dumps({"run_id": run_id, "status": manifest["status"], "attempts": ledger.counts}), flush=True)


if __name__ == "__main__":
    main()
