"""Task 03: frozen teacher comparisons and auxiliary evaluation, no training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import yaml

from src.data import load_banking77, text_hash
from src.diagnostics.metrics import meets_quality_targets
from src.diagnostics.resources import driver_memory
from src.models import local_model_path
from src.preflight import ROOT, configure_paths
from src.runtime import write_json


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_stronger(config, run):
    spec = config["strong_teacher"]
    provenance = json.loads((ROOT / "docs/task03_stronger_teacher_provenance.json").read_text())
    if provenance["revision"] != spec["revision"] or provenance["license"] != "apache-2.0":
        raise ValueError("Stronger Teacher license/revision gate failed")
    memory = driver_memory()
    required = provenance["minimum_driver_free_mib_before_download"]
    write_json(run / "stronger_download_preflight.json", {**memory, "required_mib": required,
        "weights_estimate_mib": provenance["estimated_weights_mib"], "runtime_allowance_mib": 1024,
        "headroom_mib": 1024, "at_utc": datetime.now(timezone.utc).isoformat()})
    if memory["free_mib"] < required:
        raise MemoryError("Stronger Teacher exceeds conservative available-memory budget before download")
    from huggingface_hub import snapshot_download
    directory = Path(snapshot_download(spec["model_id"], revision=spec["revision"], local_dir=local_model_path(spec),
        allow_patterns=["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "vocab.json", "merges.txt", "model.safetensors",
                        "model.safetensors.index.json", "model-*.safetensors", "LICENSE", "README.md"]))
    write_json(run / "stronger_model_artifacts.json", {
        "model": spec, "files": {p.relative_to(directory).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
                                   for p in directory.rglob("*") if p.is_file() and ".cache" not in p.parts}})


def run_worker(role, config_path, run):
    with (run / f"{role}.stdout.log").open("x", encoding="utf-8") as stdout, (run / f"{role}.stderr.log").open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen([sys.executable, "-u", "-m", "src.diagnostics.worker", "--role", role,
                                    "--config", str(config_path), "--run", str(run)], stdout=stdout, stderr=stderr)
        code = process.wait()
    if code:
        raise RuntimeError(f"{role} stopped with exit {code}; inspect its failure record")
    return json.loads((run / role / "summary.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task03_diagnostics.yaml")
    args = parser.parse_args()
    configure_paths()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "main":
        raise RuntimeError("Task03 must run on main")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Commit verified source/configuration before inference")
    validation = json.loads((ROOT / config["validation_manifest"]).read_text())
    prompts = json.loads((ROOT / config["prompt_manifest"]).read_text())
    assert validation["examples"] == 88 and len(prompts["variants"]) == 10
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_task03_diagnostics_" + uuid.uuid4().hex[:8]
    run = ROOT / "experiments/runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    result_dir = ROOT / "results/runs" / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    source = load_banking77(config["dataset_revision"])
    source_by_id = {r["id"]: r for rows in source.values() for r in rows}
    inputs = []
    for row in validation["rows"]:
        actual = source_by_id[row["id"]]
        if text_hash(actual["text"]) != row["text_sha256"] or actual["label"] != row["label"]:
            raise ValueError("Frozen validation manifest does not match pinned dataset")
        inputs.append(actual)
    write_json(run / "validation_inputs.json", inputs)
    old = json.loads((ROOT / config["prior_split_manifest"]).read_text())["subsets"]
    write_json(run / "student_inputs.json", {name: [{"id": r["id"], "text": source_by_id[r["id"]]["text"]} for r in old[name]]
                                            for name in config["student_diagnostic"]["subsets"]})
    for source_path in [config_path, ROOT / config["validation_manifest"], ROOT / config["prompt_manifest"]]:
        shutil.copy2(source_path, run / source_path.name)
    # Hash all tracked Task01/02 artifacts, except appendable research log/README.
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "a95d0ba"], text=True).splitlines()
    protected = {name: file_hash(ROOT / name) for name in tracked if name not in ("README.md", "docs/research_log.md")}
    manifest = {"schema_version": 1, "run_id": run_id, "status": "started", "stage": "small_teacher",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_clean": True, "seed": 17,
        "frozen_files_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in [config_path, ROOT / config["validation_manifest"], ROOT / config["prompt_manifest"]]},
        "environment_inventory_sha256": file_hash(ROOT / "docs/banking77_dependency_versions.json"),
        "prior_model_artifact_manifest_sha256": file_hash(ROOT / "experiments/banking77_pilot_manifest.json"),
        "adapter_files_sha256": {p.name: file_hash(p) for p in (ROOT / config["prior_run"] / "student_adapter").glob("*") if p.is_file()},
        "protected_Task01_Task02_files_sha256": protected, "failures": []}
    write_json(run / "manifest.json", manifest)
    summaries = {}
    started = time.perf_counter()
    print(f"RUN_ID={run_id}", flush=True)
    try:
        # CPU-only coordinator: every GPU model gets its own short-lived process.
        for role in ("small_teacher", "student_pretrained", "student_adapted"):
            manifest["stage"] = role
            write_json(run / "manifest.json", manifest)
            print(f"Starting {role}", flush=True)
            summaries[role] = run_worker(role, config_path, run)
            print(f"Finished {role}", flush=True)
        small_passed = any(meets_quality_targets(v) for v in summaries["small_teacher"]["variants"].values())
        if not small_passed:
            manifest["stage"] = "stronger_preflight"
            write_json(run / "manifest.json", manifest)
            prepare_stronger(config, run)
            manifest["stage"] = "strong_teacher"
            write_json(run / "manifest.json", manifest)
            print("Small Teacher failed quality criteria; starting frozen stronger comparison", flush=True)
            summaries["strong_teacher"] = run_worker("strong_teacher", config_path, run)
        else:
            summaries["strong_teacher"] = {"status": "not_needed", "reason": "At least one small-Teacher prompt met all predefined criteria; no larger model downloaded"}
        manifest["status"] = "completed"
        manifest["stage"] = "completed"
    except BaseException as error:
        import traceback
        manifest["status"] = "stopped"
        manifest["failures"].append({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(), "stage": manifest["stage"]})
        write_json(run / "failure.json", manifest["failures"][-1])
        raise
    finally:
        unchanged = {name: file_hash(ROOT / name) == value for name, value in protected.items()}
        if not all(unchanged.values()):
            manifest["failures"].append({"type": "ProtectedArtifactChanged", "files": [name for name, same in unchanged.items() if not same]})
            manifest["status"] = "stopped"
        manifest["Task01_Task02_artifacts_preserved"] = all(unchanged.values())
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["wall_seconds"] = time.perf_counter() - started
        write_json(run / "manifest.json", manifest)
        write_json(result_dir / "summary.json", {"run_id": run_id, "status": manifest["status"], "results": summaries,
            "failures": manifest["failures"], "wall_seconds": manifest["wall_seconds"], "training_steps": 0,
            "new_extraction_attempts": 0, "defense_evaluations": 0})
    print(json.dumps({"run_id": run_id, "status": manifest["status"], "wall_seconds": manifest["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
