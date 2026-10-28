"""Check research configuration guardrails without models or GPU dependencies."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_initial_config_keeps_execution_pending_design_review():
    config = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text())
    assert config["status"] == "requires_research_design_review"
    assert config["teacher"]["model_id"] is None
    assert config["student"]["model_id"] is None
    assert config["training"]["method"] is None


def test_baseline_spec_has_no_defense_and_counts_attempts():
    config = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text())
    assert config["condition"] == "no_defense"
    assert config["defense"] == {"mode": "none", "denial_rate": 0.0}
    assert config["budget"]["teacher_query_attempts"] > 0
    assert config["evaluation"]["primary_metric"] == (
        "held_out_teacher_student_agreement"
    )


def test_manifest_and_metrics_templates_do_not_invent_results():
    import json

    manifest = json.loads((ROOT / "experiments/manifest.template.json").read_text())
    metrics = json.loads((ROOT / "results/metrics.template.json").read_text())
    assert manifest["status"] == metrics["status"] == "not_run"
    assert manifest["model_versions"]["teacher_revision"] is None
    assert metrics["agreement"]["fraction"] is None
    assert metrics["resources"]["peak_gpu_allocated_mib"] is None


def test_repository_ignores_dangerous_artifact_locations():
    import subprocess

    paths = [
        ".conda/envs/llm-extraction-defense/python.exe",
        ".cache/huggingface/download",
        "models/model.safetensors",
        "data/raw/unusual-extension.payload",
        "experiments/runs/example/manifest.json",
        "results/runs/example/metrics.json",
        "accidental/model.pth",
        ".env",
        "client_credentials.json",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        input=("\0".join(paths) + "\0").encode(),
        capture_output=True,
        cwd=ROOT,
        check=True,
    )
    assert set(result.stdout.decode().rstrip("\0").split("\0")) == set(paths)
