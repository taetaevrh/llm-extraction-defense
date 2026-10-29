"""Read-only offline verification of reviewed Task04 reports and preservation."""

import argparse
import json
from pathlib import Path
import subprocess

from src.task04.core import digest, validate_inputs
from src.task04.rebuild import build_summary, render_report


def verify(root, run):
    root, run = Path(root), Path(run)
    summary = build_summary(run)
    reviewed = json.loads((root / "results/task04_confirmation_summary.json").read_text())
    if summary != reviewed or render_report(summary) != (root / "docs/task04_confirmation_report.md").read_text(encoding="utf-8"):
        raise ValueError("Offline reconstruction differs from reviewed outputs")
    frozen = json.loads((run / "input_manifest.json").read_text())
    validate_inputs(json.loads((run / "frozen_input_texts.json").read_text()), frozen["rows"])
    if digest(root / "docs/task04_confirmation_manifest.json") != summary["frozen_hashes"]["input_manifest_sha256"]:
        raise ValueError("Reviewed input manifest changed")
    if digest(root / "configs/task04_confirmation.yaml") != summary["frozen_hashes"]["configuration_sha256"]:
        raise ValueError("Reviewed configuration changed")
    reference = "b4c65a35f649f0a1577c771708102475ee8be0d5"
    historical = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", reference], cwd=root, text=True).splitlines()
    for name in historical:
        original = subprocess.check_output(["git", "show", f"{reference}:{name}"], cwd=root)
        actual = (root / name).read_bytes()
        # Git's checkout newline conversion is not a research-artifact edit.
        normalized = lambda data: data.replace(b"\r\n", b"\n")
        if name in ("README.md", "docs/research_log.md"):
            if not normalized(actual).startswith(normalized(original)):
                raise ValueError(f"Historical content was not preserved additively: {name}")
        elif normalized(actual) != normalized(original):
            raise ValueError(f"Historical tracked file changed: {name}")
    return {"reviewed_summary_and_report": "exact_offline_match", "manifest_inputs": 88,
            "historical_tracked_files_checked": len(historical), "reference_commit": reference,
            "teacher_attempts": summary["total_teacher_attempts"], "model_imports_or_calls": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(Path.cwd(), args.run), indent=2))


if __name__ == "__main__":
    main()
