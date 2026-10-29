"""Inference-only process: teacher permutations or auxiliary Student scoring."""

from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time

import yaml

from src.diagnostics.metrics import diagnostic_metrics, meets_quality_targets
from src.diagnostics.resources import GuardedPhase, driver_memory
from src.diagnostics.scoring import forced_choice
from src.evaluation.metrics import agreement
from src.models import load_model
from src.preflight import ROOT, configure_paths
from src.runtime import seed_everything, write_json
from src.task import LABELS, PROMPT_TEMPLATE, normalize_label


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class AttemptLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.touch(exist_ok=False)
        self.count = 0

    def append(self, record):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def start(self, row_id, phase, prompt_hash, candidates=1, scoring_rule_hash=None):
        self.count += 1
        self.append({"event": "start", "attempt": self.count, "id": row_id, "phase": phase,
                     "prompt_sha256": prompt_hash, "candidate_evaluations": candidates,
                     "scoring_rule_sha256": scoring_rule_hash,
                     "at_utc": datetime.now(timezone.utc).isoformat()})
        return self.count

    def finish(self, attempt, result, seconds):
        self.append({"event": "finish", "attempt": attempt, "result": result, "seconds": seconds})


def generate(model, tokenizer, template, text, max_input, max_new):
    import torch
    from transformers import GenerationConfig
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": template.format(text=text)}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    if inputs["input_ids"].shape[1] > max_input:
        raise ValueError("Frozen prompt token cap exceeded")
    inputs = inputs.to("cuda")
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new,
                                  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
                                  use_cache=True)
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=generation)
    torch.cuda.synchronize()
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def teacher_diagnosis(config, role, directory, run):
    spec = config[role]
    if role == "strong_teacher":
        before = driver_memory()
        minimum = spec["parameters_bf16"] * 2 / 2**20 + spec["runtime_allowance_mib"] + 1024
        write_json(directory / "before_load_memory.json", {**before, "required_preload_budget_mib": minimum})
        if before["free_mib"] < minimum:
            raise MemoryError("Conservative BF16 loading estimate cannot retain required headroom")
    resources = {}
    with GuardedPhase("model_load", resources) as guard:
        model, tokenizer = load_model(spec)
        guard.check()
    prompts = json.loads((ROOT / config["prompt_manifest"]).read_text())["variants"]
    rows = json.loads((run / "validation_inputs.json").read_text())
    ledger = AttemptLedger(directory / "attempts.jsonl")
    variants = {}
    try:
        for variant in prompts:
            predictions = []
            path = directory / f'{variant["id"]}.jsonl'
            seed_everything(config["seed"])
            with GuardedPhase(variant["id"], resources) as guard, path.open("x", encoding="utf-8") as log:
                for row in rows:
                    guard.check()
                    attempt = ledger.start(row["id"], variant["id"], variant["sha256"])
                    started = time.perf_counter()
                    try:
                        raw = generate(model, tokenizer, variant["template"], row["text"],
                                       variant["max_input_tokens"], config["decoding"]["max_new_tokens"])
                        pred = normalize_label(raw)
                        ledger.finish(attempt, "valid" if pred else "invalid", time.perf_counter() - started)
                    except BaseException as error:
                        ledger.finish(attempt, f"error:{type(error).__name__}", time.perf_counter() - started)
                        raise
                    log.write(json.dumps({"id": row["id"], "raw": raw, "label": pred}) + "\n")
                    log.flush()
                    predictions.append(pred)
                    guard.check()
            metrics = diagnostic_metrics(predictions, [r["label"] for r in rows], variant["order"][-1])
            metrics["meets_quality_targets"] = meets_quality_targets(metrics)
            metrics["queries"] = len(rows)
            metrics["resources"] = resources[variant["id"]]
            metrics["order"] = variant["order"]
            metrics["prompt_sha256"] = variant["sha256"]
            variants[variant["id"]] = metrics
            write_json(directory / "summary.json", {"status": "running", "model": spec, "variants": variants,
                                                    "attempts": ledger.count, "resources": resources})
            print(f'{role} {variant["id"]}: accuracy={metrics["accuracy"]:.3f} macro_f1={metrics["macro_f1"]:.3f} final={metrics["final_option_rate"]:.3f}', flush=True)
        write_json(directory / "summary.json", {"status": "completed", "model": spec, "variants": variants,
                                                "attempts": ledger.count, "resources": resources})
    finally:
        write_json(directory / "resources.json", resources)


def student_diagnosis(config, role, directory, run):
    from peft import PeftModel
    prior = ROOT / config["prior_run"]
    resources = {}
    with GuardedPhase("model_load", resources):
        model, tokenizer = load_model(config["student"])
        if role == "student_adapted":
            adapter = prior / "student_adapter"
            if not (adapter / "adapter_model.safetensors").exists():
                write_json(directory / "summary.json", {"status": "unavailable", "reason": "Saved Task02 adapter missing"})
                return
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
            model.eval()
    ledger = AttemptLedger(directory / "attempts.jsonl")
    splits = json.loads((ROOT / "docs/banking77_split_manifest.json").read_text())["subsets"]
    inputs = json.loads((run / "student_inputs.json").read_text())
    extraction = jsonl(prior / "extraction_responses.jsonl")
    majority_label = Counter(r["teacher_label"] for r in extraction if r["teacher_label"] in LABELS).most_common(1)[0][0]
    summaries = {}
    prompt_hash = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()
    scoring_rule_hash = hashlib.sha256(config["student_diagnostic"]["score"].encode()).hexdigest()
    try:
        for name in config["student_diagnostic"]["subsets"]:
            rows = inputs[name]
            predictions = []
            with GuardedPhase(name, resources) as guard, (directory / f"{name}.jsonl").open("x", encoding="utf-8") as log:
                for index, row in enumerate(rows):
                    guard.check()
                    attempt = ledger.start(row["id"], name, prompt_hash, candidates=8, scoring_rule_hash=scoring_rule_hash)
                    started = time.perf_counter()
                    try:
                        record = forced_choice(model, tokenizer, row["text"])
                        ledger.finish(attempt, "scored", time.perf_counter() - started)
                    except BaseException as error:
                        ledger.finish(attempt, f"error:{type(error).__name__}", time.perf_counter() - started)
                        raise
                    record["id"] = row["id"]
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    predictions.append(record["label"])
                    guard.check()
                    if (index + 1) % 50 == 0:
                        print(f'{role} {name}: {index + 1}/{len(rows)}', flush=True)
            refs = jsonl(prior / f"teacher_{name}.jsonl")
            truth = [r["label"] for r in splits[name]]
            if not ([r["id"] for r in refs] == [r["id"] for r in rows] == [r["id"] for r in splits[name]]):
                raise ValueError("Historical diagnostic row alignment mismatch")
            reference = [r["teacher_label"] for r in refs]
            original = jsonl(prior / f'{"after" if role == "student_adapted" else "before"}_{name}.jsonl')
            if [r["id"] for r in original] != [r["id"] for r in rows]:
                raise ValueError("Free-generation historical rows mismatch")
            old = [r["label"] for r in original]
            constant = [majority_label] * len(rows)
            summaries[name] = {
                "forced_choice": {**diagnostic_metrics(predictions, truth), "teacher_agreement": agreement(predictions, reference)},
                "historical_free_generation": {**diagnostic_metrics(old, truth), "teacher_agreement": agreement(old, reference)},
                "majority_comparator": {**diagnostic_metrics(constant, truth), "teacher_agreement": agreement(constant, reference), "label_source": "Task02 extraction teacher responses"},
                "inference_attempts": len(rows), "candidate_evaluations": 8 * len(rows), "resources": resources[name],
            }
            write_json(directory / "summary.json", {"status": "running", "role": role, "subsets": summaries, "attempts": ledger.count, "resources": resources})
        write_json(directory / "summary.json", {"status": "completed", "role": role, "subsets": summaries,
                                                "attempts": ledger.count, "candidate_evaluations": ledger.count * 8, "resources": resources})
    finally:
        write_json(directory / "resources.json", resources)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--role", choices=["small_teacher", "strong_teacher", "student_pretrained", "student_adapted"], required=True)
    args = parser.parse_args()
    configure_paths()
    seed_everything(17)
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Required CUDA/BF16 unavailable")
    config = yaml.safe_load(Path(args.config).read_text())
    run = Path(args.run)
    directory = run / args.role
    directory.mkdir(exist_ok=False)
    try:
        if args.role.endswith("teacher"):
            teacher_diagnosis(config, args.role, directory, run)
        else:
            student_diagnosis(config, args.role, directory, run)
    except BaseException as error:
        import traceback
        write_json(directory / "failure.json", {"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "driver_memory_after_failure": driver_memory(),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20})
        raise


if __name__ == "__main__":
    main()
