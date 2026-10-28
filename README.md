# LLM Extraction Defense Research

Undergraduate senior research project investigating defenses against black-box
LLM model extraction. Task 01 establishes the environment and protocol only.
No defense, model adapter, extraction runner, model download, or training is
implemented. No novelty claim is made.

## Roles and review boundary

- Student: project owner; approves research decisions.
- ChatGPT: research and algorithm design, experimental protocol, analysis.
- Codex: implementation, tests, execution, reproducibility.

Review the research question, threat model, model/data choices, and protocol
before implementation continues.

## Repository layout

```text
configs/          Versioned experiment specifications (currently provisional)
src/teacher/      Future authorized black-box teacher interface
src/attacker/     Future student and extraction code
src/defense/      Future defense code; no algorithm yet
src/evaluation/   Future agreement, task, utility, and resource evaluation
tests/            Scaffold/configuration checks; no model execution
experiments/      Run artifact contracts and manifest template
results/          Metrics contract and template; no measured results yet
docs/             Environment evidence, research question, protocol, research log
environment.yml   Conda environment specification
```

## Conda setup (PowerShell, repository root)

Use Conda exclusively. Do not install into `base` or use venv, virtualenv,
Poetry, or uv. The environment's declared name and directory basename are
`llm-extraction-defense`; the local prefix keeps artifacts inside this repository.
Set the session-local environment search path for name-based activation; the
prefix also works without global configuration changes.

```powershell
conda --version
$env:CONDA_ENVS_PATH = Join-Path (Get-Location) '.conda/envs'
$env:CONDA_PKGS_DIRS = Join-Path (Get-Location) '.conda/pkgs'
conda env create --name llm-extraction-defense --file environment.yml
conda activate llm-extraction-defense
python -m pytest -q
```

If Conda's PowerShell function is not loaded, load it for the current shell
before activation: `(& conda shell.powershell hook) | Out-String | Invoke-Expression`.

If already created, activate it; do not recreate or overwrite it. For a reviewed
dependency change, use `conda env update --prefix ... --file environment.yml`
without `--prune`, inspect the solver plan, and refresh the snapshots below.
Conda may register the prefix in its own user-level environment registry; project
files, environment packages, and download caches belong inside this repository.

Prefer Conda packages. If a necessary GPU/model package requires pip, first
activate this environment and check `python -c "import sys; print(sys.prefix)"`.
Only then use `python -m pip`; record the exact version, source/index, and
resolved dependencies in `environment.yml` and a version snapshot. Do not
install GPU packages until checking the GPU architecture, driver, Python ABI,
wheel runtime, and platform together. See [environment report](docs/environment_report.md).

The initial environment is for configuration and tests. PyTorch and a bundled
CUDA runtime are deliberately absent; GPU execution is not validated.

## Reproducibility conventions

- Configurations are YAML with `schema_version`, status, condition, seeds,
  model and tokenizer revisions, dataset/split revisions, budgets, decoding,
  and training parameters. `null` means undecided, never an automatic default.
  The baseline template cannot authorize execution.
- Pair conditions using seeds `[17, 29, 43]`; use a separate denial RNG. Before
  process startup set `PYTHONHASHSEED` to the run seed and, when applicable,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Seed Python `random`, NumPy, PyTorch CPU,
  and all CUDA devices; seed data-loader generators/workers separately.
  Request deterministic PyTorch algorithms, disable cuDNN benchmarking, and
  record failures rather than silently falling back. Record TF32, precision,
  package versions, GPU/driver, and all seed streams. Equal seeds alone do not
  guarantee reproducibility across hardware or software versions.
- Use fresh run IDs `YYYYMMDDTHHMMSSZ_<condition>_seed<N>_<short-uuid>`.
  Record UTC timestamps and Asia/Bangkok local dates in the research log.
  Never overwrite a run; write a new directory for every attempt, including
  failures. Capture stdout/stderr without credentials or sensitive prompts.
- Put each resolved config, its SHA-256, code commit and dirty diff/hash,
  dependency snapshot, model/data IDs and immutable revisions, prompt template
  hash, tokenizer, adapter, training, decoding, and quantization versions in
  `experiments/runs/<run-id>/`. See [manifest contract](experiments/README.md).
- Save numeric metrics and denominators as JSON in `results/runs/<run-id>/`;
  use `null` plus status/reason for unmeasured values. Keep failed/OOM runs.
  See [metrics contract](results/README.md). Templates contain no measurements.

Refresh dependency records from the activated environment after each change:

```powershell
conda list --explicit --md5 | Set-Content -Encoding utf8 docs/conda-win-64.explicit.txt
conda env export | Where-Object { $_ -notmatch '^prefix:' } | ForEach-Object { $_ -replace '^  - defaults$', '  - nodefaults' } | Set-Content -Encoding utf8 docs/environment.resolved.yml
conda list --json | Set-Content -Encoding utf8 docs/dependency_versions.json
```

The export command removes the machine-specific prefix and replaces Conda's
automatically appended `defaults` channel with `nodefaults`, preserving this
environment's single-channel policy. Verify all installed package channels
before applying that normalization after a future dependency change.

The explicit file is a platform-specific Conda package lock. The resolved export
also captures any later pip packages; neither locks model or dataset contents.
Use the manifest's immutable revisions/checksums for those.

## Artifact and credential hygiene

Store future Hugging Face caches under `.cache/huggingface`, Torch caches under
`.cache/torch`, and raw datasets under `data/`; set `HF_HOME` and `TORCH_HOME`
to those repository-local paths before any future download. `.gitignore`
excludes environments, caches, raw data, weights, common binary formats, and
run directories. Never force-add them. Small aggregate metrics may be copied
to a reviewed, versioned summary with provenance; raw responses stay ignored.

Before staging/committing, inspect `git status --short`, `git diff --cached`,
and staged file sizes. Reject files over 10 MiB, weights/datasets, credentials,
and private prompts even if their extension is not ignored. `.gitignore` is
not a secret scanner or a universal size limit. Keep credentials out of YAML,
logs, notebooks, snapshots, and command history. No credentials are needed now.

## Research documents

- [Environment and 8 GB GPU risks](docs/environment_report.md)
- [Provisional research question](docs/research_question.md)
- [Experiment protocol and first baseline plan](docs/experiment_protocol.md)
- [Research log](docs/research_log.md)
