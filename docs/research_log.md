# Research log

Use dated entries in Asia/Bangkok, with UTC timestamps in machine run records.
Record decisions, commands, evidence, failures, protocol changes, and reviewer
approval. Separate proposed values from measured outcomes.

## 2026-10-29 - Task 01 initialization

- Starting directory was empty and not a Git repository. No ancestor AGENTS.md
  was found. Initialized a local Git repository on `main`; no existing user
  files were overwritten and no remote/commit was created.
- Detected Windows 11 Home, Conda 25.11.1, RTX 4060 Ti (8,188 MiB), driver
  617.42, installed CUDA toolkit 13.4.92, and WSL2 with a stopped
  docker-desktop distribution. The active base interpreter was Python 3.13.9
  without PyTorch. See the environment report for evidence and limits.
- Chose a repository-local Conda prefix named llm-extraction-defense with a
  small Python 3.11 configuration/test toolchain. Base is inspected read-only.
  Exact installed versions and verification are recorded after creation.
- Added the required directories/documents, YAML baseline proposal, run and
  metrics templates, and ignore rules. All metrics are unmeasured/null.
- The no-defense, matched random-denial, and undesigned-defense conditions are
  provisional. No algorithm/novelty claim, model download, or training.
- Research review pending: models/licenses, task/data/splits, adaptation recipe,
  threat model, budget, denial matching, utility/effect thresholds, GPU build.

### Inspection commands

`git status --short`, `Get-ChildItem -Force`, ancestor `AGENTS.md` lookup,
`Get-Command conda,python,nvidia-smi,wsl`, `conda --version`, `conda info --json`,
`conda env list --json`, `Get-CimInstance` for OS/computer/CPU/logical disks,
`nvidia-smi`, GPU CSV query, `nvcc --version`, `wsl --version`, `wsl --status`,
`wsl --list --verbose`, and Python interpreter/module-discovery probes.

Official NVIDIA compatibility and PyTorch installation documentation was
consulted; the report distinguishes documented compatibility from a runtime test.

### Environment creation and verification

- Executed `conda env create --prefix ./.conda/envs/llm-extraction-defense
  --file environment.yml --yes` with repository-local `CONDA_PKGS_DIRS`;
  creation succeeded. Conda warned about spaces in the existing parent path.
- Set session-local `CONDA_ENVS_PATH=.conda/envs`, loaded the existing
  PowerShell hook, and verified `conda activate llm-extraction-defense` by
  environment name and interpreter prefix. No global Conda configuration edited.
- Installed Python 3.11.18, NumPy 2.2.6, PyYAML 6.0.3, pytest 8.4.2, pip
  26.2.2, and their Conda dependencies; pinned direct versions in environment.yml.
  No pip installs. PyTorch remains absent in the project environment.
- First `python -m pytest -q`: 3 passed, 1 failed because the test's text-mode
  stdin converted LF to CRLF for Git on Windows. Changed the probe to binary
  NUL-delimited paths. Second run: **4 passed in 0.10s**.
- Captured `conda list --explicit --md5`, `conda env export`, and
  `conda list --json` in docs. The YAML snapshot removes the machine prefix
  and changes the auto-added defaults channel to nodefaults to match the
  all-conda-forge installation policy. Source versions and snapshot agree.
- Base Conda history size and timestamp stayed unchanged. Git ignore checks
  exclude local environment/cache directories, run outputs, weight extensions,
  and representative credential paths. No commit, push, models, or training.
- Stopped for research design review at the end of Task 01.

## 2026-10-29 - Task 02 license preflight stopped

- User approved Qwen2.5-0.5B-Instruct / SmolLM2-135M-Instruct on TweetEval
  sentiment with 500 extraction attempts, 200 held-out and 100 legitimate
  validation examples, seed 17, deterministic outputs, no defense.
- Task 01 was already committed at 7ee36b11b21bd23176e4e2c947067ae526134f64
  and the working tree was clean. Created `task02/no-defense-baseline`.
- Resolved exact model/dataset revisions and read pinned cards before any
  weights/data. See task02_provenance.json for API URLs, timestamps, and hashes.
  Both model cards identify Apache 2.0. The dataset repository metadata is
  unknown, while the sentiment-specific card identifies CC BY 3.0 and also
  requires Twitter/API terms compliance.
- Checked the referenced historical task-organizer license statement,
  upstream README, and current X Developer Agreement/Policy. The agreement's
  foundation-model fine-tuning restriction raises an unresolved applicability
  question for this historical off-platform dataset. Triggered the user's
  explicit unclear-license stop condition. No categorical legal conclusion.
- No pip/Conda package installations, weight/data downloads, teacher queries,
  GPU tensor computation, split creation, or training. The project Conda
  environment was activated by name and Python 3.11.18 / PyTorch absence
  confirmed. environment.yml and all Task 01 dependency records are unchanged.
- `nvidia-smi` observed 8,188 MiB total and 6,123 MiB free VRAM, driver 617.42,
  compute capability 8.9. These are preflight values, not experiment peaks.
- Existing suite: `python -m pytest -q` in activated llm-extraction-defense:
  **4 passed in 0.16s**. No extraction pipeline tests or measurements exist yet.
- Saved non-executable pilot config, provenance, preflight manifest, null
  metrics, and task02_failure_report.md. Stopped pending a research decision
  on dataset permission/applicability or an approved replacement dataset.
