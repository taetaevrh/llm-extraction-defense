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
