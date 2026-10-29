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

## 2026-10-29 - Task 02 resumed with BANKING77

- Project owner replaced TweetEval with PolyAI/banking77 and approved eight
  banking intents, the same pinned teacher/student, seed 17, and a 500-attempt
  no-defense pilot. Original TweetEval preflight files are retained unchanged.
- Recorded BANKING77 revision 1846984189f2bc0ac51c6fc3a600fa9e1714e0c6,
  publisher commit 57ec275d8078af65b7731c2a98be812d844a6d6b, CC BY 4.0,
  creators, paper citation, transformations, and pinned-source hashes before
  downloading data. No conflicting terms found in the reviewed sources.
- Activated existing llm-extraction-defense. Added supporting dependencies
  through conda-forge; installed official Windows CPython 3.11 CUDA 12.8
  torch 2.7.1+cu128 wheel and pinned transformers/datasets/accelerate/peft via
  pip inside that environment with no dependency replacement. Initial pip
  check found that Conda xxhash was the C library only; added python-xxhash
  through Conda. Final pip check found no broken requirements. Base not modified.
- CUDA availability and BF16 support verified. A 16x16 GPU matmul matched CPU
  exactly. Recorded driver/architecture/runtime and both CUDA and NVIDIA-SMI
  memory observations, which differ under WDDM. Device-wide sampling uses
  NVIDIA-SMI in later model phases. Exact preflight numbers are in the JSON.
- Added failing behavior tests before implementation. Initial collection
  failed because src.data did not yet exist. After implementation, 26 tests
  passed in 7.24s, including real tiny causal-LM response-loss verification.
- Initial train-only allocation for extraction, validation, and legitimate
  rows lacked four distinct lost_or_stolen_card examples. No teacher queries
  had run. Revised legitimate rows to unused official-test examples, within
  the owner's source-agnostic disjoint-subset requirements. Final freeze has
  500 extraction, 100 validation, 200 test, and 100 legitimate rows; all eight
  labels meet quotas and global text/ID checks pass. Three normalized train
  duplicates removed. Split manifest SHA-256 is pinned in the config.
- Fixed prompt/normalization, separate teacher process, complete attempt
  accounting, target-response/EOS-only LoRA loss, smoke adapter discard,
  fresh pretrained measurement, paired evaluation, and failure/resource logs
  are specified in banking77_protocol.md. No defense implementation added.
- Initial execution 20261028T195641Z_banking77_seed17_fd03c5eb failed during
  model artifact preparation with WinError 1314 (cache symlink privilege).
  CUDA and frozen splits had passed; no teacher/model inference or training,
  zero teacher queries. Run-local manifest/failure files are preserved.
  Switched snapshot_download to ordinary revision-specific local directories
  under ignored models/, with local-only model loading. No OS privilege change,
  package change, revision change, or outcome-based experiment adjustment.
- After the cache-path fix, `python -m pytest -q`: **27 passed in 7.31s**.
  Executed successful pilot from clean source commit
  a11165eceadd161a50b983abd9cc7833efceeae1 on task02/no-defense-baseline.
- Run 20261028T195903Z_banking77_seed17_1a1274a4 completed. Smoke used eight
  teacher calls, produced eight valid teacher labels, and changed smoke-student
  valid outputs from 0/8 to 8/8 after 16 training steps. The smoke adapter was
  discarded. Main pilot: 500 extraction attempts, 498 valid teacher labels,
  two invalid responses; 200 test + 100 legitimate calls. **808 total** teacher
  calls including smoke; no retries or extra extraction calls.
- Main adaptation: 498 teacher-only examples, three epochs, 187 optimizer
  updates, 460800 trainable LoRA parameters, 8610 supervised response/EOS tokens.
  Losses and gradients remained finite. No pilot CUDA failure or OOM.
- On the same 200 held-out examples: teacher accuracy 95/200 (47.5%);
  student-teacher agreement 0/200 -> 164/200 (82%); student ground-truth accuracy
  0/200 -> 74/200 (37%). Invalid student outputs 197 -> 0; teacher invalid 0.
  Teacher predicted verify_my_identity 128/200; student after predicted it
  150/200. No teacher/student test exchange_rate predictions. These limitations
  prevent treating agreement as strong evidence of useful capability recovery.
- Separate legitimate set: teacher accuracy 36/100, student accuracy 0/100 ->
  35/100, agreement 0/100 -> 83/100; invalid student outputs 99 -> 0.
  Teacher coverage 100%, no denial, latency p50 0.293s and p95 0.322s.
- Measured pilot wall time 966.21s; successful preparation 21.70s; smoke 38.57s;
  total successful run including both 1026.49s. Excludes dependency installation
  and the earlier failed preparation. Student training: 327.76s, peak Torch
  allocated 637.53 MiB, reserved 876 MiB. Teacher query peak: 990.78 MiB
  allocated, 1030 MiB reserved. Highest sampled whole-device use: 4124 MiB,
  including other applications. Driver sampling may miss short spikes.
- Retained Qwen eager/sliding-window warning; pinned config explicitly has
  use_sliding_window=false, sliding_window=32768, and inputs capped at 256.
  No backend/settings changed after outcomes. All captured model-file hashes
  remained unchanged after the run.
- Independently recomputed all reported accuracy/agreement/invalid counts and
  checked row alignment, 808 ledger entries, frozen split hash, finite updates,
  and teacher-only training artifact schema. Reviewed summaries/manifests and
  baseline report saved in versioned files; raw data, predictions, weights,
  adapter, and caches remain ignored. Base Conda history observation unchanged.
- Scientific conclusion: technical pipeline feasibility and task-specific
  agreement improvement observed; teacher bias and output-format confounding
  require review before defense comparisons. One seed/eight intents do not
  establish general extraction success or full-LLM recovery. Stopped for review.

## 2026-10-29 - Task 03 diagnostic protocol and inventory

- Owner requested diagnosis only, directly on main, no branches/PRs, no new
  training or defense, with a verified commit and normal push after completion.
  Initial origin fetch showed main...origin/main = 0 ahead / 0 behind.
- Audited Task02 split/protocol/report/code/results and all local BANKING77
  response artifacts. Found 900 assigned canonical rows, 808 actually queried
  rows, and eight smoke rows. Exclude all prior assignments from a new test;
  the eight smoke rows also cannot enter the Teacher validation diagnostic.
- Froze 88 eligible reserved validation examples, 11 per class, seed 17.
  These rows were neither extraction-training nor smoke-training examples.
  Known test/legitimate outcomes are not used for prompt/model/rule selection.
- The per-class inventory covers all 77 classes. In the eight-intent scope,
  remaining mixed-source class counts are 79,37,46,38,11,77,57,33 in approved
  order. Thus at most 88 balanced fresh examples (11/class), or just 16 from
  official-test rows alone. No new independent test was created or queried.
- Froze ten unique Teacher prompts before inference: rotations 0..7 (original
  is rotation_0), reverse, and intent definitions. Every label occupies each
  position once across the rotations. Metrics, paired/stratified bootstrap,
  selection ranking and conditional larger-model comparison are preregistered
  in task03_protocol.md and configs/task03_diagnostics.yaml.
- Verified Qwen2.5-1.5B-Instruct revision
  989aa7980e4cf806f80c7fef2b1adb7bc71aa306 and Apache 2.0 card before any weight
  download. BF16 weight estimate uses actual parameter count; runtime and
  1 GiB measured-headroom requirements are separate. A driver-memory guard
  applies to loading and inference. Candidate tested only if needed.
- Student auxiliary rule frozen before scoring known historical rows: mean
  conditional label-token log probability, no EOS/prompt/padding contribution,
  equal priors and fixed tie rule. Sum likelihoods are also logged. Preserve
  original free generation as primary API behavior; no new adaptation.
- Added tests for cyclic permutations, metric correctness/invalid handling,
  data exclusions, and causal likelihood indexing. Initial red test failed
  because diagnostic modules did not exist; no package installation was needed.
- Executed new run 20261029T042435Z_task03_diagnostics_932130fe from clean
  source commit 8ac2d51 on main. The original prompt is rotation_0, an explicit
  alias rather than a duplicated independent condition. Both Teachers used
  exactly the same 88 reserved validation inputs across all ten unique prompts.
- Small Teacher: original 35/88 (39.8%) accuracy, macro-F1 0.393; best definitions
  50/88 (56.8%), macro-F1 0.534. None met all quality criteria. Across rotations,
  final-option selection was 86/704 (12.2%), paired/stratified 95% interval
  11.2–13.2%; the valid-adjusted excess interval included zero. verify_my_identity
  counts remained 48–76/88 across positions (59 first, 64 last). Order sensitivity
  was observed, but a general last-option preference did not explain the collapse.
- Conditional stronger-model pre-download check found 5770 MiB driver free
  against a conservative 4992.40 MiB budget (2944.40 estimated weights + 1024
  runtime + 1024 headroom). Before load, 5686 MiB remained. All weights were
  pinned, attributed Apache 2.0, and downloaded only after this gate passed.
- Stronger Teacher completed all ten conditions. Best validation configuration
  was Qwen2.5-1.5B-Instruct + definitions: 79/88 (89.8%), macro-F1 0.897,
  zero invalids, every class recall positive (minimum 8/11), maximum predicted
  label share 13/88 (14.8%). It met all provisional targets. These are selection
  scores, not independent new-test results. No Teacher was fine-tuned.
- Stronger model peak Torch allocated 3001.86 MiB, reserved 3194 MiB; sampled
  device peak 5593 MiB; minimum measured driver free 2357 MiB. No sample breached
  the 1 GiB guard, no OOM, and no simultaneous Teacher/Student GPU residency.
- Student audit used the original frozen prompt and rule, the saved adapter
  with is_trainable=false, and known historical test/legitimate rows only for
  retrospective diagnosis. Test forced-choice accuracy was 19% pretrained and
  35% adapted; macro-F1 0.127 -> 0.332; old-Teacher agreement 8% -> 81%.
  Original generation metrics remain unchanged (0% -> 37% accuracy and 0% ->
  82% agreement). Legitimate forced-choice accuracy 17% -> 35%, agreement
  8% -> 84%. These local-logit scores are not black-box API responses.
- Of Task02's 164 test agreement matches, 121 (73.8%) were majority-label
  matches and 97 (59.1%) copied Teacher errors; 43 were non-majority matches.
  The constant extraction-majority comparator scored 64% agreement and 12.5%
  accuracy. There is behavior beyond a constant response, but formatting,
  Teacher bias, and copied errors dominate interpretation; causal proportions
  of format learning versus semantic transfer are not identifiable here.
- Total Task03 Teacher attempts 1760 (1755 valid, five invalid), plus 600
  Student input attempts / 4800 candidate likelihood evaluations. All attempts
  have start/finish ledger records. No inference failures or replacements.
  Diagnostic execution wall time 638.62 seconds, including stronger download
  and process overhead, excluding protocol/inventory preparation and tests.
- Post-run audit found a logging-field labeling bug: Student prompt_sha256
  held the score-rule identifier hash; the correct original prompt/hash was
  already frozen elsewhere. Preserved raw ledgers and added provenance notes;
  future source records prompt and rule hashes separately. No numeric results
  changed and no inference was rerun. Added coverage for this metadata boundary.
- Final suite: python -m pytest -q, 35 passed in 6.01s; pip check passed.
  Independent checks recomputed Teacher metrics, checked score/selection and
  ledger totals, preserved protected Task01/02 files and adapter hashes, and
  rehashed stronger weights. No environment packages changed; no base install.
- Saved task03_diagnostic_report.md, full aggregate results, reviewed run
  manifest, all-class data inventory, frozen validation/prompt manifests, and
  verification records. Raw examples, responses, scores, caches, and weights
  remain ignored. No new test was manufactured by recycling known rows.
- Recommendation for review: freeze the selected Teacher/prompt and confirm
  it without training on new licensed, independently labeled balanced in-domain
  data, ideally >=200 rows. Existing unused pools permit only 88 mixed-source
  balanced rows or 16 official-test-only rows, so any smaller holdout or task
  change needs an explicit research decision. No next extraction/defense run.

## 2026-10-29 — Task 04 frozen confirmation and proposal preparation

- Continued clean main from b4c65a35f649f0a1577c771708102475ee8be0d5 after fetching origin; equal history, no branch/worktree/PR and no push.
- Reconstructed all prior usage, excluded all900 assignments/canonical hashes and subsequent raw records; froze88 unused rows, eleven per intent, seed17, source77train/11test. Rehashed all nine stronger-model/tokenizer files and exact definitions template. Committed protocol/code/config/manifest before execution as995441350ee21cf90157a42aaf28ea3c5ec0a141.
- Executed only resource preflight in run20261029T071617Z_task04_confirmation_c2f6770c. Existing Conda Python3.11.18, torch2.7.1+cu128/CUDA12.8, driver617.42 passed actual GPU matmul and BF16 checks. Conservative preload gate failed:4,406MiB driver-free <4,992.401367MiB required. Teacher never loaded; smoke0, confirmation0, total0 attempts; runtime18.4681155s. Confirmation metrics, intervals and decision are null. Model workload runtime/peaks null, distinct from tiny-check peak32.002441MiB. No retry or settings change.
- All378 pipeline-unused rows remain unobserved;88 reserved and290 outside reservation. No claim of independent Teacher confirmation or pretraining absence. The old79/88 remains a selection score; known Student/Teacher results were not regenerated or used as new confirmation.
- CPU audit reran historical aggregates, raw alignment/normalization/query hashes, all20 Teacher conditions, Student likelihood selection/agreement and ledgers. Final78 check records have no unavailable dependent artifacts. Inventoried101 local historical raw files, preserved prior evidence and the Student-ledger provenance correction. Offline Task04 rebuild exactly matches reviewed JSON/Markdown; Git preservation verifies75 prior tracked files with only additive README/log entries.
- Tests: python -m pytest -q,43 passed in7.50s after strengthened validators; pip check passed. All139 Conda package version/build/channel entries equal Task02; no dependency/base modifications or new model/data downloads.
- Reviewed ten primary papers including recent2026 work, read closest methods, recorded versions/hash snapshots and access limits. CAP full text returned403/browser verification and is not counted as reviewed. Class conditioning/pair recurrence have close priors; novelty unresolved. No defense implemented/evaluated.
- Wrote a gated fresh-Student three-seed paired/shuffled baseline plan, independent-data/precision plan, two falsifiable candidate hypotheses and Nov20 Bangkok proposal schedule. Independent human annotations/benign sessions, resource continuation and advisor contribution approval remain unresolved. Stop for research review; no Student training, defense experiment, or push.
