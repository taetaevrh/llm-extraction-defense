# Task 04 reproducibility audit

Review date: 2026-10-29 (Asia/Bangkok). Repository `taetaevrh/llm-extraction-defense`, local checkout `D:/MU/Y3S3-Y4S1/Senior Project/me-al`, branch **main**. Initial fetch found local main and origin/main equal at `b4c65a35f649f0a1577c771708102475ee8be0d5`; no divergent history or user changes. No applicable ancestor AGENTS.md was found. No branch/worktree/PR was created, no push was performed, and synced `sources/` was not modified.

## Registration and actual execution

Protocol/implementation/frozen manifest/configuration were committed **before** the single execution as `995441350ee21cf90157a42aaf28ea3c5ec0a141`. That is the executed-source SHA, distinct from the later reporting/audit commit. Preparation `experiments/runs/20261029T071400Z_task04_preparation_2ea5380c` contains local input/smoke texts. The persistent one-shot claim is `experiments/runs/task04_confirmation_claim.json`; it has not been removed/reset.

Actual run: `experiments/runs/20261029T071617Z_task04_confirmation_c2f6770c`. Status **blocked**, not a completed confirmation. The committed result is rebuilt from its local manifest and empty attempt ledger. Later source changes strengthen offline aggregate/ledger validation and reviewed-output verification; no Teacher calls were rerun. The original run manifest and raw artifact hash list remain unchanged.

| Measurement/check | Rerun result |
| --- | --- |
| Environment | Existing repository-local `llm-extraction-defense` Conda, native Windows, Python3.11.18 |
| PyTorch / runtime / cuDNN | 2.7.1+cu128 /12.8 /90701 |
| Hardware / driver | RTX4060Ti, driver617.42, compute capability8.9 |
| CUDA and BF16 | Both available; actual GPU16x16 matrix product exactly matched CPU, sum66,672,640 |
| Tiny CUDA check | 0.2777224 seconds; peak Torch allocation32.002441MiB; this is **not** model inference memory |
| CUDA-reported memory | Total8,187.5MiB; free before tiny computation7,075MiB |
| NVIDIA driver memory | 4,398MiB free in CUDA check's driver snapshot; subsequent preload snapshot4,406MiB free /3,544MiB used |
| Conservative gate | 2,944.401367MiB BF16 parameters +1,024MiB runtime allowance +1,024MiB free reserve =4,992.401367MiB required |
| Teacher model loaded / attempts | No /0 smoke,0 confirmation,0 total |
| Run wall time | 18.4681155 seconds, includes hashes/data/CUDA/resource preflight; excludes registration, tests and literature work |
| Teacher workload runtime/peak allocation/reservation | **null**: workload never started |
| Accuracy, macro-F1, per-class/confusion/distribution/invalid rate, intervals and point decision | **null**: no confirmation predictions exist; no Task03 score substituted |

Windows WDDM CUDA and physical NVIDIA driver readings differ; the gate used the measured **driver** free memory, not the larger CUDA reading. The failure is the conservative preload condition, **not an observed OOM** or proof that this model can never fit8GB. Do not reduce the historical allowance or change precision/settings to manufacture a pass. No retry was made when memory might later change. Technical model-smoke/inference compatibility under this run remains unverified. Protocol prose gives4,992.375MiB as an illustrative total; the committed code's exact formula and measured run use4,992.401367MiB. This small arithmetic transcription discrepancy does not alter the decision; the original pre-execution protocol is preserved.

## Frozen identity and data verification

Teacher/tokenizer `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; all nine pinned local files, including weights, tokenizer and license, were rehashed successfully against Task03. Definitions template SHA-256 `32c07a68858577e777521a1b7b7976b94a5e284d1cebe1161462ff3f9af2595c`; native chat identity and per-input rendered hashes are separate. Strict normalizer source hash is frozen separately from the scoring-rule identity. No remote code is trusted, and no alternative prompt, scoring rule or Teacher was tried.

Dataset `PolyAI/banking77` revision `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`; existing CC BY4.0 provenance/attribution remains intact. The curator loaded the pinned dataset, excluded all900 prior IDs/canonical hashes and scanned subsequent raw ledgers, including failed/smoke runs. Exactly88 unique rows, eleven per approved intent, were frozen with seed17. Native rendering verified all input lengths<=512 before GPU work. Source77train/11test, with per-class counts recorded. Config SHA `14a5bb8777b8e18a043a00530107cfbc9f093065db0559fa6ecb6d90f0788b31`; input-manifest SHA `2c6c64bc6f232577e5135b7ae6a2faf1ab8084a51e6365b0d4d4aa252dab5aa6`. Run metadata also records protocol/template/model hashes, immutable revisions, seed and decoding.

Because zero rows were queried, all378 remain pipeline-unused, with88 reserved and290 unassigned. No row was falsely marked observed. After any future attempt, its ID is excluded even on an inference error. Canonical dedup does not establish absence of paraphrases or model pretraining overlap. The reserved88 cannot silently become a future independent model-selection test.

## Checks rerun now versus historical evidence

The pre-execution audit contains72 check records. The strengthened final `task04_historical_audit_final.json` contains**78** records; `unavailable=[]` for these checks. Recomputed38 historical metric aggregates: six Task02 gold-label matrices, twenty Teacher conditions and twelve Student/method/subset matrices. Final validation checks accuracy, eight-class macro-F1, per-class values and distributions where published. Sparse omitted zero cells in Task02 were compared as zeros, not treated as scientific discrepancies. An initial CPU-only development check incorrectly compared sparse and dense dictionary representations; the comparator was corrected, with no historical value/output rewritten and no inference consumed.

Rerun raw checks cover:

- Task02 input/output SHA and normalization for500 extraction,200 test and100 legitimate responses;800 main query records plus8 separate smoke queries.
- Exact frozen row order, raw-normalized labels, gold-label confusion and before/after paired agreement on historical test/legitimate cohorts.
- All twenty Task03 Teacher prediction files aligned to the same88 development rows and reproduced their published confusion matrices.
- Task03 Student saved candidate token lengths, log-likelihood sums, means, argmax labels and paired old-Teacher agreement; four score files,600 inputs/4,800 candidate scores.
- Teacher ledgers880 starts/finishes each, Student ledgers300 each, unique attempt identities and matching starts/finishes; no regeneration.
- All51 Task03-recorded protected Task01/02 file hashes, three frozen Task03 input/prompt/config hashes and eight saved adapter-file hashes. Nine stronger-model files were rehashed during Task04 preparation/execution.

Current local availability/hashes for**101 files across four historical run directories** are recorded in `task04_artifact_availability.json`. This new inventory proves present availability, not that every historical raw file had a trusted preexisting hash. Prior hash comparisons are limited to artifacts actually recorded in earlier manifests. The raw directories and adapter are available here; a fresh Git clone alone does not contain them. Missing artifacts in another checkout must block dependent raw verification; do not recreate responses/adapters and describe them as originals.

Historical-only claims: previous wall times, sampled GPU usage, training execution and outputs are **not replayed** by this audit. Task03's recorded executed source was `8ac2d51107a0ab735423df5d826c879e0b477efd`, while its final source includes the documented metadata correction. The original Student ledger's `prompt_sha256` was actually a scoring-rule hash. Preserve those bytes and `task03_provenance_notes.json`; the correct template was separately frozen. No numeric results or raw records were changed to repair that field. A confusion matrix against gold labels alone cannot reconstruct Student–Teacher agreement; actual paired rows were checked.

Historical interpretation remains: small Teacher is order-sensitive, rotations do not establish a universal last-option explanation;1.5B/definitions79/88 is a selection score; Student forced-choice accuracy19%→35%, generation agreement82% after training. Majority121/164 and copied-error97/164 match counts overlap. No defense evaluation exists. None of these historical outcomes substitutes for Task04 confirmation.

## Rebuild and tests

From the repository root with the existing Conda environment activated:

```powershell
$env:PYTHONHASHSEED = '17'
python -m src.task04.rebuild --run experiments/runs/20261029T071617Z_task04_confirmation_c2f6770c --output .cache/task04_rebuilt/review-copy
python -m src.task04.verify --run experiments/runs/20261029T071617Z_task04_confirmation_c2f6770c
python -m pytest -q
python -m pip check
```

Choose a new output directory on each rebuild; it refuses overwrites. Rebuild reads local ledger/predictions, frozen input/configuration and manifest, verifies raw hashes and attempted-ID alignment, enforces caps, and reconstructs nulls or metrics/intervals. Verify compares the JSON structurally and Markdown exactly against reviewed files, validates input hashes/class counts, and checks all75 historical tracked files against starting Git content, allowing only additive README/research-log updates and normal Git newline conversion. These commands do not load a model. Original executed source and later reporting source remain distinct and reviewable in Git.

Repository suite was run after the final aggregate-validation changes: **43 passed in7.50s**, including eight Task04 synthetic tests. They cover exclusions, deterministic selection, canonical duplicates, balance/integrity, budgets counting errors/invalids, denominator handling, tampered aggregate fields, complete-row reconstruction, and blocked raw-hash reconstruction. `pip check`: **No broken requirements found**. The offline verification actually matched reviewed outputs and checked75 historical files; no historical model output was rerun. Final verification metadata records the last execution and any later full-suite timing separately.

## Dependencies, provenance and limits

All139 Conda inventory entries match the Task02 version/build/channel inventory. New read-only snapshots: `task04_dependency_versions.json`, `task04_conda_win64.explicit.txt`; unchanged `environment.yml` remains authoritative for intent. The explicit Conda lock is platform-specific and excludes pip-installed packages; the captured runtime pip inventory and pinned existing environment specification supplement it. Local `file://` build paths in pip freeze are historical package metadata, not portable installation commands. No environment creation, installation, upgrade or base modification occurred. No new weights/datasets were downloaded.

Literature alone used public paper downloads into ignored `.cache/task04_literature/`. Existing Conda Python imported the bundled pure-Python pypdf6.10.0 read-only for text parsing; it was not installed in the research environment or used for model execution. PDF/text hashes and source versions are recorded. OpenReview CAP fetch failed403/browser verification; its full methods were unavailable, so novelty comparison remains explicitly incomplete. Papers and references were read, not experimentally reproduced.

Raw run manifests/texts/attempt ledgers, caches, model weights and adapters stay ignored; no credentials or raw responses are staged. Copying the repository alone cannot reproduce raw audits or rebuild predictions. Controlled local artifact transfer must retain file hashes, original executed-source commits, the one-shot claim and license attribution. The preparation/run failure is preserved with actual nulls. GPU contention, no independent final annotations/sessions, tiny confirmation size, known historical evaluation cohorts, source/label confounding and unresolved novelty/advisor review are remaining research limitations.
