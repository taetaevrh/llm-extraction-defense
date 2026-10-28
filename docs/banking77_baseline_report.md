# BANKING77 no-defense pilot report

Date: 2026-10-29 (Asia/Bangkok). Completed one seed-17 pilot; no defense was implemented or evaluated.

Run ID: `20261028T195903Z_banking77_seed17_1a1274a4`. Executed source commit: `a11165eceadd161a50b983abd9cc7833efceeae1` on `task02/no-defense-baseline`, with a clean source working tree at run start.

The software pipeline and small-model GPU feasibility checks passed. Held-out behavioral agreement improved, but this pilot is **not yet a sound basis for defense-effectiveness claims**: the teacher is weak and heavily biased, and the pretrained student usually fails the required output format. No full-LLM recovery or general extraction-success claim follows.

## Immutable artifacts and attribution

| Role | Repository | Revision |
| --- | --- | --- |
| Teacher | Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Student | HuggingFaceTB/SmolLM2-135M-Instruct | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Dataset | PolyAI/banking77 | `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6` |

BANKING77 is attributed to Inigo Casanueva, Tadas Temcinas, Daniela Gerz, Matthew Henderson, and Ivan Vulic (2020), *Efficient Intent Detection with Dual Sentence Encoders*, PolyAI. [Publisher](https://github.com/PolyAI-LDN/task-specific-datasets), [paper](https://arxiv.org/abs/2003.04807), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The model cards declare Apache 2.0. License evidence and transformation notice were recorded before data use in [provenance](banking77_provenance.json).

## Primary held-out evaluation (200 examples)

| Measurement | Before adaptation | After adaptation |
| --- | ---: | ---: |
| Student-teacher agreement | 0.0% (0/200) | 82.0% (164/200) |
| Student ground-truth accuracy | 0.0% (0/200) | 37.0% (74/200) |
| Invalid student outputs | 197/200 | 0/200 |

Teacher accuracy: **47.5% (95/200)**, with 0 invalid outputs. All three evaluations use the same frozen row IDs and ground truth. Invalid outputs remain in denominators; two invalid outputs cannot agree.

## Class distributions on the held-out test

| Label | Ground truth | Teacher | Student before | Student after |
| --- | ---: | ---: | ---: | ---: |
| card_arrival | 25 | 8 | 3 | 1 |
| card_not_working | 25 | 12 | 0 | 10 |
| change_pin | 25 | 23 | 0 | 23 |
| exchange_rate | 25 | 0 | 0 | 0 |
| lost_or_stolen_card | 25 | 9 | 0 | 9 |
| pending_transfer | 25 | 15 | 0 | 6 |
| transfer_timing | 25 | 5 | 0 | 1 |
| verify_my_identity | 25 | 128 | 0 | 150 |
| invalid | 0 | 0 | 197 | 0 |

The teacher selected verify_my_identity on 128/200 test examples; the adapted student selected it on 150/200. Neither predicted exchange_rate on the test set. A descriptive constant-label comparator selecting the most frequent extraction-teacher label (verify_my_identity) would score 64.0% agreement and 12.5% task accuracy on this test. This is a post-hoc diagnostic, not a preregistered primary comparison.

## Separate legitimate-user subset (100 examples)

Teacher accuracy: 36.0% (36/100). Student accuracy: 0.0% -> 35.0%. Student-teacher agreement: 0.0% -> 83.0%. Invalid student outputs: 99 -> 0; teacher invalid outputs: 0.
Teacher valid coverage was 100%, denial rate 0%, and end-to-end/served-valid accuracy both 36%. Request latency p50 0.293s, p95 0.322s. These are public in-domain prompts, not measurements on real banking users.

## Query counts, splits, and training

- Exactly 500 extraction attempts: 498 valid teacher labels, two invalid outputs, no replacement queries.
- Exactly 200 test + 100 legitimate evaluation calls, plus eight separate validation smoke calls: **808 total teacher queries**. Earlier preparation failure consumed zero.
- Four frozen, stratified, disjoint subsets: 500 extraction and 100 validation from official train; 200 primary test and 100 legitimate from separate official-test rows. Three normalized duplicate train texts removed. The initial all-non-test-from-train allocation was rejected before inference and replaced because it lacked four lost_or_stolen_card examples. See [split preflight](banking77_split_preflight.json).
- Training artifacts contain only text and teacher-generated labels. Ground truth was used only by the trusted curator for stratification/evaluation; the learner received no truth labels or teacher internals.
- A separate-process teacher served text only. It was terminated before student GPU loading. The smoke adapter was discarded; the pilot began from the original pretrained student.
- LoRA rank 8, alpha 16, dropout 0.05, q_proj/v_proj; 460,800 trainable parameters; BF16 base; three epochs, micro-batch 1, accumulation 8, AdamW lr 0.0003. All prompt tokens masked; label and EOS tokens supervised.
- Completed 187 optimizer updates and 8610 supervised target tokens including EOS. Training losses/gradients were finite.

Frozen split SHA-256: `b418e182eb3549a0b10bf597e23f7dfb6b961224dd8ad1f07b2c791ef4f78849`. See [split manifest](banking77_split_manifest.json), [protocol](banking77_protocol.md), and [configuration](../configs/task02_banking77.yaml).

## CUDA, runtime, and peak memory

Official Windows CPython 3.11 PyTorch **2.7.1+cu128**, bundled CUDA **12.8**, cuDNN 90701, driver **617.42**, RTX 4060 Ti capability **8.9**. CUDA availability and BF16 support passed; a GPU matrix multiplication matched CPU exactly. The successful-run preflight observed 5,092 MiB free via NVIDIA-SMI after the computation (CUDA reported 7,075 MiB before it). Timing/measurement APIs and WDDM differ; both readings are retained.

| Phase | Wall seconds | Peak Torch allocated MiB | Peak Torch reserved MiB | Sampled whole-device used MiB |
| --- | ---: | ---: | ---: | ---: |
| Teacher load (worker) | 6.22 | 950.17 | 992.00 | 4045 |
| Teacher queries (worker) | 235.76 | 990.78 | 1030.00 | 4123 |
| Student pretrained evaluation | 284.76 | 327.07 | 358.00 | 3332 |
| Student training | 327.76 | 637.53 | 876.00 | 3925 |
| Student adapted evaluation | 107.94 | 328.83 | 876.00 | 3114 |

Pilot wall time: **966.21s (16.10 min)**. Successful-run preparation: 21.70s; smoke: 38.57s; total including those phases: **1026.49s (17.11 min)**. These totals exclude dependency installation and the earlier failed preparation attempt; cached downloads affect preparation time.

Highest sampled whole-device usage during the pilot was 4,124 MiB, including other desktop applications. Driver sampling waits 0.2 seconds plus command execution per sample and can miss short spikes; it is separate from process allocator peaks. Teacher worker allocations must not be inferred from the controller process counters. No OOM or runtime failure occurred in the completed pilot.

## Errors, verification, and scientific limits

1. The earlier run failed before inference with Windows cache symlink WinError 1314. It is retained in [failure evidence](../experiments/banking77_preparation_failure.json). Ordinary local model files resolved this without OS privilege changes or revision changes.
2. Qwen emitted a sliding-window/eager-attention warning. Its pinned config has use_sliding_window=false and sliding_window=32768; the frozen input cap is 256 tokens. The warning is retained in worker stderr; attention backend was not changed after seeing outcomes.
3. The pretrained student produced 197/200 invalid test outputs (and 99/100 legitimate outputs). The large agreement change therefore combines output-format learning with behavioral imitation; zero initial agreement does not imply zero general student capability.
4. Teacher utility is poor: 47.5% test accuracy, 36% legitimate accuracy, and strong verify_my_identity bias. Agreement can reflect imitation of teacher errors and class imbalance rather than useful capability.
5. One seed, eight balanced intents, 200 primary examples, 100 same-domain legitimate examples, public-data/pretraining overlap uncertainty, label-aware subset curation, and canonical rather than semantic duplicate filtering limit generality. No multi-seed significance or broad LLM recovery claim is supported.
6. The interface separation is a simulation, not an OS security sandbox. Public dataset/weight files remain on the trusted research host; the supplied learner code does not read them except for its own pinned student weights and approved training records.

Validation: **27 tests passed** (`python -m pytest -q`, 7.31s); pip dependency check passed. Independent post-run verification recomputed every accuracy/agreement/invalid count, checked ID alignment and query totals, verified 187 finite training updates, and rehashed the model artifacts without changes. See [verification](banking77_pilot_verification.json).

Reviewed aggregates and class confusion matrices are saved in [results](../results/banking77_pilot_summary.json); hashes/configuration/source revision are in the [reviewed manifest](../experiments/banking77_pilot_manifest.json). Raw rows, responses, caches, weights, and adapter remain ignored in the run/model directories.

**Research review required before continuing.** Pipeline feasibility and a measured task-specific agreement increase are established. Improve/review teacher utility and the pretrained output-format control, then consider replication across seeds before using this as evidence for defense research. No further experiment or defense was run.
