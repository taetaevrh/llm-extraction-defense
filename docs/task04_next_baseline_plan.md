# Next baseline: preregistered plan, not executed

Prepared 2026-10-29. **Gate closed:** Task04 confirmation was blocked before model loading by the conservative memory preflight; it supplied no confirmation score or decision. Resolve GPU contention and obtain research review of a continuation of the same frozen confirmation before treating the stronger Teacher as approved. If its point criteria fail, stop model experimentation. This document authorizes no execution. No Task02 adapter is reused.

## Frozen task and access boundary

Eight-intent BANKING77, revision `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`, attribution and CC BY 4.0 recorded in `docs/banking77_provenance.json`. Strong Teacher Qwen/Qwen2.5-1.5B-Instruct and tokenizer `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; exact Task03 definitions template SHA `32c07a68858577e777521a1b7b7976b94a5e284d1cebe1161462ff3f9af2595c`. Greedy BF16/eager, 512-token cap with no truncation and twelve generated tokens; strict unchanged normalization. No search against confirmation/final evaluation.

Fresh Student for every arm/seed: HuggingFaceTB/SmolLM2-135M-Instruct and tokenizer `12fd25f77366fa6b3b4b768ec3050bf629380bac`. Keep model licenses/pinned hashes and dependency inventory. Teacher/Student modules are separate processes; unload Teacher and verify driver-free memory before adaptation, one model at a time. Preserve the measured 1 GiB memory guard and a predeclared conservative loading estimate. Smoke failures, OOM, truncation, missing data, or incompatible dependencies stop execution; no silent smaller sample, precision change, or parameter reduction.

**Input reuse decision:** use the original 500 extraction input IDs/texts in `docs/banking77_split_manifest.json`, once in original frozen order. Reuse is deliberate: it holds input acquisition fixed while changing the Teacher. These are historically observed inputs, not a fresh extraction-distribution study. Requery the stronger Teacher; never relabel old small-Teacher outputs as new outputs. A trusted curator originally balanced inputs with ground truth; this grants task-aware acquisition and must be reported as an assumption, not unrestricted real-world stealing. The learner sees only text and released Teacher label strings, with opaque IDs for accounting; ground truth remains in a separate evaluator/curator store. No logits, hidden states, weights or Teacher-derived confidence enter training.

## Queries and adaptation arms

One chronological transcript, exactly 500 **attempted** extraction calls. Persist each attempt before inference. Every invalid/error consumes budget; keep the transcript and train only on completed valid labels, with no replacement. A technical failure stops the run and reports incomplete status, rather than filling to 500 with different inputs. Record successful N, label counts, invalids and all failures. A reusable cache is keyed by Teacher, template, rendered-input, normalizer and decoding hashes, not just text. Cache hits across training seeds consume zero additional Teacher calls; 500 unique attempts shared across all six trained Students remain 500, not 3,000. Repeated actual calls count again. Diagnose duplicates and account for any technical retries explicitly under a separately approved budget; no hidden calls.

For seeds **17, 23, 42**, compare:

1. Fresh pretrained Student, measured before any adaptation; record per-seed duplicate deterministic outputs honestly (not three independent pretrained models).
2. Fresh Student trained on valid text–Teacher-label pairs.
3. Fresh Student trained on a seeded random permutation of those N Teacher labels across exactly the same texts, preserving their frequency multiset, example count, architecture, optimizer, epochs and seed pairing. Freeze and hash permutation before training; report fixed points and the fraction retaining the original label by chance. Do not use ground truth to choose the permutation.
4. Constant label = most frequent valid extraction Teacher label, ties resolved in the approved label order. Evaluate on every evaluation row; do not infer its test performance from the extraction distribution.

The shuffled-pair arm removes systematic input–label association in expectation while retaining label priors, prompt/response formatting, optimization and pretrained knowledge. It is **not a pure isolation of formatting**, and a single permutation can retain accidental associations. Three permutations/seeds remain a small control study.

## Fixed optimization proposal

Before any baseline inference, materialize these settings in a new config and commit its hashes, full manifests, permutations and protocol. These settings are a research-reviewed proposal, not a completed compatibility claim:

| Setting | Fixed choice |
| --- | --- |
| Adaptation | PEFT LoRA, causal-LM task; only q_proj and v_proj |
| Rank / alpha / dropout | 8 / 16 / 0.05 |
| Optimizer | PyTorch AdamW, lr 0.0003 (retain Task02), betas (0.9,0.999), epsilon 1e-8, weight_decay 0 |
| Schedule | Constant learning rate, zero warmup, max_grad_norm 1 |
| Training effort | Three complete epochs; microbatch1, accumulation8; update at every epoch's final partial accumulation with correct loss scaling |
| Precision / tokens | BF16; 512 input cap, target label plus native end token; 528 combined token cap; no packing or truncation |
| Randomness | Seed all Python/NumPy/Torch/CUDA/DataLoader paths; seeds17/23/42; zero DataLoader workers; fixed deterministic options |
| Checkpoint selection | Final epoch only; no early stopping or hyperparameter selection |

Use the same frozen classification content for Student training and generation, with its own pinned **native** chat rendering. Loss mask = -100 on all prompt/padding tokens, including user/assistant header; train only on the label response and end token. Verify target token boundaries, at least one unmasked response token, shifted causal loss, finite gradients, and changed LoRA parameters with base weights frozen. Shuffling label lengths can change target-token totals; report this residual, while matching rows, epochs and optimizer updates exactly. Implement a synthetic supervised smoke fixture and a small already-observed technical smoke, with separate accounting, before the 500-query run; no tuning based on held-out accuracy. Retain working dependency versions; incompatibility requires review.

For that separately reviewed run, freeze a technical smoke allowance of at most eight Teacher attempts on already-observed development inputs and sixteen Student optimizer updates on smoke-only copies discarded before fresh initialization. These eight are outside the500 extraction attempts and cannot silently be added to the attack transcript. With continuity evaluation300 and no other Teacher diagnostics, the complete planned maximum is808 actual Teacher attempts (8+500+300); any added final/calibration cohort requires a separate explicit evaluator budget. No new validation Teacher querying is needed when final-checkpoint selection is fixed; validation cannot trigger unregistered tuning.

## Evaluation and inference limits

Primary: strict free-generation Student–Teacher agreement on the **same frozen** held-out rows before/after, invalids incorrect; report Student and undefended Teacher ground-truth accuracy, eight-class macro-F1, confusion, class distributions and invalid rates. Use the fixed undefended Teacher reference throughout; never compare defended Students only to corrupted/withheld answers. Auxiliary forced choice uses the eight exact candidate labels, mean conditional token log probability with the existing audited rule; its results cannot replace generation or choose checkpoints. Compute pretrained/paired-control results for the same rows.

Retain the historical 200 test and 100 legitimate-user inputs only as **known-cohort continuity diagnostics**. They have already influenced research decisions and cannot establish independent generalization. New stronger-Teacher references require 300 **evaluator** calls, explicitly outside the500 extraction attempts; cache once across seeds, with identities and actual hits/misses. Do not expose them to the learner. The Task04 reserved88 are neither a tuning set nor a newly untouched future test; if eventually queried, they are observed confirmation data. A separately frozen independent final cohort is required for confirmatory defense conclusions; see the data plan. Until available, label the next baseline a feasibility/continuity study even if it improves.

Within each seed, report before–after agreement and accuracy differences, aligned-pair counts, McNemar discordances and descriptive paired class-stratified bootstrap95% intervals (2,000 resamples, seed17; same sampled rows for both predictions). Report paired association-arm minus shuffled-arm and majority/pretrained gaps. Across seeds show all three results, mean, range and SD; three seeds do not support a reliable population confidence interval over training randomness. A pooled row bootstrap alone ignores seed uncertainty; do not present it as resolving that uncertainty. Mark multiple endpoint/arm comparisons exploratory. No causal attribution of historical majority matches versus copied errors: those events overlap.

Log Teacher extraction/evaluator/smoke/diagnostic counts separately, Student generated examples and forced-choice candidate evaluations, all model/load/inference/training times, peak Torch allocated/reserved memory, minimum sampled driver-free memory, coverage and failures. A later defense comparison must match attempted budgets, denial rates and training settings, and use paired seeds/inputs where compatible with adaptive query selection. Record differing selected inputs when adaptation changes acquisition.

## Review decision required

Do not proceed merely because Task03's selected Teacher scored79/88 or because historical generation agreement reached82%. Review the blocked confirmation, future final data, actual pretrained output validity and control behavior. If paired training does not improve valid generation agreement over both pretrained and shuffled controls, report that negative result and diagnose pipeline validity before any defense test. Improvement is evidence of task-specific behavioral imitation only, not parameter recovery, full-LLM copying, or successful protection. No Student training was run in Task04.
