# Initial experiment protocol (provisional; no execution authorized by this file)

Task 01 only creates the scaffold. Model/data choices, training parameters,
utility tolerance, and defense design require research review. No model
weights have been downloaded and no training or experiments have run.

## Initial conditions

| Condition | Response policy | Purpose |
| --- | --- | --- |
| No-defense baseline | Serve normal teacher responses; no added denials | Establish extraction feasibility and legitimate utility |
| Random response denial | Independently deny each attempt with a fixed, seeded probability, calibrated to the proposed defense's denial rate | Control for reduced access to teacher responses |
| Proposed defense | Undesigned; leave implementation and parameters pending | Later compare extraction reduction and utility trade-off |

Random denial is a control condition, not a new algorithm. No defense is
implemented now. Use the same teacher, student initialization, datasets,
attacker strategy, attempted query budget, training recipe, and paired run
seeds across conditions. Keep extraction and legitimate-user streams labeled
only for evaluation; do not give that label to the defense.

For the later comparison, select proposed-defense parameters on a separate
calibration split. Estimate its attacker-query denial rate there and freeze
that probability for random denial before any held-out test. Bernoulli denial
will match the rate in expectation, not exactly on each run: report realized
denials on both extraction and legitimate streams with uncertainty. Compare
legitimate-user denial/utility separately; do not tune using test outcomes or
quietly claim an exact match. If the design later requires exact-count masks
or matching legitimate denial rates as well, preregister that change first.

## Data and access

Use only an authorized teacher and appropriately licensed public/synthetic
task data. Select a classification-style text task with fixed labels and a
documented normalization function. The teacher interface exposes text only;
the extraction learner receives no teacher internals. Limit claims to this
interface/task and student capacity.

Freeze a versioned split manifest before querying: extraction pool, validation
and defense calibration, held-out test, and legitimate-user prompts. Deduplicate
across splits by content/hash and inspect near-duplicates. Keep held-out
prompts, teacher reference answers, and ground-truth labels out of extraction
and training. Tune hyperparameters and stopping rules on validation only.
Use separate legitimate prompts representative of the approved task. Record
whether the teacher/student pretraining data might overlap the benchmark;
unknown pretraining overlap is a limitation.

## Metrics and comparison

**Primary extraction metric: held-out teacher-student agreement.** For N frozen
test prompts, compare the extracted student's normalized task label to the
same undefended teacher's frozen reference label. Agreement = matching labels
/ N. Use the undefended reference for every condition so defense output changes
cannot change the target. An invalid, empty, or refused label from either side
is a disagreement, even if the raw strings match. Report invalid/refusal counts
and N. Finalize the normalization rules before generating test references.

Also record:

- Student task accuracy against ground truth; teacher task accuracy; the
  untrained student's accuracy and teacher agreement as controls.
- Legitimate-user utility: correct responses / all legitimate attempts, denial
  rate, coverage, served-only accuracy, and p50/p95 response latency relative
  to the undefended teacher. Fix an acceptable utility-loss threshold before
  choosing defense parameters. Denials count as utility failures.
- Query budget: every extraction attempt, including denials, malformed replies,
  retries, and errors, consumes one query. Report successful labels separately.
  Record input/output tokens and validation/evaluation calls separately; they
  cannot be reused for extraction or hidden in the extraction budget.
- Runtime by teacher collection, student training, and evaluation, plus total
  wall time; synchronize GPU work before recording boundaries.
- GPU memory by phase: PyTorch peak allocated/reserved memory after resetting
  counters, sampled total device memory (with interval), and OOM events.

Use paired seeds [17, 29, 43] and independent denial/data-loader RNG streams.
Save individual values, mean and sample standard deviation, paired condition
differences, and bootstrap intervals (candidate 2,000 resamples). State whether
the resampling unit is test prompts or runs; prompts alone do not characterize
training-seed uncertainty. Three seeds are a feasibility starting point,
not a strong statistical power claim. Predeclare comparisons and effect/utility
thresholds before testing a proposed defense. See the run and metrics contracts.

## Short plan for the first no-defense baseline

1. Review a small openly licensed teacher and a smaller pretrained student,
   initially in roughly the sub-billion-parameter range subject to an actual
   memory estimate. Choose one label-based language task. Freeze model,
   tokenizer, dataset revisions, split checksums, and prompt/label rules.
2. Select a compatible GPU PyTorch build inside the Conda environment, pin it,
   and verify import, CUDA availability, architecture support, and a tiny GPU
   tensor operation. Check free VRAM and estimate weights plus training state;
   test a tiny inference/training batch only after review. No task-01 model
   execution is included.
3. Candidate pilot: 500 extraction attempts, 200 held-out test prompts, and
   100 separate legitimate prompts; size validation/calibration separately
   before approval. Start with max 256 input tokens, 8 output tokens, greedy
   decoding, micro-batch 1, and seed 17. These values are proposals in
   `configs/baseline.yaml`; unresolved fields block future execution.
4. Cache normal teacher outputs to the extraction pool, unload the teacher,
   then adapt the student using only those outputs. Choose and freeze a
   feasible adaptation method, steps, learning rate, and gradient accumulation
   before execution. Keep the teacher and student off the GPU simultaneously.
5. Measure the untrained and extracted student on the same untouched test set;
   record agreement, ground-truth accuracy, utility, query counts, runtime,
   memory, and errors. Teacher reference/test calls use a separate ledger.
6. If the pilot is feasible, repeat with seeds 29 and 43 and review whether
   extraction improved over the untrained student before designing defenses.
   Save failures and do not change budgets/hyperparameters after inspecting
   test outcomes. Amend the protocol and start a new experiment if needed.

## Failure and resource rules

Start each run with at least a provisional 1,024 MiB free-memory safety margin
above the estimated peak. Measured memory, not parameter count alone, decides
feasibility. Stop on OOM, a violated budget, unresolved config/revisions, split
leakage, or unavailable required determinism; record status and reason. Any
reduced context length, precision, batch size, quantization, or CPU offload
creates a new resolved configuration and must be held constant across paired
conditions. Do not silently discard hard examples or failed seeds.
