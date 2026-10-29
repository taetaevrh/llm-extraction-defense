# Task 03 frozen diagnostic protocol

Approved task: teacher diagnosis, evaluation audit, and next-data assessment.
No defense, new adaptation, next extraction pilot, or new independent-test claim.
Work and commits stay on main. Initial fetch found main and origin/main aligned.
Before the final push, fetch again; if remote main is not an ancestor of local
main, stop instead of merging, forcing, or rewriting history.

## Phase A: usage and data inventory

Audit the pinned BANKING77 dataset and all local Task02 BANKING77 response
artifacts, not merely the successful model's valid training records. Exclude
all 500 extraction-query rows, 200 test rows, and 100 legitimate-user rows.
The 100 validation reservations include eight actual smoke-training/query rows;
those eight are also excluded from diagnostic validation. Keep failed-run
metadata, canonical text hashes, and raw row IDs distinct from query counts.
Task02 had 900 assigned rows, of which 808 distinct rows were actually queried.

The frozen diagnostic set is **88 reserved validation rows, 11 per class**,
sampled deterministically with seed 17 after the smoke exclusions. It consumes
none of the remaining, previously unassigned pool. The 92 non-smoke validation
rows allow 11 per class; four first-class-remainder rows remain reserved.
See `task03_validation_manifest.json` and `task03_data_inventory.json`.
All 900 prior assignments stay excluded from any proposed new independent test,
a conservative rule that avoids silently laundering known or reserved rows.
Inventory all 77 classes, marking the approved eight-class scope. Global
canonical deduplication uses the same NFKC/case/punctuation/space rule as Task02.

## Phase B: position-bias intervention

Freeze exact templates, order, caps, IDs, and SHA-256 in
`configs/task03_prompts.json` **before any diagnostic inference**:

- Eight cyclic rotations; rotation_0 is byte-for-byte the original Task02
  template and is also the original-prompt condition. It is an explicit alias,
  not a second supposedly independent repeat.
- The reverse order of the original labels.
- One definitions prompt, describing every intent in plain English, with the
  original label order. This changes content as well as context length and is
  not an isolated position intervention.

There are ten unique prompts, 88 examples each: **880 attempts per Teacher**.
Every label occupies every position exactly once within the eight rotations.
Use identical native chat wrapping, deterministic greedy decoding, 12 output
tokens, BF16, eager attention, and the unchanged Task02 strict normalizer.
Simple permutations retain the 256-token input cap; definitions use 512 to fit
the descriptions. Oversized inputs fail rather than truncate or change prompts.
Reset seed 17 per configuration. There are no retries replacing invalid labels.

Report accuracy, eight-label macro-F1, per-class precision/recall/F1, invalid
rate, label distributions, complete confusion matrix including invalid
predictions, final-listed-option count/rate, runtime, query count, and memory.
Invalid predictions stay in denominators and are false negatives for their
true class; undefined precision/recall/F1 is zero. All eight labels enter the
macro average, including zero-recall classes.

The position hypothesis is exploratory. Summarize the label-by-position
selection matrix, final-option rate across the eight rotations, and prediction
changes relative to rotation_0. For an order-invariant classifier, the aggregate
final-option rate across all eight rotations is exactly 1/8 of valid outputs,
even if it always chooses one label. Report the 12.5% reference and a descriptive
95% percentile interval using 2,000 seed-17 bootstrap resamples: resample rows
within each of the eight true classes, retaining all rotations for each sampled
row. Do not treat 704 correlated responses as independent test subjects or
claim mechanistic proof from this intervention.

## Phase C: predefined conditional stronger comparison

Candidate Qwen/Qwen2.5-1.5B-Instruct revision
989aa7980e4cf806f80c7fef2b1adb7bc71aa306 is pinned. Its official card declares
Apache 2.0; card URL/hash and pre-download verification are saved in
`task03_stronger_teacher_provenance.json`.

Only if **no** small-Teacher prompt meets **all** provisional targets, test
the candidate with all the same ten prompts on the same 88 validation rows.
Do not choose its prompt using the old test/legitimate rows. Criteria:
accuracy >=70%, macro-F1 >=0.65, every class recall >0, and no single predicted
label >35% on this balanced set. Rank validation configurations by accuracy,
then macro-F1, then fewer invalids, then frozen model/prompt order. Report
selection-set performance as such; do not label it fresh test performance.
If none qualifies, stop further model search and extraction work.

Before downloading, use the actual 1,543,714,304 BF16 parameter count to estimate
weight bytes (two per parameter), add 1,024 MiB runtime allowance and **1,024
MiB required measured headroom**, and compare with fresh NVIDIA-SMI free
memory. This is an estimate, not a measured peak or guarantee. Stop if the
conservative budget fails. During load/inference, sample driver free memory
every 0.2 seconds plus CLI latency, check between attempts, and stop if any
measured sample falls below 1,024 MiB. Save Torch allocated/reserved peaks and
sampled device peaks/minimum free memory. Sampling can miss brief spikes.
Models run one per subprocess, with a CPU-only coordinator. No teacher tuning.

## Phase D: auxiliary Student classification

Use the unchanged original Task02 prompt and pinned Student/tokenizer.
Evaluate both the original Student and the saved Task02 LoRA adapter, loaded
with is_trainable=false; no optimizer or training step. Retain the same known
200 test and 100 legitimate examples explicitly as **retrospective diagnostics**.
Do not use these known outcomes for Teacher/prompt/scoring selection.

For each input, separately tokenize the rendered native-chat prompt and each
exact label as in Task02 response construction. Compute each label's sum of
conditional next-token log probabilities, then **select by mean log probability
per label token**, with FP32 log-softmax, equal class priors, no EOS/prompt/pad
score, and first-approved-label tie-breaking. This length normalization is
fixed before scoring; report sums and means in ignored per-input records.
Batch the eight label candidates together; mask right padding, disable cache,
and use inference_mode. Tests verify the causal shift and excluded prompt/pad
positions. Do not change scoring based on test outcomes or compare alternative
rules selectively. Length normalization is a heuristic and can affect labels.

Each model produces 300 input-level attempts and 2,400 candidate likelihood
evaluations; both models total 600 and 4,800 respectively. Each attempt gets
an append-only start/finish event, including failures. These forwards access
Student logits locally and **are not responses from the original text API**.
No new Teacher queries are needed for Student comparison: use frozen Task02
references. Report accuracy, macro-F1, agreement, distributions, and invalids
for the historical primary free generation beside the auxiliary forced choice.
Comparator: constant most frequent valid extraction-Teacher label from Task02,
chosen without looking at test labels. Decompose the existing 82% agreement
into majority-label matches and other matches; causal attribution to format
learning versus semantic transfer is not identifiable from this audit alone.

## Phase E and stopping boundary

No new test is created or queried. Assess remaining capacity conservatively.
The limiting approved class has 11 unassigned canonical texts across both
source splits, allowing at most 88 balanced examples; official-test-only
capacity is 16 (two per class). A second balanced 200-example test cannot be
formed under these exclusions. Even an 88-example mixed-source set would be
small and must be reserved once before further selection. Alternatives for
review include newly collected/independently annotated licensed in-domain
data or a new approved benchmark with aligned ontology; synthetic prompts
must be labeled as a separate diagnostic, not an independent real-user test.
Remaining data in the other 69 classes does not solve the same eight-intent
test problem without a research-approved task change.

Run the full test suite, record code/environment/prompt/dataset/model/adapter
hashes and all failures, preserve Task01/02 tracked files, save reviewed
aggregates only in Git, then commit and push main if remote remains compatible.
Stop for review after this diagnostic task. No next extraction or defense run.
