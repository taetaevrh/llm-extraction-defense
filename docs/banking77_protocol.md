# Task 02 BANKING77 pilot protocol

The project owner replaced TweetEval with BANKING77 on 2026-10-29. The original
TweetEval stop report, configuration, manifest, and provenance remain historical
evidence. This document and `configs/task02_banking77.yaml` specify the resumed
experiment. No defense is implemented or evaluated.

## Attribution and licenses

BANKING77: Inigo Casanueva, Tadas Temcinas, Daniela Gerz, Matthew Henderson,
and Ivan Vulic (2020), *Efficient Intent Detection with Dual Sentence Encoders*,
PolyAI. [Publisher](https://github.com/PolyAI-LDN/task-specific-datasets),
[paper](https://arxiv.org/abs/2003.04807),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The pinned card and publisher README/LICENSE were reviewed before downloading
data; no conflicting additional terms were found in those sources. See
`banking77_provenance.json` for immutable revisions, URLs, and hashes. Changes
are eight-intent filtering, canonical-text deduplication, stratified splitting,
and teacher-generated labels for adaptation. Raw data is not redistributed.
Teacher/student model cards identify Apache 2.0; their previous revisions stay
fixed. Attribute all sources in later reports.

## Frozen task and subsets

Use exactly the eight labels in the approved config and `src/task.py`.
The curator loads the pinned dataset with ground truth for stratification and
evaluation. All filtering occurs before any query. Canonicalize text by NFKC,
case folding, punctuation removal, and whitespace collapse. Keep a single
canonical text globally, giving official test rows precedence, then train rows.
Record available unique counts and discarded duplicates. If any class cannot
meet quotas, fail before teacher inference. Distinct text hashes and source
row IDs must be disjoint across all selected subsets.

Use a local Python RNG seeded 17. Shuffle per-class pools in fixed label order,
allocate integer-balanced quotas (remainder to the first labels), then shuffle
each final subset. Reserve official test rows exclusively for the test set.
Extraction/validation use official train rows; legitimate examples use separate,
unused official test rows. An initial train-only allocation for all non-test
subsets was rejected before inference: lost_or_stolen_card was short by four
unique examples. The revised allocation fits the approved counts without
reusing text or changing the requested class balance.

| Subset | Size | Per-class quotas | Use |
| --- | --- | --- | --- |
| Extraction | 500 | 63 for first four labels, 62 for last four | Teacher-labeled adaptation only |
| Validation | 100 | 13 for first four labels, 12 for last four | Eight smoke prompts, one per class; remaining reserved |
| Test | 200 | 25 per label | Frozen primary evaluation before and after training |
| Legitimate | 100 | 13 for first four labels, 12 for last four | Independent unused official-test rows for utility evaluation |

Freeze row IDs, class quotas, text hashes, canonicalization, dataset revision,
and seed in `split_manifest.json`. The curator may use ground truth to select
balanced pools; the learner receives only selected text and teacher outputs.
This label-aware experimental curation is a limitation for claims about fully
autonomous attacker query selection. Exact canonical duplicates are excluded;
paraphrase/semantic overlap and pretraining contamination are not ruled out.

## Black-box interface and budgets

Freeze the exact prompt in `src/task.py` and save its SHA-256. Use native chat
templates for each model with the same user content, greedy decoding, batch 1,
max 256 input tokens and 12 new output tokens. The longer output cap than the
sentiment proposal accommodates underscore intent names. Reject oversized
inputs rather than silently truncating. No constrained decoding or fallback
classifier silently repairs model errors.

Accept only a whole exact allowed label, ignoring case, surrounding whitespace,
quotes/backticks, and one final period/exclamation. Explanations, multiple
labels, and other text are invalid and saved separately. Match valid labels
only; two invalid replies never count as agreement.

Teacher inference runs in a separate OS process with JSON text input/output.
Pinned artifacts use ordinary local files under ignored
`models/<repository>/<revision>/`; model loading is local-only. This avoids
Windows cache symlink privilege requirements. Downloaded files are hashed in
the run manifest; model revisions do not change.
Weights, tensors, logits, hidden states, and truth are not returned to the
training component. Every backend attempt is logged before budget accounting
can advance to another query, including invalid responses and errors. No
automatic retries or replacements for invalid outputs. Budget exhaustion
prevents another backend call. Trusted operator logs are outside learner inputs.
This is a local API simulation, not a security sandbox against an attacker
with unrestricted filesystem/process access to the research machine.

Primary extraction budget is **500 attempts**, including invalid/error results.
Additional evaluation ledger: 200 test queries and 100 legitimate queries.
Smoke uses 8 validation teacher queries and is recorded separately. A fully
completed first run therefore consumes 808 teacher calls, of which 500 are
extraction calls. A failure stops the run; all ledgers and failure records stay.

## Smoke gate and adaptation

Verify official CUDA PyTorch with a GPU matrix multiplication checked against
CPU before model use. Require BF16 support and at least 2,048 MiB free before
loading. Teacher and student never reside on the GPU at the same time.

Smoke: query one validation example per class, require eight valid teacher
labels, unload the teacher, load a fresh student, evaluate eight prompts,
and perform 16 LoRA optimizer steps on only those teacher-generated labels.
Require finite loss/gradients, at least one valid generated label after training,
and at least 1,024 MiB remaining VRAM after training. Record per-phase allocated,
reserved, and sampled device memory. Discard that adapter and unload the model.
These are technical feasibility checks, not an extraction-improvement test.
No smoke response enters pilot adaptation.

After the gate passes, collect 500 extraction responses and 300 evaluation
references with the undefended teacher, then terminate its process. Load a
fresh pinned pretrained student, evaluate the frozen test and legitimate sets,
then adapt using only valid extraction teacher labels. Training inputs have
two fields: text and teacher label. Ground truth, teacher references for held-out
prompts, and teacher internals are never arguments to the learner.

LoRA q_proj/v_proj, rank 8, alpha 16, dropout 0.05; base BF16, eager attention,
micro-batch 1, accumulation 8, three epochs, AdamW lr 0.0003, weight decay 0,
gradient clipping 1.0. Mask every prompt token with -100; supervise only the
teacher-label tokens and EOS. HF causal loss performs the next-token shift.
Group-average example losses, including the final incomplete accumulation group.
Reset seed 17 immediately before adapter initialization/training. No tuning or
early stopping on held-out outcomes. Save the local adapter outside Git.

## Evaluation, resources, and interpretation

Primary outcome is exact normalized student agreement with the same frozen
undefended teacher references on 200 held-out examples, before and after
adaptation. All 200 examples stay in each denominator. Also report teacher and
student ground-truth accuracy, invalid counts, class distributions/confusion,
and the same metrics on the 100 legitimate prompts. Utility includes end-to-end
teacher accuracy, valid coverage, served-only accuracy, no-defense denial rate,
and p50/p95 request latency.

Record preparation, smoke, and pilot wall time separately; GPU phases synchronize
at boundaries. Record peak Torch allocated/reserved memory per process/phase
and device usage sampled every 0.2 seconds (which may miss short spikes and
includes other applications). Worker resources measure teacher allocations;
controller allocation counters do not. Record failures and memory headroom.

One balanced eight-intent, 200-example test with one seed supports a feasibility
result only. It cannot establish full-LLM recovery, robustness across seeds,
or defense effectiveness. Improvement must be measured against the pretrained
student. A valid pipeline with no measured improvement is not extraction success.
