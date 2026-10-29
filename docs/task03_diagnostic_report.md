# Task 03 — Teacher diagnosis and Student evaluation audit

Date: 2026-10-29 (Asia/Bangkok). All work is on main. No new training, extraction pilot, or defense was run. Task01/02 artifacts and results remain unchanged.

Inference experiment: `20261029T042435Z_task03_diagnostics_932130fe`. Executed source: `8ac2d51107a0ab735423df5d826c879e0b477efd`. Postprocessing and one logging-field correction are part of the final Task03 results commit; original run records are preserved.

## Findings

The small Teacher is sensitive to label order, but the controlled rotations do **not support a general last-listed-option preference** as the explanation for its verify_my_identity collapse. Qwen2.5-1.5B with the definitions prompt passed every provisional validation target. Student forced choice exposes nonzero initial classification signal; the original 82% agreement remains heavily influenced by majority-label imitation and copying Teacher errors. No untouched balanced 200-example test remains in this eight-intent slice.

## Phase A — Usage and remaining data

The audit read the prior split manifest, protocol, code, report, results, adapter, and local response artifacts, including the failed preparation run. There were 900 assigned canonical examples: 500 extraction, 100 validation, 200 test, 100 legitimate. Of those, 808 distinct inputs were actually queried: all 500 extraction rows (including two invalid responses), 300 evaluation rows, and eight validation smoke rows. Smoke rows were used for technical Student adaptation even though that adapter was discarded; they cannot enter this Teacher validation set.

The frozen diagnostic set contains 88 of the 92 non-smoke reserved validation rows, exactly 11 per intent. It is disjoint from extraction, smoke, known test, and legitimate rows by ID and canonical text hash. No unassigned example was consumed. Old test/legitimate outcomes were used only for retrospective Student diagnosis, never Teacher/prompt/scoring selection. All 900 prior assignments remain excluded from any new independent test.

| Intent | Unused official train | Unused official test | Total |
| --- | ---: | ---: | ---: |
| card_arrival | 77 | 2 | 79 |
| card_not_working | 35 | 2 | 37 |
| change_pin | 44 | 2 | 46 |
| exchange_rate | 36 | 2 | 38 |
| lost_or_stolen_card | 8 | 3 | 11 |
| pending_transfer | 74 | 3 | 77 |
| transfer_timing | 54 | 3 | 57 |
| verify_my_identity | 30 | 3 | 33 |

The [inventory](task03_data_inventory.json) covers all 77 labels, prior assignments, observed queries, and diagnostic validation metadata. Untouched means unused by this research pipeline, not proven absent from model pretraining. Canonical deduplication does not rule out paraphrases. Full unused row metadata is retained only in the ignored inventory experiment directory.

## Phase B/C — Frozen Teacher comparison

Ten unique prompts were frozen before inference: original = rotation_0, rotations 1–7, reverse, and intent definitions. Original is an explicit rotation-zero alias, avoiding a redundant deterministic repeat. Every label occupies every position once across rotations. Every condition uses the same 88 reserved rows. Exact prompts/hashes, caps, ranking, and criteria are in the [protocol](task03_protocol.md), [configuration](../configs/task03_diagnostics.yaml), and [prompt manifest](../configs/task03_prompts.json).

| Teacher | Prompt | Correct / 88 | Macro-F1 | Invalid | Final-option rate | Targets pass |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen 0.5B | rotation_0 | 35 | 0.393 | 0 | 72.7% | no |
| Qwen 0.5B | rotation_1 | 23 | 0.243 | 0 | 0.0% | no |
| Qwen 0.5B | rotation_2 | 29 | 0.302 | 0 | 0.0% | no |
| Qwen 0.5B | rotation_3 | 34 | 0.389 | 0 | 6.8% | no |
| Qwen 0.5B | rotation_4 | 27 | 0.263 | 0 | 0.0% | no |
| Qwen 0.5B | rotation_5 | 39 | 0.371 | 0 | 10.2% | no |
| Qwen 0.5B | rotation_6 | 35 | 0.363 | 3 | 3.4% | no |
| Qwen 0.5B | rotation_7 | 36 | 0.376 | 0 | 4.5% | no |
| Qwen 0.5B | reverse | 35 | 0.335 | 1 | 0.0% | no |
| Qwen 0.5B | definitions | 50 | 0.534 | 1 | 43.2% | no |
| Qwen 1.5B | rotation_0 | 76 | 0.857 | 0 | 12.5% | yes |
| Qwen 1.5B | rotation_1 | 76 | 0.857 | 0 | 6.8% | yes |
| Qwen 1.5B | rotation_2 | 69 | 0.755 | 0 | 8.0% | yes |
| Qwen 1.5B | rotation_3 | 68 | 0.732 | 0 | 12.5% | no |
| Qwen 1.5B | rotation_4 | 70 | 0.757 | 0 | 12.5% | no |
| Qwen 1.5B | rotation_5 | 71 | 0.779 | 0 | 14.8% | yes |
| Qwen 1.5B | rotation_6 | 70 | 0.771 | 0 | 2.3% | yes |
| Qwen 1.5B | rotation_7 | 74 | 0.823 | 0 | 2.3% | yes |
| Qwen 1.5B | reverse | 68 | 0.731 | 0 | 0.0% | no |
| Qwen 1.5B | definitions | 79 | 0.897 | 0 | 12.5% | yes |

Every condition records per-class precision/recall/F1, complete confusion including invalid outputs, predicted distributions, invalid rates, query count, runtime, and memory in the [machine-readable summary](../results/task03_results_summary.json). Undefined precision/recall/F1 is zero; all eight classes enter macro-F1, and invalids remain in denominators and count as false negatives.

### Position sensitivity versus a last-option explanation

Small Teacher final-option selection across eight rotations: **86/704 = 12.2%**, paired/stratified 95% percentile interval **11.2–13.2%**, spanning the raw 12.5% reference. The three invalid rotation outputs give a valid-adjusted order-invariant reference of 12.45%; the excess interval is -1.26 to +0.76 percentage points and includes zero. The exploratory bootstrap resamples 88 input rows within class, retaining all eight responses per row, 2,000 times with seed 17; 704 correlated outputs are not treated as independent subjects.

Moving verify_my_identity through positions 1–8 yielded **59, 58, 48, 68, 64, 70, 76, 64** predictions out of 88. It remained dominant when first and was most frequent in position 7, not last. Rotations changed 8–30 predictions compared with the original. This supports order sensitivity and persistent label-identity/task bias, rather than a simple last-option account. Jointly rotating all labels cannot identify a unique internal causal mechanism or prove absence of every position effect.

Stronger Teacher final-option rate was 8.95%, interval 8.10–9.80%; it selected verify_my_identity exactly 11 times at every position. It still showed sensitivity on other labels. Full label-by-position matrices and paired changes are in the summary.

### Best candidate and memory fit

Best under the frozen ranking: **Qwen/Qwen2.5-1.5B-Instruct + definitions**, **79/88 = 89.8% accuracy**, **macro-F1 0.897**, zero invalids. Lowest class recall is 8/11 (72.7%); maximum predicted-label share is 13/88 (14.8%). All provisional targets pass. Small Teacher's best was definitions: 50/88 (56.8%), macro-F1 0.534; no small-Teacher configuration passed all targets.

These are model/prompt selection-set results, not unbiased new-test estimates. No Teacher was queried on old test/legitimate rows for this selection, and no Teacher was fine-tuned. Choosing the best of multiple prompts on 88 examples can be optimistic.

| Asset | Immutable revision |
| --- | --- |
| Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| SmolLM2-135M-Instruct | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| PolyAI/banking77 | `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6` |

The [official stronger-model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) declares Apache 2.0. Its card hash/revision and 1,543,714,304 BF16 parameters were verified before downloading. Estimated weights: 2,944.40 MiB; runtime allowance: 1,024 MiB; required measured headroom: 1,024 MiB; total conservative budget: 4,992.40 MiB. Fresh driver free memory was 5,770 MiB before download and 5,686 MiB before load. These were estimates/preflight readings, not inference peaks.

The stronger model **fit safely for this workload**: peak allocated **3,001.86 MiB**, reserved **3,194 MiB**, sampled whole-device peak **5,593 MiB**, minimum measured free **2,357 MiB**. No sample breached the 1 GiB guard. This is batch-one short-context inference, not a training-memory claim. Samples wait 0.2 seconds plus CLI latency and can miss brief spikes. Models ran one per subprocess with no simultaneous Teacher/Student GPU residency.

## Phase D — Formatting versus imitation

Historical free generation remains the primary text-API behavior. Forced choice is an auxiliary local-logit diagnostic: calculate each exact label string's conditional token log-likelihood, select by mean per-token log probability, exclude prompt/padding/EOS, equal class priors, fixed tie-breaking. The rule was frozen before scoring and not tuned. Length normalization has its own biases. Forced-choice predictions are not original API responses; zero invalids are guaranteed by the selection mechanism, not evidence of generation compliance.

Agreement below references the same frozen **old Task02 0.5B Teacher responses**, not the newly selected stronger Teacher.

| Known subset | Student | Method | Accuracy | Macro-F1 | Old-Teacher agreement | Invalid |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| test | pretrained | original free generation | 0.0% | 0.000 | 0.0% | 197/200 |
| test | pretrained | auxiliary forced choice | 19.0% | 0.127 | 8.0% | 0/200 |
| test | saved adapter | original free generation | 37.0% | 0.353 | 82.0% | 0/200 |
| test | saved adapter | auxiliary forced choice | 35.0% | 0.332 | 81.0% | 0/200 |
| test | constant comparator | extraction-Teacher majority | 12.5% | 0.028 | 64.0% | 0/200 |
| legitimate | pretrained | original free generation | 0.0% | 0.000 | 0.0% | 99/100 |
| legitimate | pretrained | auxiliary forced choice | 17.0% | 0.132 | 8.0% | 0/100 |
| legitimate | saved adapter | original free generation | 35.0% | 0.319 | 83.0% | 0/100 |
| legitimate | saved adapter | auxiliary forced choice | 35.0% | 0.324 | 84.0% | 0/100 |
| legitimate | constant comparator | extraction-Teacher majority | 12.0% | 0.027 | 73.0% | 0/100 |

The comparator always predicts verify_my_identity, selected from the 350/498 valid extraction-Teacher labels without test labels. The pretrained forced-choice test distribution concentrated on card_arrival (103) and lost_or_stolen_card (87), so 19% reflects limited classification signal, not a strong initial classifier. The adapted forced-choice model predicted verify_my_identity 154/200 times. Full method/model distributions and per-class metrics are in the summary.

Of the original **164/200 agreement matches**, **121/164 (73.8%)** were verify_my_identity matches; **97/164 (59.1%)** copied incorrect Teacher predictions, and 67 matched correct ones. There were also **43 non-majority matches**. Adapted generation's 82% agreement exceeds the constant comparator's 64%; under the unchanged auxiliary rule, accuracy rose **19% -> 35%**, macro-F1 **0.127 -> 0.332**, and agreement **8% -> 81%**.

Thus the apparent zero initial baseline was substantially format-confounded, and much of 82% reflects a biased Teacher and its errors. There is measured behavior beyond a constant majority response and format compliance alone. This audit cannot uniquely partition causal contributions of formatting, majority imitation, and semantic transfer, nor establish full-LLM recovery or general extraction effectiveness. No additional adaptation occurred.

## Phase E — Next clean evaluation

No new test was created or queried. Remaining examples permit at most **88 balanced mixed-source rows (11/class)** or **16 official-test-only rows (2/class)**. lost_or_stolen_card has just 11 unused canonical texts; a balanced 200-example test requires 25/class, at least 14 more in that class. Recycling old test/legitimate or prompt-selection rows would not make a new test independent.

Recommended next experiment for review: freeze Qwen1.5B/definitions, then perform a **no-training confirmation on new licensed, independently labeled in-domain examples**, ideally at least 200 balanced rows, with ontology, score/normalization rules, deduplication, and test lock fixed before querying. Collect all classes under a consistent process rather than supplementing only one from a new source, which could confound label and source. Withhold labels from any later extraction learner.

If new data is unavailable, explicitly approve either (a) the limited 88-row mixed-source holdout once, with wide per-class uncertainty and no claim equivalent to the 200-row design; (b) a new benchmark with aligned ontology; or (c) different unused BANKING77 intents, treated as a new task. Synthetic examples may be a separate diagnostic, not an independent real-user test. Remaining data in the other 69 labels cannot silently replace this same eight-intent evaluation. No next extraction pilot was launched.

## Reproducibility, resources, and errors

| Worker | Measured phase seconds | Peak allocated MiB | Peak reserved MiB | Minimum driver free MiB |
| --- | ---: | ---: | ---: | ---: |
| small_teacher | 312.08 | 996.27 | 1032 | 4471 |
| student_pretrained | 33.43 | 412.92 | 990 | 4616 |
| student_adapted | 36.37 | 414.68 | 990 | 4611 |
| strong_teacher | 140.17 | 3001.86 | 3194 | 2357 |

Diagnostic execution wall time: **638.62 seconds (10.64 minutes)**, including stronger-model download/process overhead, excluding protocol/inventory preparation and tests. Teacher attempts: **880 + 880 = 1,760**, with 1,755 valid and five invalid responses and no failed/replacement queries. Student attempts: **600 inputs / 4,800 label-candidate likelihood evaluations**. Every attempt has append-only start/finish records. There were no inference failures.

The existing Conda environment, Python 3.11.18, and PyTorch 2.7.1+cu128 were used without package changes or base installation. Seed 17, deterministic BF16/eager attention, frozen prompts/hashes/revisions, adapter hashes, row IDs, scores, and memory records are preserved. The existing Qwen eager/sliding-window warning remains in stderr; pinned configs disable sliding windows and contexts are short. No backend changed after outcomes.

A post-run audit found a metadata-labeling issue: executed Student ledgers stored the scoring-rule identifier hash in a field named prompt_sha256. The correct original prompt/hash was already frozen in the prompt manifest and protected Task02 source. An [additive note](task03_provenance_notes.json) supplies the mapping; source now records prompt and rule hashes separately. Raw ledgers were not overwritten, scores/predictions did not change, and no rerun occurred.

**35 tests passed** (`python -m pytest -q`, 6.01s); pip check found no broken requirements. Independent audit recomputed Teacher metrics, checked all attempt/candidate counts and scoring selection, rehashed protected historical files, saved adapter, and stronger-model artifacts, and verified memory headroom. See [verification](task03_verification.json), [reviewed manifest](../experiments/task03_diagnostic_manifest.json), and [results](../results/task03_results_summary.json).

Only reviewed aggregates, curated IDs/hashes, source, prompts, and docs enter Git. Raw texts, responses, score records, caches, and weights remain ignored. Public-data pretraining exposure, small validation size, single seed, prompt-selection optimism, and forced-choice access differences constrain interpretation.

**Stopped for research review.** No further training, extraction, or defense implementation/evaluation followed these diagnostics.
