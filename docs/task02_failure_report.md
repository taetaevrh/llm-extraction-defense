# Task 02: stopped at dataset-license preflight

Date: 2026-10-29 (Asia/Bangkok). Branch: `task02/no-defense-baseline`.
Task 01 remains intact at parent commit
`7ee36b11b21bd23176e4e2c947067ae526134f64`.

**The baseline was not implemented or executed.** The user's explicit stop
condition for unsuitable or unclear licensing was reached before any package
installation, model-weight download, dataset-data download, or teacher query.

## Exact candidate revisions

These revisions were resolved through public Hugging Face APIs and checked
again at the immutable revision endpoints. Only metadata and README cards
were fetched. Tokenizer/model execution was not attempted.

| Role | Repository | Revision | License evidence |
| --- | --- | --- | --- |
| Teacher | Qwen/Qwen2.5-0.5B-Instruct | 7ae557604adf67be50417f59c2c2f167def9a775 | Model card and API identify Apache 2.0 |
| Student | HuggingFaceTB/SmolLM2-135M-Instruct | 12fd25f77366fa6b3b4b768ec3050bf629380bac | Model card and API identify Apache 2.0 |
| Dataset | cardiffnlp/tweet_eval, sentiment | b3a375baf0f409c77e6bc7aa35102b7b3534f8be | Repository metadata says unknown; card names CC BY 3.0 for sentiment and requires Twitter/API terms |

Public model cards: [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).
Pinned card URLs, card SHA-256 hashes, metadata endpoints, and checked-at time
are recorded in [task02_provenance.json](task02_provenance.json).

## Reason for stopping

The repository-level unknown tag alone is not the reason for stopping.
The [dataset card](https://huggingface.co/datasets/cardiffnlp/tweet_eval#licensing-information)
identifies a sentiment-specific CC BY 3.0 license, links a historical
[task-organizer response](https://groups.google.com/g/semevaltweet/c/k5DDcvVb_Vo/m/zEOdECFyBQAJ),
and requires compliance with Twitter and Twitter API terms. The
[upstream README](https://github.com/cardiffnlp/tweeteval#license) similarly
defers to individual dataset licenses and Twitter restrictions.

The current [X Developer Agreement, III.A(k)](https://docs.x.com/developer-terms/agreement)
restricts X API or X Content use for training or fine-tuning foundation or
frontier models. The proposed experiment adapts a pretrained LLM using tweet
text and teacher labels. I did not establish whether a permission, exception,
historical grant, or different interpretation resolves those referenced terms
for this particular off-platform dataset and research use.

This is an unresolved applicability question, not a categorical legal finding
that using TweetEval is unlawful. CC BY 3.0 is identifiable; its presence alone
does not resolve the separate terms incorporated by the dataset card for this
experiment. No project-specific rights evidence was available in the repository.
The user's instruction was to stop when licensing is unclear, so downloading
weights or continuing implementation after this finding would exceed that
boundary. No alternative dataset was substituted.

## Environment and measured preflight observations

The existing Conda environment was activated by name after setting the
session-local `CONDA_ENVS_PATH` to `.conda/envs`. Its interpreter is Python
3.11.18 under the repository-local environment prefix. PyTorch is still absent.
No package installation targeted either the project environment or base;
`environment.yml` and Task 01 dependency snapshots remain unchanged. Their
SHA-256 hashes are in the preflight manifest.

The GPU CSV probe reported RTX 4060 Ti, 8,188 MiB total VRAM, 6,123 MiB free,
driver 617.42, and compute capability 8.9. Free VRAM is an observation during
preflight, not a training-memory measurement. Official PyTorch Windows wheels
were researched, but a build was not installed or validated after the license
stop. `torch.cuda.is_available()` and a GPU computation were not run; framework
compatibility therefore remains unverified.

The existing test suite ran inside the activated environment:
`python -m pytest -q` returned **4 passed in 0.16s**. These are Task 01 scaffold
checks, not tests of normalization, leakage, budgets, metrics, LoRA, or model
output quality. No new pipeline behavior is claimed to be tested.

## Experiment status and saved records

- Approved pilot choices and candidate revisions:
  [task02_pilot.yaml](../configs/task02_pilot.yaml), explicitly non-executable.
- Environment observations, source-parent commit, stop reason, and test result:
  [preflight manifest](../experiments/task02_preflight_manifest.json).
- Unmeasured metrics, zero teacher-query attempts, and baseline-readiness status:
  [preflight results](../results/task02_preflight.json).

There are no frozen subsets, split manifest, teacher responses, training
examples, adapter, model evaluations, measured agreement/accuracy, or experiment
runtime/peak-memory values. Unmeasured results are null, not zero estimates.
No extraction or defense success conclusion is possible, and this baseline is
not yet evidence suitable for defense research.

## Research decision needed before resuming

Establish the permission/applicability basis for this dataset and planned use,
or approve a replacement three-class sentiment dataset with clear suitable
terms. A replacement changes the research configuration and needs the project
owner's decision. After resolving that gate, resume CUDA build validation,
implementation/tests, frozen splits, and the actual seed-17 experiment.
