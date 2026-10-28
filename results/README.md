# Metrics

There are no experimental results. `metrics.template.json` contains null values
for unmeasured quantities. Store future JSON metrics under ignored
`runs/<run-id>/metrics.json`, linked to the experiment manifest.

Use fractions in [0, 1] for agreement/accuracy/denial/coverage, seconds for time,
MiB (2^20 bytes) for GPU memory, and integers for counts. Save numerator and
denominator, seeds, per-run values, and aggregation method. Report mean and
sample standard deviation across seeds and paired differences; give bootstrap
confidence intervals with method, seed, and resampling unit. Do not silently
exclude failed runs or replace missing values with zero.

Measure task performance against ground truth separately from agreement with
the teacher. Legitimate-user utility includes end-to-end correct-answer rate
(denials count as failures), coverage, conditional served accuracy, and latency.
Record peak allocated and reserved CUDA memory plus sampled device-wide usage
and sampling interval. Synchronize CUDA around timed regions; reset peak stats
per phase. These are measurement requirements for later implementation.

Only reviewed, small aggregate summaries with provenance belong in Git.
