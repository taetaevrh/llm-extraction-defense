# Experiment records

No experiments have run. `manifest.template.json` defines a starting record,
not measured evidence. Future runs go into ignored `runs/<run-id>/` folders.

Each run must save:

- `manifest.json`: unique ID, status, UTC start/end, seed streams, command,
  Git commit, dirty status and diff hash, machine and dependency versions.
- `config.resolved.yaml` and SHA-256: all values concrete before execution.
- `environment.resolved.yml`, `conda.explicit.txt`, and later pip freeze if used.
- Model and tokenizer IDs and immutable revisions; adapter and base revision;
  precision/quantization configuration; license; dataset revision and checksums;
  split IDs, normalization rules, and prompt-template hash.
- Append-only `queries.jsonl` with query IDs, phase (extraction/evaluation),
  attempt index, tokens, denial/error status, latency, and response hashes.
  Cache authorized raw responses separately in the ignored run directory.
- `stdout.log`/`stderr.log`, resource samples, result path, failures and OOMs.

Never record tokens, passwords, full environment variables, or private prompts.
Evaluation calls count in their own ledger and cannot feed student training.
Create the run directory exclusively and fail on an existing run ID. A later
retry gets a new ID linked to the failed attempt. There is no runner yet.
