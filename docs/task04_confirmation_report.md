# Task 04 frozen Teacher confirmation

Run: `20261029T071617Z_task04_confirmation_c2f6770c`. Status: **blocked**.

Executed source: `995441350ee21cf90157a42aaf28ea3c5ec0a141`.

Metrics: null.
Reason: MemoryError: Conservative preload gate: 4406 MiB driver-free < 4992.40 MiB required; no Teacher loaded

Attempts: {"smoke": 0, "confirmation": 0}; total 0. Invalid/error attempts consume budget; no replacement or retry.
Wall time (seconds): 18.46811549999984.

Resource records:

```json
{}
```

Official source composition by class:

```json
{
  "card_arrival": {
    "train": 11
  },
  "card_not_working": {
    "train": 11
  },
  "change_pin": {
    "train": 10,
    "test": 1
  },
  "exchange_rate": {
    "train": 10,
    "test": 1
  },
  "lost_or_stolen_card": {
    "train": 8,
    "test": 3
  },
  "pending_transfer": {
    "train": 10,
    "test": 1
  },
  "transfer_timing": {
    "train": 8,
    "test": 3
  },
  "verify_my_identity": {
    "train": 10,
    "test": 1
  }
}
```

Unused inventory after this run:

```json
{
  "pipeline_unused_by_class_and_source": {
    "card_arrival": {
      "test": 2,
      "train": 77
    },
    "card_not_working": {
      "test": 2,
      "train": 35
    },
    "change_pin": {
      "test": 2,
      "train": 44
    },
    "exchange_rate": {
      "test": 2,
      "train": 36
    },
    "lost_or_stolen_card": {
      "test": 3,
      "train": 8
    },
    "pending_transfer": {
      "test": 3,
      "train": 74
    },
    "transfer_timing": {
      "test": 3,
      "train": 54
    },
    "verify_my_identity": {
      "test": 3,
      "train": 30
    }
  },
  "pipeline_unused_total": 378,
  "unassigned_after_confirmation_reservation_by_class": {
    "card_arrival": 68,
    "card_not_working": 26,
    "change_pin": 35,
    "exchange_rate": 27,
    "lost_or_stolen_card": 0,
    "pending_transfer": 66,
    "transfer_timing": 46,
    "verify_my_identity": 22
  },
  "unassigned_after_confirmation_reservation_total": 290,
  "balanced_pipeline_unused_maximum": 88
}
```

These rows were unused by this research pipeline at freezing, not proven absent from pretraining. The mixed official train/test source, canonical rather than semantic deduplication, eleven rows per class, and one frozen configuration limit inference. Attempted rows are observed; unqueried reserved rows remain unobserved. A blocked run supplies no confirmation evidence. Reserved/observed rows cannot silently become an untouched future model-selection test.
The 79/88 Task03 result was a selection-set score. Small-Teacher order sensitivity does not establish a general last-option explanation. Historical Student forced-choice accuracy rose 19% to 35% and generation agreement to 82%; majority/error matches overlap and do not identify causal shares. No defense has been evaluated.

Offline rebuild (no model loading): `python -m src.task04.rebuild --run experiments/runs/20261029T071617Z_task04_confirmation_c2f6770c --output .cache/task04_rebuilt/20261029T071617Z_task04_confirmation_c2f6770c`.
See task04_reproducibility_audit.md for checks rerun here, historical-only claims, hashes, raw-record locations and missing-artifact boundaries. No Student training or defense work followed this run.
