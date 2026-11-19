# P14: Audio-only baseline on paired objects (RQ4)

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 14 | 11 JAN 2027

Roadmap note: Use matched photo/audio object IDs, not all museum images.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] Load the verified paired audio dataset and consistent image embeddings for the same objects.
- [ ] Extract CLAP and/or HuBERT features; verify recording identity, normalization and alignment.
- [ ] Run audio-only and image-only baselines with the same object subset and compatible clustering settings.
- [ ] Measure AMI/ARI plus audio-pair consistency and characterize sparse-pair/museum bias.

**DONE WHEN:** Audio-only and image-only reference results are comparable on the paired subset.

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P14_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
