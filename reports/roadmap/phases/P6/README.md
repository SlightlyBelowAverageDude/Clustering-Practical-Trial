# P6: Prepare background removal (RQ2/RQ5)

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 8 | 30 NOV 2026

Roadmap note: Feasibility stage only, not a mass transformation of 12k images.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] Select a candidate segmentation pipeline (e.g., BiRefNet or SAM 2) and log model/checkpoint/license.
- [ ] Create a small audited image sample with long instruments, cluttered backgrounds and several museums.
- [ ] Generate masks and inspect failure cases: cut-off strings, keys, mouthpieces, transparent objects and residual shadows.
- [ ] Define a controlled raw vs padded vs cropped vs background-removed experiment for Week 10; freeze sample and metrics.

**DONE WHEN:** A segmentation feasibility report and ready-to-run Week 10 protocol exist.

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P6_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
