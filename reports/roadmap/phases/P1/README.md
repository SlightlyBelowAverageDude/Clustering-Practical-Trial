# P1: Finish DINOv2 S/B/L at scale (RQ3)

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 8 | 30 NOV 2026

Roadmap note: Complete the model-size comparison before introducing newer families.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] Reuse already validated DINOv2-B embeddings at 224 px; rerun B only if settings or alignment differ.
- [ ] Extract DINOv2-S and DINOv2-L embeddings for the same 12,000 development objects, using matching 224 px settings.
- [ ] Measure runtime, peak memory, dimensions, failures and preprocessing for every size; record model checkpoint IDs.
- [ ] For supported models compare last and intermediate layers (-2, -4); cluster each with identical reference settings.

**DONE WHEN:** A matched DINOv2 S/B/L table with AMI/ARI and runtime, plus a layer-depth chart.

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P1_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
