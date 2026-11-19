# P10: Run full image-preprocessing ablations (RQ2)

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 10 | 14 DEC 2026

Roadmap note: Evaluate image preparation on identical development objects.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] Complete raw/center crop, square padding and background removal on a controlled dev subset, keeping source images fixed.
- [ ] Audit segmentation for damaged instrument parts; discard or label failed masks before benchmarking.
- [ ] Extract comparable embeddings for the finalist encoders, then test clustering with identical seeds and metrics.
- [ ] Quantify whether improvement follows museum background suppression, changes in object size or crop artifacts.

**DONE WHEN:** The image-preparation decision is evidence-based and reproducible.

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P10_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
