# P4: Feature and image preprocessing (RQ2)

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 8 | 30 NOV 2026

Roadmap note: Change one factor at a time; retain a clean raw baseline.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] Compare raw L2, global centering + L2, PCA50/100 + L2 and UMAP on shortlisted backbones/methods.
- [ ] Run a limited padding vs center-crop experiment on the same dev image subset with identical encoder weights.
- [ ] Track instrument truncation, aspect ratio, cluster collapse and metric differences; preserve processed-image manifests.
- [ ] Only scale promising preprocessing choices after small checks; avoid mixing native-resolution effects with padding effects.

**DONE WHEN:** A controlled ablation table says what helps, hurts or makes no consistent difference.

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P4_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
