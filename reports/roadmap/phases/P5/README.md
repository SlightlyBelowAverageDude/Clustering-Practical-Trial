# P5: Bias, leakage and robustness (RQ5)

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 8 | 30 NOV 2026

Roadmap note: Measure confounders before attempting to remove them.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] Quantify Museum-AMI and the class distribution by museum; identify whether clusters reflect museum identity.
- [ ] Screen exact/near-duplicate objects using hashes and perceptual candidates; document unresolved human-review cases.
- [ ] Check object-level and near-duplicate leakage across development/held-out boundaries without viewing held-out model scores.
- [ ] Repeat selected fits across seeds, object/museum-aware dev subsets and class-balance sensitivity tests.
- [ ] Compare any simple mitigation (e.g., reweighting or global centering) to the same baseline; do not assume bias is removed.

**DONE WHEN:** A bias audit, leakage check and a robustness table are available.

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P5_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
