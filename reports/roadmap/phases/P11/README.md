# P11: Cross-test robustness and debiasing (RQ5)

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 10 | 14 DEC 2026

Roadmap note: Check that the finalist is not simply learning museum or camera style.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] Audit and adjudicate near-duplicate candidates; rerun representative dev metrics after deduplication.
- [ ] Compare museum-aware dev resampling, class imbalance sensitivity and embedding/background mitigation options.
- [ ] Report AMI/ARI versus Museum-AMI, cluster-size balance and changes across controlled subsets.
- [ ] Retain negative results; do not choose a debiasing technique solely because it improves an evaluation label.

**DONE WHEN:** The remaining confounding risks and their observed effect sizes are documented.

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P11_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
