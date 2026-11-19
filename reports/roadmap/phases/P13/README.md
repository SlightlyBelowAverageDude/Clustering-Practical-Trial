# P13: Freeze pipeline and prepare final evaluation

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 10 | 14 DEC 2026

Roadmap note: Week 10 gate: finish any deferred P5A, check the protected split, then freeze the chosen pipeline.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] Freeze backbone, weights, input size, preprocessing, representation, method, hyperparameters and k rule.
- [ ] Confirm P5A was completed in Week 8 or Week 10, or explicitly document a limitation; re-audit the duplicate-aware, audio-excluded protected final-test manifest without reading test metrics.
- [ ] Lock code commit, model checkpoint hashes, environment, dev scores and reproducibility instructions.
- [ ] Specify Week 14 multimodal hypotheses and evaluation rules without changing the frozen image pipeline.

**DONE WHEN:** The Week 8-or-10 validation status is documented, pipeline choices are frozen, and the protected one-shot final test remains untouched.

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P13_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
