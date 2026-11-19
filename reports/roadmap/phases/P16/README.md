# P16: One-shot held-out test (primary research milestone)

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 14 | 11 JAN 2027

Roadmap note: Run once, only after code and evaluation rules are frozen.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] Confirm the frozen source commit, checkpoints, and protected final-test manifest established or re-audited in Week 8 or Week 10; verify audio and near-duplicate exclusions.
- [ ] Run the frozen image pipeline once on the protected final-test objects (NOT any P5A validation subset), without reselection of k or settings.
- [ ] Compute primary HS3 AMI, ARI, NMI, class coverage, cluster balance and museum-aware diagnostics.
- [ ] Compare development, any Week 8-or-10 P5A validation, and final-test metrics using matching definitions; treat historical references separately when samples differ.
- [ ] Document unexpected failures or regressions; never retune on held-out and relabel the outcome as one-shot.

**DONE WHEN:** Signed-off held-out evaluation exists with exact code, inputs, timestamps and no test-driven retuning.

## Carried in from P0

- Reserved objects are untouched since the 15 Nov 2026 split, but they were part of pre-split exploratory analyses; see [`../P0/P0_REPORT.md`](../P0/P0_REPORT.md) §8 and [`../P0/artifacts/RESERVED_PRIOR_EXPOSURE.csv`](../P0/artifacts/RESERVED_PRIOR_EXPOSURE.csv). Describe the final test accordingly.

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P16_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
