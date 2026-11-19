# P9: Select the finalist pipelines

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 10 | 14 DEC 2026

Roadmap note: First complete any Week 8-deferred P5A validation and split checks, then finalize model selection using predeclared development criteria.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] Rank pipelines on development results. If P5A was deferred, run it in Week 10 as a one-time new-data cross-check; do not repeatedly tune on the validation subset.
- [ ] Shortlist two or three defensible pipelines, including a simple k-Means reference and one strong candidate.
- [ ] Declare primary metric, tie-breakers, seed aggregation and k-selection policy in writing before final comparisons.

**DONE WHEN:** A timestamped shortlist and selection protocol are frozen.

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P9_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
