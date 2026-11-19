# P15: Image+audio fusion and VLM descriptions (RQ4)

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 14 | 11 JAN 2027

Roadmap note: Test whether sound or language resolves visually confusing cases.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] Evaluate weighted normalized concatenation and one consensus approach on the paired subset.
- [ ] Fix fusion weights using unsupervised or dev-only criteria; report full sensitivity curves without held-out tuning.
- [ ] Generate instrument descriptions with one open VLM on a defined subset; embed text and cluster descriptions.
- [ ] Compare image-only, audio-only, fused and description-based groups; inspect look-alike failures.

**DONE WHEN:** A multimodal comparison and concrete failure examples are documented.

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P15_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
