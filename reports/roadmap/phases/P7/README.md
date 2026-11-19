# P7: Prepare paired-audio experiments (RQ4)

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 8 | 30 NOV 2026

Roadmap note: Prepare only: full fusion is scheduled for Week 14.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] Locate and audit audio-object mapping; verify whether the historically reported 868 usable paired recordings are available.
- [ ] Record duration, quality, licensing, missing audio and shared ensemble duplicates; create a gold pair manifest.
- [ ] Finish planned audio-object exclusions in the held-out manifest, without scoring held-out images.
- [ ] Define audio-only CLAP/HuBERT and later image+audio fusion experiments on exactly the same paired objects.

**DONE WHEN:** Audio pairing/exclusion evidence and a multimodal protocol are prepared.

## Carried in from P0

- The held-out audio-object exclusion is still pending (`reports/full_dataset/FULL_DATASET_REPORT.md` §5). It has to be finished before P5A locks its split.

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P7_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
