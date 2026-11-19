# P3: Method comparison and blind k (RQ1)

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 8 | 30 NOV 2026

Roadmap note: Apply identical clustering/evaluation rules to shortlisted encoders.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] For each available backbone, run k-Means, Agglomerative, Spectral, GMM and HDBSCAN on matched dev embeddings.
- [ ] Use the same fixed-k reference (k=10); keep k=13 as a labeled-class-count reference, never a blind-k claim.
- [ ] Test silhouette, elbow, eigengap and/or BIC as appropriate; select k without inspecting HS labels.
- [ ] Save AMI/ARI/NMI, largest-cluster share, HDBSCAN noise fraction, seed variability and runtime for paired comparisons.

**DONE WHEN:** A per-backbone × method matrix and a blind-k results table can be reproduced.

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P3_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
