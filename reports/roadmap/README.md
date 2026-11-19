# Execution roadmap: phase index

Tracks every work package of the execution roadmap (Week 8 → Week 10 → Week 14 → Final, 19 Nov 2026 – 1 Feb 2027).
The roadmap itself is [`docs/roadmap/Clustering_Execution_Roadmap_W8_to_Final.docx`](../../docs/roadmap/Clustering_Execution_Roadmap_W8_to_Final.docx); it is the authoritative plan. This index records what has actually been done.

| | |
|---|---|
| **Current state** | P0 and P1 complete (2026-11-19). No other phase has started. |
| **Next permitted phase** | **P2** (Add several newer image encoders), after the team has reviewed the P1 report. |
| **Frozen dataset** | `MIMO-HS3-DEV12K-v1`: [`phases/P0/artifacts/DEV_MANIFEST.json`](phases/P0/artifacts/DEV_MANIFEST.json) |
| **Frozen protocol** | `FAIR-CMP-v1`: [`phases/P0/artifacts/EXPERIMENT_PROTOCOL.json`](phases/P0/artifacts/EXPERIMENT_PROTOCOL.json) |
| **Frozen baselines** | [`phases/P0/artifacts/BASELINE_REFERENCE.csv`](phases/P0/artifacts/BASELINE_REFERENCE.csv); DINOv2 S/B/L references: [`phases/P1/artifacts/DINOV2_SBL_12K.csv`](phases/P1/artifacts/DINOV2_SBL_12K.csv) |

## Phases

| Phase | Objective (roadmap name) | Milestone | Status | Report | Completed |
|---|---|---|---|---|---|
| **P0** | Freeze a fair comparison protocol | Week 8 | **COMPLETE** | [P0_REPORT.md](phases/P0/P0_REPORT.md) | 2026-11-19 |
| **P1** | Finish DINOv2 S/B/L at scale (RQ3) | Week 8 | **COMPLETE** | [P1_REPORT.md](phases/P1/P1_REPORT.md) | 2026-11-19 |
| P2 | Add several newer image encoders (RQ1) | Week 8 | NOT STARTED | [placeholder](phases/P2/README.md) | — |
| P3 | Method comparison and blind k (RQ1) | Week 8 | NOT STARTED | [placeholder](phases/P3/README.md) | — |
| P4 | Feature and image preprocessing (RQ2) | Week 8 | NOT STARTED | [placeholder](phases/P4/README.md) | — |
| P5 | Bias, leakage and robustness (RQ5) | Week 8 | NOT STARTED | [placeholder](phases/P5/README.md) | — |
| P5A | OPTIONAL in Week 8, alternatively Week 10 (new-data validation) | Week 8 | NOT STARTED (optional; Week 8 or Week 10) | [placeholder](phases/P5A/README.md) | — |
| P6 | Prepare background removal (RQ2/RQ5) | Week 8 | NOT STARTED | [placeholder](phases/P6/README.md) | — |
| P7 | Prepare paired-audio experiments (RQ4) | Week 8 | NOT STARTED | [placeholder](phases/P7/README.md) | — |
| P8 | Validate and present Week 8 | Week 8 | NOT STARTED | [placeholder](phases/P8/README.md) | — |
| P9 | Select the finalist pipelines | Week 10 | NOT STARTED | [placeholder](phases/P9/README.md) | — |
| P10 | Run full image-preprocessing ablations (RQ2) | Week 10 | NOT STARTED | [placeholder](phases/P10/README.md) | — |
| P11 | Cross-test robustness and debiasing (RQ5) | Week 10 | NOT STARTED | [placeholder](phases/P11/README.md) | — |
| P12 | Finalize blind k and method selection (RQ1) | Week 10 | NOT STARTED | [placeholder](phases/P12/README.md) | — |
| P13 | Freeze pipeline and prepare final evaluation | Week 10 | NOT STARTED | [placeholder](phases/P13/README.md) | — |
| P14 | Audio-only baseline on paired objects (RQ4) | Week 14 | NOT STARTED | [placeholder](phases/P14/README.md) | — |
| P15 | Image+audio fusion and VLM descriptions (RQ4) | Week 14 | NOT STARTED | [placeholder](phases/P15/README.md) | — |
| P16 | One-shot held-out test (primary research milestone) | Week 14 | NOT STARTED | [placeholder](phases/P16/README.md) | — |
| P17 | Validate and communicate Week 14 | Week 14 | NOT STARTED | [placeholder](phases/P17/README.md) | — |
| P18 | Build and test the demo (Week 16) | Final | NOT STARTED | [placeholder](phases/P18/README.md) | — |
| P19 | Final presentation and rehearsal (Week 16) | Final | NOT STARTED | [placeholder](phases/P19/README.md) | — |
| P20 | Final report, code and reproducibility (Week 17) | Final | NOT STARTED | [placeholder](phases/P20/README.md) | — |

Status values: NOT STARTED / IN PROGRESS / COMPLETE / PARTIAL / BLOCKED / DEFERRED. "Completed" is the date the report was signed off. It is left as "—" until that happens and is never filled in ahead of time.

P5A is optional in Week 8. It may run in Week 8 or be deferred to Week 10 (before P13), and it does not block P6, P7 or P8. If it is deferred, set its status to **DEFERRED (to Week 10)**, not COMPLETE.

## Rules for every phase

1. Every phase ends with `phases/<PHASE_ID>/<PHASE_ID>_REPORT.md`, written from [`PHASE_REPORT_TEMPLATE.md`](PHASE_REPORT_TEMPLATE.md) (12 sections).
2. Evidence goes in `phases/<PHASE_ID>/artifacts/` and scripts go in `phases/<PHASE_ID>/source/`. Large files (embeddings, weights, images) stay in the git-ignored `data/`, and only their paths and SHA-256 hashes are committed.
3. A report states the actual work, findings, interpretation, limitations and artifact paths, and it names every incomplete experiment.
4. Results are recomputed from saved outputs by a script before a phase is marked COMPLETE.
5. Each phase gets a completion decision, recorded here, before the next phase begins. The next phase is never started automatically.
6. Section 12 of each report (presentation-ready summary) is the only source for progress slides; see [`presentations/README.md`](presentations/README.md).
7. Commit to `Aiman-testing`. Never commit secrets (`.env`), raw images or model weights.
8. Reserved objects (`reports/full_dataset/benchmark/heldout_ids_DO_NOT_USE_FOR_SELECTION.csv`) are not embedded, clustered or scored outside P5A (validation portion) and P16 (protected final test).

## Updating this index

When a phase finishes, change its row: status, a link to `<PHASE_ID>_REPORT.md`, and the completion date. Then update **Current state** and **Next permitted phase**. Tick the matching checkboxes in the Word roadmap only for tasks the report marks Complete.
