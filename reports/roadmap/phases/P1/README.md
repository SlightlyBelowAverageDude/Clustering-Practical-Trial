# P1: Finish DINOv2 S/B/L at scale (RQ3)

**Status: COMPLETE (2026-11-19).** Report: [`P1_REPORT.md`](P1_REPORT.md). Validation: 121/121 checks ([`artifacts/P1_VALIDATION.md`](artifacts/P1_VALIDATION.md)).

Milestone: WEEK 8 | 30 NOV 2026

Roadmap note: Complete the model-size comparison before introducing newer families.

## Roadmap tasks (verbatim from the execution roadmap)

- [x] Reuse already validated DINOv2-B embeddings at 224 px; rerun B only if settings or alignment differ.
- [x] Extract DINOv2-S and DINOv2-L embeddings for the same 12,000 development objects, using matching 224 px settings.
- [x] Measure runtime, peak memory, dimensions, failures and preprocessing for every size; record model checkpoint IDs.
- [x] For supported models compare last and intermediate layers (-2, -4); cluster each with identical reference settings.

**DONE WHEN:** A matched DINOv2 S/B/L table with AMI/ARI and runtime, plus a layer-depth chart.

## Contents

| Path | What |
|---|---|
| `P1_REPORT.md` | 12-section phase report |
| `artifacts/DINOV2_SBL_12K.csv` | Model-scaling table (layer −1; CLIP/SigLIP context rows marked as cached) |
| `artifacts/DINOV2_LAYER_ABLATION.csv` | S/B/L × layer −1/−2/−4 × 3 methods |
| `artifacts/DINOV2_RUNTIME_MEMORY.csv`, `DINOV2_MODEL_MANIFEST.json` | Cost, memory, checkpoints, hashes |
| `artifacts/figures/` | Four figures + `figure_values.json` |
| `artifacts/assignments/` | 99 saved cluster assignments |
| `artifacts/P1_VALIDATION.md`, `DINOV2_EXTRACTION_VALIDATION.md` | Independent verification evidence |
| `source/` | Extraction driver, clustering runner, figures, verifier |
