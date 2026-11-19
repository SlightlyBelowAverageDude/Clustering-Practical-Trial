# P2: Add several newer image encoders (RQ1)

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 8 | 30 NOV 2026

Roadmap note: Broaden beyond three old backbones while keeping the evaluation comparable.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] Check usable pretrained image embeddings, licenses and compute for DINOv3, SigLIP 2, EVA-CLIP and Qwen3-VL-Embedding.
- [ ] Run a smoke test on the same small dev sample for all four families: outputs, image-only mode, dimensions, finiteness and row alignment.
- [ ] Run at least two additional model families on all 12,000 development objects; target all four if compute permits, logging any deferrals.
- [ ] Use a common comparison arm for consistent normalization and clustering, and report each model’s native input resolution separately.
- [ ] Cache and hash embeddings; keep model name, exact checkpoint, resolution and image preprocessing in the results.

**DONE WHEN:** At least two new families have verified 12k results, with an explicit status for every candidate.

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P2_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
