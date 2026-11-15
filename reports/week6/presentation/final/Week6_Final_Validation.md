# Week 6 final deck: validation

**Result: 224 / 224 independent checks pass** (`validation_checks.csv`, produced by `source/final_verify.py`). Every number shown in the deck and notes was traced to a verified source (§3).

## 1. Numerical checks (`source/final_verify.py`)

| Group | Checks | What was recomputed |
|---|---|---|
| Headline scores | 189 | HS3 AMI and ARI for CLIP, SigLIP and DINOv2-B; all methods, representations and k = 10 / 13. Recomputed from the saved cluster assignments (`reports/dinov2/dev/assignments/*.npy`) and the dev labels, then compared with `dev/summary.csv` (tolerance 1e-9) |
| User-supplied table | 15 | Every value of the requested headline tables: k=10 method × backbone AMI, HDBSCAN, ARI 0.417 / 0.311, raw-L2 and PCA100 agglomerative, k=13 0.584 (3-dp agreement) |
| Dataset counts | 13 | 63,478 / 63,473 / 59,658 / 40,694 / 27,347 / 12,000 / 14,696 / 651; 13 HS3, 129 names, 6 families. Recounted from the download ledger, `clean_records.csv` and `benchmark_objects.csv`, and cross-checked with `stats.json` |
| Isolation and configuration | 6 | dev ∩ held-out = ∅. The three dev embedding manifests cover exactly the 12,000 dev rows. No held-out embedding exists. DINOv2 dev = `timm/vit_base_patch14_dinov2.lvd142m` at **224 px**. 12,000 labelled HS3 rows |
| Rights | 1 | All benchmark records are CC BY-NC-SA 4.0 |

These checks come on top of the P5 verification (`reports/dinov2/p5_validation/`, 812 / 812). That run recomputed all dev and pilot scores, embedding and weight hashes, and row alignment.

## 2. Configuration consistency

- **Every main-slide performance number** is from the 12,000-object development benchmark: HS3, 13 classes, k = 10 (HDBSCAN chooses its own k), centered + L2 unless the slide names another representation.
- **DINOv2 on the main slides is DINOv2-B at 224 px.** The 518 px and S/L results (pilot) appear only in backup B5, labelled "exploratory, 2,000-image pilot".
- **The label-informed k = 13 reference** appears only in backup B2, labelled as not blind.
- **No pilot number or chart is used on the main slides.** No AMI is compared across the pilot and the development set.
- **No held-out claim.** No statement says the 59,658-photo collection was clustered; slides 2, 4 and 8 say the opposite explicitly.

## 3. Deck text audit

- Slide text and notes were extracted from the PPTX (`slide_text.py`).
- **All 83 decimal numbers match `final_claims.json`**, except the six schedule dates.
- **All thousand-separated integers match**, except 2,000 (the pilot size, a context statement in the notes) and 3,012 (instrument-part exclusions, from `reports/full_dataset/stats.json`).
- No placeholder text, `NaN` or `undefined`.

## 4. Visual QA

- The PPTX passes OOXML validation (`validate.py`). The PDF was exported with LibreOffice from the same PPTX: 15 pages = 15 slides.
- All slides were rendered and inspected. Fixed during QA:
  - an arrow overlapping the benchmark box and a wrapped chip label (slide 3);
  - crowded pipeline text and over-tall cards (slide 4);
  - a small chart leaving empty space (slide 7, re-laid out);
  - a wrapped card heading (slide 8);
  - a legend colliding with bar labels (backbone figure).
- Figures are placed with their native aspect ratio (no stretching).

## 5. Timing

The speaker notes plan **9:30** for slides 1–9: 0:40 + 1:00 + 1:10 + 1:00 + 1:20 + 1:20 + 1:10 + 1:00 + 0:50. This leaves 30 s of buffer within 10 minutes.

## 6. Image rights

The final deck contains **no museum photographs**; the pilot contact sheet from the previous version was removed from the main slides. All charts are generated from project data. The earlier deck and its attribution file (`reports/week6/figures/ATTRIBUTION.md`) are unchanged.

## 7. Reproduce

```bash
.venv/bin/python reports/week6/presentation/final/source/final_verify.py    # writes final_claims.json + validation_checks.csv
.venv/bin/python reports/week6/presentation/final/source/final_figures.py
NODE_PATH=<node_modules with pptxgenjs> node reports/week6/presentation/final/source/build_final_deck.js "$PWD" <pptx-skill>/scripts/apply_theme.js
```
