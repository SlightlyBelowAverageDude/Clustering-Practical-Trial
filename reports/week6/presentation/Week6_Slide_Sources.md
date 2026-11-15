# Week 6 slide sources

Deck: `Week6_Clustering_StatusUpdate.pptx` (editable) and `.pdf` (LibreOffice export), A4 landscape, 14 slides. Generator: `reports/week6/source/build_deck.js`. Every number on a slide is read from `reports/week6/p4_validation/presentation_claims.json`, which `reports/week6/source/p4_verify.py` recomputes from saved assignment `.npy` files, representations and the phase-2 masks (299/299 checks pass; see `reports/week6/P4_VALIDATION_REPORT.md`). Paths below are relative to `reports/week6/`.

**Common context for all results:** MIMO/Europeana pilot of 2,000 images from 216 museums. HS3 is scored on 1,549 labelled rows in 22 classes (HS1 1,565; HS2 1,563), using the masks in `results/phase2/inputs/evaluation_masks.csv`. Development pilot, already explored; not held out.

| Slide | Content | Source files | Configuration |
|---|---|---|---|
| 1 Title | RQ wording | Week 4 deck (style reference) | — |
| 2 Progress | 816/816 reproduced; 2,000 / 216 / 1,549 / 22; 1,556 fits; 299/299 checks; 51/56 | `P1_BASELINE_RESULTS.csv`, `P1_DATASET_REPORT.json`, `experiments/runs.csv` (456) + `p3_experiments/robustness/runs.csv` (770) + `p3_experiments/preprocessing/runs.csv` (330), `p4_validation/checks.csv` | — |
| 3 Pipeline | Protocol, masks, k grid, seeds; plan vs pilot | `P1_EXPERIMENT_PROTOCOL.json`, `P3_EXPERIMENT_PROTOCOL.json`, `P1_FOUNDATION_AUDIT.md` | — |
| 4 RQ1 | Figure `figures/P2_METHOD_COMPARISON.png`; table of k=10 AMI/ARI; 0.544, 0.520, 0.360 vs 0.309, 51/56, HDBSCAN 58.7–69.8% noise | `P2_METHOD_COMPARISON.csv`, `P3_ROBUSTNESS_RESULTS.csv`, `experiments/paired_contrasts.csv` | centered + L2, k=10, seeds 0–4 (mean); agglomerative and HDBSCAN deterministic; HDBSCAN noise counted as one cluster |
| 5 AMI vs ARI | Figure `figures/P3_MIXED_CLUSTER_EXAMPLES.png`; 774 images (38.7%), 20 classes, purity 0.23, 61% of pairs, precision 0.33; purity percentages; 52/46 and 46/45 merges | `p3_experiments/ami_vs_ari/{summary,cluster_profiles,cluster_composition,mixed_cluster_examples}.csv` | SigLIP, centered + L2, k=10, **seed 0** (AMI 0.543, slightly different from the 5-seed mean 0.544) |
| 6 Blind k | Figure `figures/P3_BLIND_K_ROBUSTNESS.png`; P2 silhouette selections; −0.073 / −0.093; eigengap k 5–20, 5/2/3 split, gaps −0.020 / −0.038 | `P2_BLIND_K_RESULTS.csv`, `experiments/blind_k_selection.csv`, `P3_BLIND_K_ROBUSTNESS.csv` | k grid {2,3,5,8,10,15,20}; selections written before labels were read; subsets: phase-2 seeds 2026–2030, 1,600 rows each |
| 7 RQ2 | Figure `figures/P3_PREPROCESSING_ABLATION.png`; 0.03 → 0.46 (0.03 → 0.50), 99% → 24%, −0.009 / −0.022, +0.027 / +0.027 | `P3_PREPROCESSING_RESULTS.csv`, `p3_experiments/preprocessing/parity.csv` (240/240) | k=10, seeds 0–4, 5 phase-2 representations |
| 8 Limitations | 0.544 within 0.502–0.550; 52% aerophones; 34–39% catch-all; 191 pairs; ≈1.6 GB; ≈6 min | `P3_ROBUSTNESS_RESULTS.csv`, `P1_DATASET_REPORT.json`, `results/phase2/review/possible_duplicate_pairs.csv`, `p3_experiments/rq3/cpu_timing_random_weights.json` | 868 audio objects: **Week 4 / IDEAS.md count, not recomputed** |
| 9 Roadmap | Week 8–17 schedule | Week 4 deck schedule; `P4_NEXT_STEPS.md` | — |
| 11 Backup | Figure `figures/P3_AMI_VS_ARI.png` | `p3_experiments/ami_vs_ari/contingency_siglip_*.npy` | as slide 5 |
| 12 Backup | Figure `figures/P3_SPECTRAL_ROBUSTNESS.png` | `P3_ROBUSTNESS_RESULTS.csv` | 5 × 80% subsets, k=10 |
| 13 Backup | Figure `figures/P3_CLUSTER_DIAGNOSTICS.png` | `P3_PREPROCESSING_RESULTS.csv`, `experiments/assignments/` | k=10 |
| 14 Backup | Figure `figures/P2_BACKBONE_COMPARISON.png` | `P2_METHOD_COMPARISON.csv`, `experiments/paired_contrasts.csv` | k=10; cell counts over k grid |

**Not shown, by design:** the research plan's 0.53 (DINOv2 k-Means), 0.83 (linear probe) and 0.68 (target) are mentioned only as *not comparable* (different 27,348-object dataset, 13 HS3 classes, DINOv2, k = number of classes; not reproducible locally).

**Image rights:** the photos on slide 5 are MIMO partner-museum images under CC BY-NC-SA 4.0. Record IDs, museums and licence for each image are in `p3_experiments/ami_vs_ari/mixed_cluster_examples.csv`. The figure file lives in `reports/week6/figures/`, which, unlike `data/`, is not git-ignored; keep that in mind before publishing the repository.

**Style reference:** `/home/hayvan/Documents/LMU/Clusteringmeetsvisionandlangguage/W4-Presentation-StatusUpdate-TimHenninger.pdf`. Only the PDF was found; no editable `.pptx` exists. The new deck copies its general style (A4 landscape, dark title and closing slides, green accent, grey cards, numbered section titles, footer line) and was built from scratch. The original was not modified.
