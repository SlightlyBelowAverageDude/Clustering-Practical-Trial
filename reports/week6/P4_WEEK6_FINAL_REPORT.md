# Week 6 final report: Visual Form vs. Acoustic Function in Instrument Clustering

LMU practical course *Clustering meets Vision and Language*, Topic 1. Milestone: 16 November 2026. Every number here was independently recomputed (`P4_VALIDATION_REPORT.md`, 299/299 checks).

**Scope.** Everything below is from the **2,000-image MIMO/Europeana development pilot**: 216 museums; HS1/HS2/HS3 = 1,565/1,563/1,549 labelled rows; 22 HS3 classes. The pilot was already explored before this phase. These are descriptive development results, **not held-out estimates**, and no significance tests were run. The research plan's 27,348-object benchmark, DINOv2 baseline (0.53), linear probe (0.83) and target (0.68) are **not available or comparable locally**.

## 1. P1: foundation (`P1_FOUNDATION_AUDIT.md`)

- Audited the repository: data acquisition, two embedding caches (CLIP ViT-B/32 512-d, SigLIP ViT-B/16 768-d), five phase-2 representations, k-Means and agglomerative results, tests (11/11 pass).
- Recounted the dataset: 2,000 unique images, 216 museums, 0 exact duplicates, 191 unconfirmed near-duplicate candidates. Image IDs, embeddings and masks are aligned row for row.
- Reproduced all **816/816** saved metric rows exactly. The plan's 0.53/0.83 cannot be reproduced locally, and the "0.53 at HS3" baseline is ambiguous (the plan's own table gives 0.55 at HS3 and 0.53 at HS2).
- Declared the P2 protocol before any fit. Amendment A1 (float32 fits, for exact phase-2 parity) is documented, and the float64 run is kept as a sensitivity analysis.

## 2. P2: RQ1 methods, backbones, blind k (`P2_RQ1_FINDINGS.md`)

- 456 fits: 5 families (k-Means, agglomerative average-cosine, GMM diagonal, Spectral 15-NN, HDBSCAN) × 2 encoders × 2 representations × k ∈ {2,…,20} × 5 seeds. Labels were read only after fitting and k selection.
- Blind k: silhouette, elbow, BIC and eigengap.

## 3. P3: robustness, AMI vs ARI, preprocessing, RQ3 readiness

- **Robustness** (`P3_ROBUSTNESS_RESULTS.csv`, `P3_BLIND_K_ROBUSTNESS.csv`): 770 fits on the five phase-2 80% subsets (60/60 parity with phase 2).
- **AMI vs ARI** (`p3_experiments/ami_vs_ari/`): cluster × class composition, pair precision/recall, mixed-cluster examples with catalogue metadata and local thumbnails.
- **RQ2 preprocessing** (`P3_PREPROCESSING_RESULTS.csv`): 330 fits, 5 representations × 3 methods × k 5/10/20 (240/240 parity).
- **RQ3 readiness** (`P3_RQ3_READINESS.md`): code path, dimensions and CPU timing checked offline; weights not downloaded.
- Findings: `P3_RQ2_FINDINGS.md`.

## 4. Most important verified findings

| # | Finding | Evidence |
|---|---|---|
| F1 | **SigLIP beats CLIP** on identical rows and masks | 51/56 method × k × representation cells; at k=10 in 5/5 subsets for k-Means and Spectral, 4/5 for agglomerative |
| F2 | **The method ranking depends on the metric.** Spectral has the highest HS3 AMI; agglomerative has the highest ARI | SigLIP k=10: Spectral AMI 0.544 ± 0.001 / ARI 0.309; agglomerative 0.499 / 0.360; agglomerative ARI > Spectral in 10/10 subsets |
| F3 | The disagreement comes from **one large mixed Spectral cluster** next to many pure clusters | 774 images (38.7%), 20 classes, purity 0.23, 61% of same-cluster pairs; brass/lute/harp clusters 90–100% pure |
| F4 | **Visual form groups across acoustic function** (descriptive) | Reedpipes 52% + flutes 46% in one cluster (agglomerative 46% / 45%); small hand-held flutes, idiophones and rattles in one catch-all cluster |
| F5 | **Centering is decisive for average-link agglomerative only** | Raw L2 collapses (≈99% in one cluster, AMI 0.03 / 0.02); centered 0.459 / 0.499. k-Means −0.009 / −0.022. Spectral +0.027 at k=10, not at every k |
| F6 | **No reliable blind k yet** | Silhouette picks coarse k (Spectral k=5 loses 0.07–0.09). Eigengap selections range from 5 to 20 across subsets and beat silhouette in 5/10 |
| F7 | **HDBSCAN is not competitive** on these embeddings | 59–70% noise; noise-as-cluster AMI 0.12–0.37; its assigned-only score is matched or beaten by k-Means on the same rows (8/8) |

## 5. Best observed method and configuration (post-hoc, development only)

**SigLIP ViT-B/16 + centered L2 + Spectral (15-NN graph), k = 10: HS3 AMI 0.544 ± 0.001** (5 seeds), NMI 0.556, ARI 0.309. On 80% subsets it averages 0.520 (range 0.502–0.550), so the full-pilot value is near the top of its range. It was selected after examining development results; a held-out estimate needs the frozen pipeline on unseen data. By ARI there is no single winner: at k=10 several SigLIP configurations lie between 0.36 and 0.37 (k-Means on raw L2 0.370 or PCA50 0.369, agglomerative on PCA100+L2 0.369 or centered 0.360), all above Spectral (≤ 0.31).

## 6. Metric disagreement and robustness

- Spectral > k-Means (AMI) at k=10 in 9/10 encoder × subset cases and 43/50 paired seeds (SigLIP 25/25, CLIP 18/25); on the full pilot in 5/5 seeds for both encoders. Spectral > agglomerative (AMI) holds only for SigLIP (5/5), not CLIP (2/5).
- Spectral is **seed-stable but composition-sensitive**. Its seed SD is ≤ 0.005, but its subset SD (0.019–0.020) is the largest of the three methods. Partition agreement between subset fits and the full-pilot fit is lowest for Spectral (ARI 0.63 CLIP / 0.70 SigLIP).
- The headline conclusions also hold in the float64 sensitivity run, and the precision effect is ≤ 0.0065 AMI (k-Means only).

## 7. Remaining limitations

- The pilot is not held out and was explored. All "best" choices are post-hoc.
- The 27,348-object benchmark (13 HS3 classes), its held-out split and DINOv2 are missing locally, so the plan's 0.53/0.83/0.68 cannot be addressed.
- Class imbalance (aerophones 52% of HS1; 3 singleton HS3 classes) and dominant clusters. AMI and ARI disagree.
- Museum confounding: 1–14 images per museum; per-museum centering untested; no museum-level resampling.
- 191 unconfirmed near-duplicate pairs. Subsets overlap, so there are no confidence intervals.
- Blind k is unresolved. HS3's 22 classes include 9 with fewer than 10 images.
- Possible catalogue label noise; the human review (180 images) has not been done.
- Sparse paired audio (868 objects, Week 4 count, not recomputed) for RQ4.

## 8. Week 8 plan (details in `P4_NEXT_STEPS.md`)

Authorize the DINOv2 weights and run S/B/L plus layers −1/−2/−4 with a linear-probe reference. Extend RQ1 (Spectral graph sensitivity, GMM covariance, HDBSCAN on UMAP). Complete the RQ2 ablations (crop/padding/background, whitening, UMAP). Run museum- and duplicate-aware stability. Restore the 27k benchmark and its held-out split before the Week 10 freeze.

## 9. Readiness for 16 November

| Item | Status |
|---|---|
| Reproducible working baseline | Yes. All earlier and new numbers recompute exactly from saved assignments |
| Method comparison (5 families) | Yes, with subset robustness for k-Means, agglomerative and Spectral |
| Backbone comparison | CLIP vs SigLIP only. **DINOv2 missing** (weights not authorized) |
| Blind-k experiment | Yes. Full pilot plus subsets; outcome: no reliable criterion yet |
| Preprocessing (RQ2) | Yes. 5 representations × 3 methods |
| Defensible findings | F1, F2/F3, F5 (verified); F4 descriptive |
| Presentation | `presentation/Week6_Clustering_StatusUpdate.pptx` (editable) + `.pdf`; 9 talk slides + closing + 4 backup; speaker notes ≈ 10 min |

## 10. Output paths and reproduction

All paths are relative to `Clustering-Practical-Trial-main/reports/week6/`.

- **Reports:** `P1_FOUNDATION_AUDIT.md`, `P1_DATASET_REPORT.json`, `P1_BASELINE_RESULTS.csv`, `P1_EXPERIMENT_PROTOCOL.json`, `P2_RQ1_FINDINGS.md`, `P2_METHOD_COMPARISON.csv`, `P2_BLIND_K_RESULTS.csv`, `P3_EXPERIMENT_PROTOCOL.json`, `P3_ROBUSTNESS_RESULTS.csv`, `P3_BLIND_K_ROBUSTNESS.csv`, `P3_PREPROCESSING_RESULTS.csv`, `P3_RQ2_FINDINGS.md`, `P3_RQ3_READINESS.md`, `P4_VALIDATION_REPORT.md`, `P4_NEXT_STEPS.md`, `P4_WEEK6_FINAL_REPORT.md`.
- **Figures** (`figures/`): `P2_METHOD_COMPARISON.png`, `P2_BACKBONE_COMPARISON.png`, `P2_BLIND_K.png`, `P2_CLUSTER_DIAGNOSTICS.png`, `P3_SPECTRAL_ROBUSTNESS.png`, `P3_BLIND_K_ROBUSTNESS.png`, `P3_AMI_VS_ARI.png`, `P3_PREPROCESSING_ABLATION.png`, `P3_CLUSTER_DIAGNOSTICS.png`, `P3_MIXED_CLUSTER_EXAMPLES.png`.
- **Experiments:** `experiments/` (P2: runs, scores, 456 assignments, blind-k curves/selection, environment), `experiments_sensitivity_float64/`, `p3_experiments/{robustness,preprocessing,ami_vs_ari,rq3}/`, `p4_validation/`.
- **Presentation:** `presentation/Week6_Clustering_StatusUpdate.pptx`, `.pdf`, `Week6_Speaker_Notes.md`, `Week6_Slide_Sources.md`.

```bash
cd Clustering-Practical-Trial-main
.venv/bin/python reports/week6/source/p1_audit.py
.venv/bin/python reports/week6/source/p2_protocol.py              # refuses to overwrite
for s in fit select score; do .venv/bin/python reports/week6/source/p2_benchmark.py $s; done
.venv/bin/python reports/week6/source/p2_summarize.py
.venv/bin/python reports/week6/source/p2_figures.py
.venv/bin/python reports/week6/source/p3_protocol.py              # refuses to overwrite
for s in robust_fit robust_select robust_score preproc amiari; do .venv/bin/python reports/week6/source/p3_experiments.py $s; done
.venv/bin/python reports/week6/source/p3_figures.py
.venv/bin/python reports/week6/source/p4_verify.py
# deck: needs pptxgenjs (npm) and the pptx skill's apply_theme.js
NODE_PATH=<node_modules> node reports/week6/source/build_deck.js "$PWD" <path>/apply_theme.js
```

No commit or push was made. The repository is not a git repository. No model or dataset was downloaded.
