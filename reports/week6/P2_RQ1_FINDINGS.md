# P2: RQ1 — clustering methods, backbones and blind k (Week 6)

**RQ1.** Which clustering method and embedding backbone best recover the expert Hornbostel–Sachs (HS) classification, and can k be chosen without labels?

**Setting.** Validated MIMO pilot: 2,000 images from 216 museums, all clustered. HS3 is scored on the 1,549 labeled rows (22 classes, 13 of them with ≥10 images; largest class 18.5%). Two frozen encoders: CLIP ViT-B/32 (512-d) and SigLIP ViT-B/16 (768-d). Five clustering families. Two representations: `centered_l2` (primary) and `pca50_l2`. k ∈ {2, 3, 5, 8, 10, 15, 20}, five seeds, 456 fits, 0 errors, 120 s total fit time on CPU. The protocol (`P1_EXPERIMENT_PROTOCOL.json`) was declared before any fit. Labels are read only after fitting and k-selection.

**Read this first.** Every number below is a **descriptive development-set result** on a pilot that phase 1/2 had already explored. There is no held-out test. Grid cells share images and are not independent replicates, so no significance test was run and no significance is claimed. Choosing "the best method" from these tables is post-hoc.

## Main table: fixed predeclared k = 10 (HS3, mean ± SD over 5 seeds)

| Encoder | Repr. | Method | HS3 AMI | HS3 NMI | HS3 ARI | HS1 AMI | HS2 AMI | Largest cluster % (worst seed) | Cos. silhouette |
|---|---|---|---|---|---|---|---|---|---|
| CLIP | centered_l2 | k-Means | 0.440 ± 0.007 | 0.454 | 0.306 | 0.317 | 0.395 | 17.5 | 0.134 |
| CLIP | centered_l2 | Agglomerative (avg, cos) | 0.459 (det.) | 0.473 | **0.366** | 0.332 | 0.399 | 24.4 | 0.119 |
| CLIP | centered_l2 | GMM diag | 0.452 ± 0.006 | 0.466 | 0.262 | 0.321 | 0.389 | 28.8 | 0.114 |
| CLIP | centered_l2 | Spectral 15-NN | **0.470 ± 0.003** | 0.484 | 0.285 | 0.323 | 0.409 | 33.6 | 0.099 |
| SigLIP | centered_l2 | k-Means | 0.476 ± 0.017 | 0.489 | 0.344 | 0.348 | 0.428 | 13.9 | 0.157 |
| SigLIP | centered_l2 | Agglomerative (avg, cos) | 0.499 (det.) | 0.511 | **0.360** | 0.344 | 0.423 | 28.3 | 0.120 |
| SigLIP | centered_l2 | GMM diag | 0.463 ± 0.016 | 0.477 | 0.297 | 0.344 | 0.418 | 20.7 | 0.135 |
| SigLIP | centered_l2 | Spectral 15-NN | **0.544 ± 0.001** | 0.556 | 0.309 | 0.395 | 0.485 | 38.7 | 0.126 |
| SigLIP | pca50_l2 | k-Means | 0.493 ± 0.011 | 0.506 | 0.367 | 0.365 | 0.448 | 13.9 | 0.229 |
| SigLIP | pca50_l2 | Spectral 15-NN | 0.540 ± 0.000 | 0.552 | 0.298 | 0.399 | 0.484 | 39.4 | 0.179 |

HDBSCAN (deterministic; it picks its own k). Two scorings are reported:

| Encoder | Repr. | mcs | Clusters | Noise % | HS3 AMI, noise = own cluster (all 1,549 rows) | HS3 AMI, assigned rows only (coverage) | k-Means k=10 HS3 AMI on the same assigned rows |
|---|---|---|---|---|---|---|---|
| CLIP | centered_l2 | 20 | 5 | 67.4 | 0.269 | 0.689 (35.6%) | **0.772 ± 0.006** |
| CLIP | centered_l2 | 50 | 4 | 67.5 | 0.251 | 0.651 (35.4%) | **0.771 ± 0.002** |
| SigLIP | centered_l2 | 20 | 6 | 65.2 | 0.342 | 0.815 (37.8%) | **0.825 ± 0.007** |
| SigLIP | centered_l2 | 50 | 2 | 63.5 | 0.121 | 0.207 (39.9%) | **0.800 ± 0.003** |
| SigLIP | pca50_l2 | 20 | 6 | 58.7 | 0.373 | 0.777 (44.4%) | **0.798 ± 0.012** |

Full table, including pca50 and all k: `P2_METHOD_COMPARISON.csv`. HDBSCAN controls: `experiments/hdbscan_same_rows_control.csv`.

## Answers to the RQ1 questions

**1. Which method performed best?** By HS3 AMI, **Spectral clustering on a 15-NN graph** is highest at the predeclared k = 10 in all four encoder × representation settings. It beats k-Means in all 5/5 seed pairs in each setting: Δ = +0.030 and +0.023 (CLIP centered/pca50), +0.068 and +0.047 (SigLIP). Across the full k grid it beats k-Means in 11/14 (centered) and 10/14 (pca50) cells. **The ranking depends on the metric.** Spectral's ARI is *lower* than agglomerative's and k-Means' (SigLIP centered k=10: 0.309 vs 0.360 / 0.344). It also produces a dominant cluster (≈34–39% of images at k=10). So "Spectral is best" holds for AMI/NMI only. With 22 imbalanced classes, AMI and ARI reward different partition shapes.

**2. Did GMM, Spectral or HDBSCAN beat k-Means?**
- *Spectral:* yes, mostly (see 1). It is worse at k = 5 and 8 for CLIP (both representations), at k = 3 for SigLIP, and at k = 5 for SigLIP-pca50.
- *GMM (diagonal):* no consistent gain. It is level with k-Means on centered_l2 (6/14 cells better, mean Δ −0.002) and **worse in 14/14 cells on pca50_l2** (mean Δ −0.048, e.g. CLIP k=10: 0.401 vs 0.441, largest cluster up to 50%).
- *Agglomerative (average, cosine):* better than k-Means in 11/14 (centered) and 9/14 (pca50) cells, mean Δ ≈ +0.012. It also has the highest ARI.
- *HDBSCAN:* **no.** With noise counted as its own cluster, HS3 AMI is 0.12–0.37, far below every k-requiring method (≈0.44–0.54). The "assigned-only" score looks high (0.65–0.82), but HDBSCAN leaves 59–70% of images unassigned. On exactly those retained rows, plain k-Means k=10 scores **higher in all 8 configurations** (0.77–0.83). HDBSCAN's high assigned-only score comes from which rows it keeps, not from better clustering.

**3. Does SigLIP beat CLIP consistently?** Yes, on this pilot. Paired on identical rows and masks, SigLIP's HS3 AMI is higher in **25/28 (centered) and 26/28 (pca50) method × k cells**. At the seed level that is 101/112 and 106/112 paired seeds. Mean Δ is +0.036 and +0.040. By method: Spectral 14/14, GMM 14/14, k-Means 12/14, agglomerative 11/14. Of the 5 exceptions, 4 are at k = 2 and one is agglomerative at k = 8 (Δ −0.012). This agrees with phase 2 (38/40 cells) and extends it to three new methods. Caveat: this compares whole encoder bundles (architecture, patch size, data, loss and transform), so the gain cannot be attributed to the pretraining loss alone.

**4. DINOv2?** **Not available and not compared.** There are no local weights or embeddings, and the download was not made without approval. The research plan's DINOv2 numbers come from a different dataset (see P1 §1.4) and are not comparable.

**5. Which k did silhouette choose without labels?** Over the 16 encoder × representation × method curves, maximum cosine silhouette chose **k = 5 (3×), 8 (6×), 10 (6×), 20 (1×)**. It never chose 2 or 3. Other criteria: the k-Means elbow (Kneedle) chose 8 in 4/4 settings. GMM BIC chose 15 or 20. The spectral-graph eigengap chose 15, 15, 10 and 15. Full curves are in `experiments/blind_k_curves.csv`, and selections with HS scores are in `P2_BLIND_K_RESULTS.csv`.

**6. Was the silhouette choice meaningful?** Partly. The chosen k (5–10) lies between the HS1 (5) and HS2 (11) class counts, far below the 22 HS3 classes. Against the fixed k=10 baseline, silhouette-k HS3 AMI was higher in 4/16 cases, equal (k=10 chosen) in 6 and lower in 6 (mean Δ −0.014). Against the post-hoc best k (labelled *oracle, not blind*), it gives up on average −0.026 AMI. It fails worst for **Spectral**: it chose k=5, which costs −0.073 (CLIP) and −0.093 (SigLIP) against k=10. Spectral's own eigengap chose k=10 or 15 and lands within 0.004 AMI of the oracle k on average. That is a promising observation, but it rests on 4 settings and is a hypothesis to test in P3/P4. Silhouette values are low overall (0.07–0.23): this embedding space does not have well-separated cluster structure.

**7. Did any algorithm collapse?** Not at the declared k=10 anchor on `centered_l2`: the largest cluster is ≤ 39% for every k-requiring method (Spectral 34–39%). On `pca50_l2`, **GMM puts 47–50% of images in one cluster at k=10**. At small k the imbalance is severe: at k ≤ 3, **Spectral puts 76–89% of images in one cluster** and GMM 52–83% (worst seed). HDBSCAN is degenerate in another way: 59–70% noise, and SigLIP centered mcs=50 finds only 2 clusters (HS3 AMI 0.12). The raw-L2 average-linkage collapse (99.65% in one cluster) was documented in phase 2 and is why `centered_l2` is the primary representation here.

**8. Are the gains consistent across seeds?** Spectral is almost seed-invariant: pairwise seed-partition ARI is 0.988 on average (min 0.86) and the HS3 AMI SD is ≤ 0.005. k-Means is less stable (mean pairwise ARI 0.835, AMI SD up to 0.018). GMM is the least stable (mean ARI 0.753, min 0.51; AMI SD up to 0.024). The Spectral > k-Means and SigLIP > CLIP contrasts at k=10 hold in **every** paired seed. Seed agreement only measures initialization robustness, though. It does not measure sampling uncertainty. Phase 2's subset analysis (80% subsets, SD of seed-means 0.004–0.019) gives a rough sense of composition sensitivity for k-Means and agglomerative only. A further sensitivity source: running k-Means in float64 instead of float32 changed seed-mean AMI by up to 0.0065 (`experiments_sensitivity_float64/`).

**9. What remains untested?**
- DINOv2 (and DINOv3, CLIP-L/14) on the validated pilot.
- Any held-out or cross-museum evaluation. Every number here is from the development pilot.
- The larger 27k/59k benchmark.
- Spectral hyperparameters other than n_neighbors = 15 (affinity, k-NN size, `cluster_qr`); GMM with full or tied covariance; HDBSCAN `min_samples`, `cluster_selection_epsilon`, or HDBSCAN on UMAP-reduced data (an obvious P3 candidate given the noise rates).
- Spectral and HDBSCAN under subset resampling and chance permutation (phase 2 did these only for k-Means and agglomerative).
- Statistical uncertainty that accounts for museum or near-duplicate structure (191 unconfirmed near-duplicate pairs).
- Hierarchy-aware scoring against the HS tree.

## Observed results vs hypotheses

| Observed (on this pilot) | Hypothesis (not tested) |
|---|---|
| SigLIP > CLIP in 51/56 method × k × representation cells | The advantage holds on the 27k benchmark and for DINOv2 comparisons |
| Spectral has the highest HS3 AMI at k=10 in 4/4 settings, all seeds; ARI disagrees | The k-NN graph captures local, type-level similarity better than centroid methods |
| HDBSCAN leaves 59–70% as noise; its assigned-only advantage disappears on matched rows | Density in 512–768-d embeddings is too flat for HDBSCAN without dimensionality reduction |
| Silhouette picks k = 5–10, below the HS3 class count | The eigengap is a better blind-k criterion for graph methods |

## Comparison to the research-plan numbers

The best pilot result here (SigLIP + Spectral, HS3 AMI 0.544 at the predeclared k=10) is **not** a step toward the plan's "0.53 → 0.83 gap". Those numbers use DINOv2 on a different 27k-object dataset with 13 HS3 classes and k = number of classes. The method choice here is also post-hoc. The target AMI ≥ 0.68 on held-out data is not addressed by any Week 6 result.

## Figures (`figures/`, 220 dpi PNG, 13.3 in wide, for 16:9 slides)

- `P2_METHOD_COMPARISON.png`: HS3 AMI vs k for the four k-requiring methods, CLIP | SigLIP panels (centered_l2), with HDBSCAN points at their own k.
- `P2_BACKBONE_COMPARISON.png`: CLIP vs SigLIP at k=10 for every method (centered_l2 | pca50_l2), with cell-win counts.
- `P2_BLIND_K.png`: silhouette vs k with selected k ringed (top), and HS3 AMI at the same k (bottom).
- `P2_CLUSTER_DIAGNOSTICS.png`: largest-cluster share vs k (worst seed) and HDBSCAN noise/largest-cluster bars.

## Artifacts and reproduction

| Path | Content |
|---|---|
| `P2_METHOD_COMPARISON.csv` | Mean/SD per encoder × repr. × method × k (HS1/2/3 AMI, NMI, ARI, coverage, silhouette, sizes, noise, runtime) |
| `P2_BLIND_K_RESULTS.csv` | Per criterion: selected k, HS scores at that k, fixed-k=10 comparison, post-hoc oracle (flagged NOT_BLIND) |
| `experiments/runs.csv` | One row per fit (456): config, hyperparameters (JSON), seed, dtype, runtime, silhouettes, sizes, inertia/BIC, assignment sha256, status |
| `experiments/scores.csv` | 1,392 per-run × level × scoring-variant rows |
| `experiments/assignments/*.npy` | Every partition (2,000 rows, phase-2 order; −1 = HDBSCAN noise) |
| `experiments/cluster_sizes.csv`, `blind_k_curves.csv`, `blind_k_selection.csv`, `eigengap.csv` | Per-k diagnostics; blind selection written before scoring (`labels_accessed=False`) |
| `experiments/paired_contrasts.csv`, `seed_partition_stability.csv`, `hdbscan_same_rows_control.csv`, `phase2_consistency_check.csv` | Paired method and backbone deltas, seed stability, HDBSCAN control, exact phase-2 parity (216 rows, Δ=0) |
| `experiments/environment.json` | Python/package versions, source and protocol hashes |
| `experiments_sensitivity_float64/` | First (float64) run, kept as a precision-sensitivity analysis |

```bash
cd Clustering-Practical-Trial-main
.venv/bin/python reports/week6/source/p1_audit.py
.venv/bin/python reports/week6/source/p2_protocol.py        # refuses to overwrite the declared protocol
.venv/bin/python reports/week6/source/p2_benchmark.py fit    # no labels
.venv/bin/python reports/week6/source/p2_benchmark.py select # no labels
.venv/bin/python reports/week6/source/p2_benchmark.py score
.venv/bin/python reports/week6/source/p2_summarize.py
.venv/bin/python reports/week6/source/p2_figures.py
```
