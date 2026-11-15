# P1: Foundation audit and reproducibility (Week 6)

Audit date: 14–15 Nov 2026. Scope: `Clustering-Practical-Trial-main/`. The audit only reads earlier artifacts. Everything new is under `reports/week6/`. Script: `reports/week6/source/p1_audit.py`. Machine-readable outputs: `P1_DATASET_REPORT.json`, `P1_BASELINE_RESULTS.csv`, `P1_EXPERIMENT_PROTOCOL.json`.

**Verdict: PASS.** The pilot dataset, both embedding caches and the evaluation pipeline validate. All 816 saved metric rows reproduce **exactly** (|Δ| = 0) from saved assignments. P2 went ahead on this basis.

## 1.1 Status table

| Component | Implemented | Validated | Missing work |
|---|---|---|---|
| Data acquisition (Europeana/MIMO harvest, pilot sampling) | Yes: `spikes/pilot_acquire.py`, 63,478-record metadata snapshot, 2,000-image pilot | Yes: 2,000 image files present, all byte hashes match the sample CSV | Larger cleaned benchmark (59,661 photos / 27,348 scored objects) is **not present locally** (`runs/` absent) |
| HS label resolution (HS1/2/3) | Yes: Skosmos-resolved, conflicts → missing | Yes: recounted (see 1.2) | Expert/human review of the 180-image sample not done (template is blank) |
| Embeddings: CLIP ViT-B/32 (LAION-2B) | Yes | Yes: shape, finiteness, unit norm, sha256, row order | — |
| Embeddings: SigLIP ViT-B/16 | Yes (phase 2) | Yes: same checks | — |
| Embeddings: DINOv2 | Only in old spikes (`spikes/common.py`, timm `vit_base_patch14_dinov2`) | **No**: no weights, cache or embeddings on disk | Needs a ~350 MB weight download (not done without approval); then run through the validated pilot pipeline |
| Preprocessing (L2, centered+L2, PCA50, PCA50+L2, PCA100+L2) | Yes (phase 2) | Yes: 10 saved arrays rehashed; centering and L2 states confirmed | Whitening, UMAP and per-museum centering not on the validated pilot |
| Clustering: k-Means, agglomerative (avg, cosine) | Yes (pilot, phase 2) | Yes: exact metric reproduction | — |
| Clustering: GMM, Spectral, HDBSCAN | **No** before Week 6 | Added in P2 (`reports/week6/source/p2_benchmark.py`) | — |
| Evaluation (AMI/NMI/ARI, silhouette, masks) | Yes | Yes: 816/816 metric rows exact; 240/240 run silhouettes exact | Hierarchy-aware HS scoring |
| Blind k selection | **No** | Added in P2 | — |
| Chance control, seed and subset stability | Yes (phase 2) | Previously audited (`FINAL_VALIDATION.json`); not rerun | Museum- or specimen-level resampling |
| Reproducibility (manifests, hashes, env, protocol) | Yes | Yes: pilot snapshot (59 files) unchanged; phase-2 source and report hashes unchanged | Repository is not a git repo; no commit history |
| Tests | `tests/test_pilot.py`, `results/phase2/source/test_phase2.py` | **11/11 pass** | No tests for the new P2 methods (covered by the parity check instead) |
| Visualization | Phase-2 SVG/PNG figures, galleries | Present | New P2 figures added under `reports/week6/figures/` |
| Held-out split | **None exists** | — | Needed before any generalization claim (P4) |

## 1.2 Dataset and label verification (recomputed)

**Pilot `mimo_pilot_2000`** (`data/mimo_pilot/sample_pilot.csv`, sha256 `77ca3b43…`):

| Check | Result |
|---|---|
| Images / unique record IDs / unique image IDs | 2,000 / 2,000 / 2,000 |
| Unique byte sha256 / unique pixel sha256 | 2,000 / 2,000 (0 exact duplicates) |
| Image files present and hash-verified | 2,000 |
| Museums | **216** (1–14 images each, median 13); 7 museums have no HS3 label |
| Near-duplicate candidates (phase-2 screen, unconfirmed) | 191 pairs involving 208 rows; no exclusion applied |
| Records with multiple HS concepts | 73 |

| Level | Valid labels | Missing | Missing % | Classes | Largest class (share of valid) | Classes with ≥10 images | Singleton classes |
|---|---|---|---|---|---|---|---|
| HS1 | 1,565 | 435 | 21.75 | 5 | 4 Aerophones (52.1%) | 5 | 0 |
| HS2 | 1,563 | 437 | 21.85 | 11 | 42 (48.0%) | 8 | 1 |
| HS3 | **1,549** | 451 | 22.55 | **22** | 423 (18.5%) | 13 | 3 |

The previously reported counts (≈2,000 images, 216 museums, 1,549 HS3 labels) are **confirmed**.

**Alignment.** The pilot `image_order.csv`, the CLIP manifest inputs, the SigLIP manifest inputs, the phase-2 `image_order.csv` and the phase-2 `evaluation_masks.csv` (labels and row order) all match the sample CSV row for row. Ordered-image sha256: `07550be6…`.

**Larger benchmark (planned, not local).** The local data are the full 63,478-record metadata snapshot (`data/mimo_pilot/metadata.csv`; 41,141 HS1-valid, 40,824 HS3-valid records) plus the 2,000-image pilot. The 59,661-photo cleaned set and the 27,348-object scored subset come from `EXPLAINER.md`/`IDEAS.md`. Their IDs, images, embeddings and assignments are **absent**. Nothing in Week 6 substitutes the pilot for that benchmark.

## 1.3 Embedding caches

| Name | Shape | NaN/Inf | Row norm | Mean-vector norm | Hash |
|---|---|---|---|---|---|
| CLIP raw (`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`) | 2000×512 float32 | 0/0 | 1.000 | 0.702 (uncentered) | ok |
| SigLIP raw (`timm/ViT-B-16-SigLIP` @41f5757) | 2000×768 float32 | 0/0 | 1.000 | 0.771 (uncentered) | ok |
| {clip,siglip}_l2 | as raw | 0/0 | 1.000 | 0.70 / 0.77 | ok |
| {clip,siglip}_centered_l2 | as raw | 0/0 | 1.000 | 0.018 / 0.018 (centered, then re-normalized) | ok |
| {clip,siglip}_pca50 | 2000×50 | 0/0 | 0.44–0.78 / 0.37–0.71 (not normalized) | 0.0 | ok |
| {clip,siglip}_pca50_l2 | 2000×50 | 0/0 | 1.000 | 0.017 / 0.014 | ok; retained variance 70.4% / 68.2% |
| {clip,siglip}_pca100_l2 | 2000×100 | 0/0 | 1.000 | 0.018 / 0.016 | ok; retained variance 82.9% / 80.1% |

Raw embeddings are strongly uncentered (mean-vector norm ≈0.7). That explains the phase-2 finding that raw-L2 average linkage collapses into one cluster.

**DINOv2** is not available. A filesystem search found no weights or embeddings, and `~/.cache/huggingface` is empty. The extraction code exists only in the old spike pipeline. Running it would need about 350 MB of Hugging Face weights, which were **not downloaded without approval**. CPU inference for 2,000 images would take about 5–10 minutes (SigLIP took 253 s).

## 1.4 Baseline reproduction

`P1_BASELINE_RESULTS.csv` has 818 rows. The results below were recomputed from the saved `.npy` assignments, the saved representations and the exact phase-2 masks:

| Source | Runs | Metric rows | Max \|ΔAMI\| / \|ΔNMI\| / \|ΔARI\| | Silhouettes | Assignment hashes |
|---|---|---|---|---|---|
| `results/pilot` (CLIP, l2 & pca50_l2, k-Means ×3 seeds + avg-cosine, k=5/10/20/40) | 32 | 96 | 0 / 0 / 0 | n/a | n/a |
| `results/phase2/clip` | 120 | 360 | 0 / 0 / 0 | 120/120 exact | 120/120 ok |
| `results/phase2/siglip` | 120 | 360 | 0 / 0 / 0 | 120/120 exact | 120/120 ok |

Reference anchor (centered_l2, k=10, HS3): CLIP k-Means 0.4400 ± 0.0066, CLIP avg-cosine 0.4591, SigLIP k-Means 0.4762 ± 0.0166, SigLIP avg-cosine 0.4989. All of these reproduce.

**Research-plan references (not reproduced):**

| Value | Origin | Status |
|---|---|---|
| DINOv2 k-Means AMI **0.53** | `IDEAS.md`/`EXPLAINER.md`: DINOv2-B, padded, PCA50, k = number of classes, on the 27,348-object cleaned set | **Not reproducible locally**: dataset, embeddings and assignments absent. The plan's own tables give **0.55 at HS3** (13 classes) and **0.53 at HS2**. 0.53 at HS3 appears only in the depth table on 40,697 coded objects. The "0.53 at HS3" baseline is therefore ambiguous and should be checked before it goes on a slide. |
| Linear probe AMI **0.83** | Same source: 5-fold logistic regression, HS3 | **Not reproducible locally**: supervised reference from a different dataset |

These numbers come from a different, larger dataset (13 HS3 classes with ≥30 objects) with a different encoder and k = number of classes. They are **not comparable** to pilot AMIs (22 HS3 classes, many tiny). The pilot's best numbers must not be read as "closing the gap to 0.83".

## 1.5 Shared P2 protocol

`P1_EXPERIMENT_PROTOCOL.json` was written at 2026-11-14T22:52:35Z, **before any P2 fit**. The script refuses to overwrite it. Key points:

- Dataset: all 2,000 pilot rows in fixed order. Masks: phase-2 `evaluation_masks.csv` (HS1/2/3 = 1,565/1,563/1,549). No class filtering.
- Encoders: CLIP (512-d) and SigLIP (768-d), from the saved phase-2 arrays. Representations: `centered_l2` (primary; the phase-2 anchor) and `pca50_l2` (secondary).
- Methods and hyperparameters: k-Means (n_init 10); agglomerative (average, cosine); GMM (diagonal, n_init 3); Spectral (15-NN graph, kmeans assignment); HDBSCAN (min_cluster_size 20, 50).
- k grid: [2, 3, 5, 8, 10, 15, 20]. Fixed-k baseline: k=10. Seeds 0–4. Metrics: AMI, NMI, ARI, and cosine + Euclidean silhouette.
- Blind k: maximum seed-mean cosine silhouette (ties → smaller k). Secondary criteria: k-Means elbow (Kneedle), GMM BIC, and the eigengap of the spectral graph.
- Labels are read only in the `score` stage, after the `fit` and `select` stages have written their outputs.
- No held-out data exists, so none was accessed. All results are development-set results on an already-explored pilot.

**Amendment A1 (made after the first fit run, before any interpretation).** The first P2 run cast the saved float32 representations to float64. A parity check against phase 2 then showed different k-Means local optima in 15 of 180 overlapping runs (max |ΔAMI| 0.041; agglomerative exact). The reported run now fits k-Means, agglomerative, Spectral and HDBSCAN on the native float32 arrays, as phase 2 did. GMM is fitted in float64 for likelihood stability. Parity with phase 2 is now exact (216/216 rows, |Δ| = 0). The float64 run is kept as a numerical-precision sensitivity analysis in `experiments_sensitivity_float64/`. It changes k-Means seed-mean HS3 AMI by at most 0.0065 and 1 of 28 blind-k selections (CLIP pca50 Spectral silhouette: 10 vs 8). This shows that k-Means results on this pilot are sensitive to floating-point precision at about the level of the seed SD.
