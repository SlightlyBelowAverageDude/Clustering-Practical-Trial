# DINOv2 benchmark: CLIP vs SigLIP vs DINOv2 and RQ3 model scaling (P5)

The protocol (`DINOV2_PROTOCOL.json`) was declared at 2026-11-15T13:03Z, before any DINOv2 clustering fit. Independent checks: `p5_validation/` (see §6).

**Scope.** §2–§4 use the **2,000-image pilot**: the same rows and HS masks as Week 6, HS3 n = 1,549 in 22 classes. The pilot was already explored, so these are descriptive development results. §5 uses the **development partition** of the reconstructed benchmark: 12,000 objects, 13 HS3 classes. Scores from the two sets are **not comparable** (different rows and class sets). The held-out partition was never embedded, clustered or scored. No significance tests were run, and "best" configurations are post-hoc.

## 1. Models and compute

| Model | Checkpoint (timm, official LVD-142M weights) | Params | Dim | Blocks | CPU s/img at 224 / 518 px |
|---|---|---|---|---|---|
| DINOv2-S/14 | `timm/vit_small_patch14_dinov2.lvd142m` @ `4610ca1` | 22.1 M | 384 | 12 | 0.053 / 0.358 |
| DINOv2-B/14 | `timm/vit_base_patch14_dinov2.lvd142m` @ `4685c99` | 86.6 M | 768 | 12 | 0.148 / 1.083 |
| DINOv2-L/14 | `timm/vit_large_patch14_dinov2.lvd142m` @ `4741e1c` | 304.4 M | 1,024 | 24 | 0.566 / 3.55 |
| CLIP ViT-B/32 | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` (existing cache) | — | 512 | — | — |
| SigLIP ViT-B/16 | `timm/ViT-B-16-SigLIP` (existing cache) | — | 768 | — | — |

- **Weights:** revisions and sha256 hashes are in `models.json`. The weights stay in the ignored `data/dinov2_model/` (≈ 1.65 GB).
- **Compute:** CPU only (12 cores, no CUDA), fp32, 4–6 threads; clustering used 4 BLAS threads.
- **Input pipeline:** identical to CLIP/SigLIP (EXIF transpose, RGB, white-square padding), then the checkpoint's timm transform at 224 or 518 px (518 = timm default and the teammate setting).
- **Feature:** the CLS token after the final LayerNorm. For layer −1 it equals the pooled output; this was checked at run time, and the check caught one layer-ordering bug before any results were written.
- **Extracted on the pilot:** S, B and L at 224 px; S and B at 518 px. L at 518 px was not run (≈ 2 CPU-hours).

## 2. Three-backbone comparison (pilot, centered + L2, k = 10)

HS3 AMI (mean ± seed SD) / HS3 ARI / largest cluster (worst seed). Agglomerative and HDBSCAN are deterministic. HDBSCAN uses min_cluster_size 20, with noise counted as its own cluster.

| Backbone | k-Means | Agglomerative (avg, cos) | Spectral 15-NN | GMM diag | HDBSCAN (noise %) |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | 0.440 ± 0.007 / 0.306 / 17% | 0.459 / 0.366 / 24% | 0.470 ± 0.003 / 0.285 / 34% | 0.452 / 0.262 / 29% | 0.269 / 0.042 (67%) |
| SigLIP ViT-B/16 | 0.476 ± 0.017 / 0.344 / 14% | 0.499 / 0.360 / 28% | **0.544** ± 0.001 / 0.309 / 39% | 0.463 / 0.297 / 21% | 0.342 / 0.057 (65%) |
| DINOv2-S (518) | 0.466 ± 0.001 / 0.289 / 20% | 0.499 / **0.442** / 20% | 0.473 / 0.295 / 24% | 0.439 / 0.237 / 22% | 0.360 / 0.122 (54%) |
| DINOv2-B (518) | 0.495 ± 0.005 / 0.310 / 24% | 0.544 / **0.437** / 18% | 0.497 / 0.214 / 48% | **0.494** / 0.290 / 31% | 0.430 / 0.116 (57%) |
| DINOv2-L (224) | **0.508** ± 0.005 / 0.327 / 19% | **0.551** / 0.412 / 19% | 0.529 ± 0.001 / 0.261 / 45% | 0.467 / 0.246 / 32% | **0.473** / 0.175 (49%) |

**Reading the table**
- **DINOv2 improves k-Means, agglomerative and HDBSCAN over both VLMs.** k-Means rises from 0.440 (CLIP) and 0.476 (SigLIP) to 0.508 (DINOv2-L). Agglomerative rises from 0.459 / 0.499 to 0.551.
- **DINOv2 + agglomerative also gives the highest ARI (0.41–0.44, against ≤ 0.37 for the VLMs) while keeping clusters balanced (largest 18–20%).** Unlike SigLIP + Spectral, it is not buying AMI with a catch-all cluster. This is the strongest new result: DINOv2-B/518 + agglomerative is level with SigLIP + Spectral on AMI (0.544 vs 0.544) and clearly better on ARI (0.437 vs 0.309).
- **Spectral benefits least from DINOv2.** Its largest cluster grows to 45–48%, and its ARI falls (0.21–0.26).
- **HDBSCAN** leaves less noise with DINOv2 (49–57% vs 65–67%) and doubles to triples its ARI, but remains the weakest method.
- **HS1, same configs (k-Means / agglomerative / Spectral):** CLIP 0.317 / 0.332 / 0.323; SigLIP 0.348 / 0.344 / 0.395; DINOv2-B 0.395 / 0.406 / 0.421; DINOv2-L 0.408 / 0.443 / 0.409.
- **Highest HS3 AMI anywhere on the pilot (post-hoc):** DINOv2-L + PCA50+L2 + agglomerative, **0.578** (ARI 0.402).

Figure: `figures/DINOV2_BACKBONE_COMPARISON.png`. Full tables: `pilot/summary.csv` (472 fits, 0 errors), per-run `pilot/runs.csv`, assignments `pilot/assignments/`.

**Reproducibility:** the CLIP and SigLIP rows are recomputed inside the same script; representations equal the saved phase-2 arrays, and AMIs equal the Week 6 numbers on all 24 shared cells (max |Δ| = 0.0; `pilot/parity_with_week6.csv`). Running PCA with 6 instead of 4 BLAS threads changes SigLIP PCA50 values by up to 2.3e-5. The parity gate caught this, and 4 threads (the declared protocol) are used.

## 3. Preprocessing (pilot, k = 10; `figures/DINOV2_PREPROCESSING.png`)

| HS3 AMI | raw L2 | centered + L2 | PCA50 + L2 | PCA100 + L2 |
|---|---|---|---|---|
| DINOv2-B agglomerative | 0.403 | **0.544** | 0.532 | 0.527 |
| DINOv2-L agglomerative | 0.258 | 0.551 | **0.578** | 0.557 |
| DINOv2-B k-Means | 0.503 | 0.495 | **0.513** | 0.499 |
| DINOv2-B Spectral | 0.490 | 0.497 | 0.537 | **0.541** |
| (CLIP / SigLIP agglomerative) | 0.032 / 0.025 | 0.459 / 0.499 | 0.461 / 0.498 | 0.462 / 0.483 |

The raw-L2 agglomerative collapse is **milder for DINOv2** (AMI 0.26–0.40 instead of ≈ 0.03), but global centering still adds +0.14 to +0.29. k-Means is again insensitive to preprocessing (±0.02). PCA helps DINOv2-B Spectral (+0.04), unlike for CLIP.

## 4. RQ3: model size, layer depth, resolution and the probe gap (pilot)

**Size × layer** (224 px, centered + L2, k = 10, HS3 AMI; `figures/DINOV2_SCALING.png`):

| Model | k-Means L−1 / L−2 / L−4 | Agglomerative L−1 / L−2 / L−4 | Spectral L−1 / L−2 / L−4 |
|---|---|---|---|
| S | 0.459 / 0.374 / 0.143 | 0.467 / 0.386 / 0.143 | 0.475 / 0.414 / 0.153 |
| B | 0.488 / 0.406 / 0.155 | 0.512 / 0.465 / 0.140 | 0.489 / 0.434 / 0.178 |
| L | 0.508 / 0.493 / 0.398 | 0.551 / 0.500 / 0.352 | 0.529 / **0.535** / 0.441 |

- **Size:** S < B < L at the last layer for all three methods (k-Means +0.049, agglomerative +0.084 from S to L).
- **Depth:** the last layer is best in 8 of 9 size × method cells; the exception is L Spectral at L−2 (+0.006). Earlier layers degrade quickly for S and B: at L−4 the CLS token is almost uninformative (AMI 0.14–0.18). L is far more robust (L−4 0.35–0.44).
- **Resolution 224 → 518 px:** helps agglomerative (S +0.032, B +0.032) and k-Means slightly (+0.007), with mixed effects on Spectral (S −0.003, B +0.009). It costs about 7× more compute.

**Linear-probe reference** (supervised, separate; 13 HS3 classes with ≥ 10 pilot images, 1,520 rows; 5-fold out-of-fold predictions; `pilot/linear_probe.csv`, `figures/DINOV2_PROBE_GAP.png`):

| Backbone (layer −1) | Probe acc. | Probe AMI | k-Means AMI (same rows) | Gap |
|---|---|---|---|---|
| CLIP | 0.746 | 0.568 | 0.445 | 0.123 |
| SigLIP | 0.793 | 0.644 | 0.484 | 0.160 |
| DINOv2-S 518 | 0.789 | 0.635 | 0.472 | 0.163 |
| DINOv2-B 518 | 0.830 | 0.693 | 0.504 | 0.189 |
| DINOv2-L 224 | 0.837 | **0.719** | **0.516** | **0.203** |

**On the RQ3 question:** larger DINOv2 models improve *both* the probe and blind clustering on this pilot. The probe gains faster (+0.122 probe AMI, 0.597 → 0.719, vs +0.051 k-Means, 0.465 → 0.516, from S/224 to L/224), so **the probe gap widens with model size** (0.13 → 0.20). This is descriptive: one pilot, 5-fold CV, no confidence intervals.

## 5. Development partition (12,000 benchmark objects)

`reports/full_dataset/benchmark/dev_ids.csv`: the 12,000-row subset drawn exactly as the teammate's methods subset (`clean_analysis.py`, `default_rng(0)`). There are 13 HS3 classes and 465 rows overlap the explored pilot. CLIP and SigLIP were embedded with the exact pilot pipeline: re-embedding 32 pilot images reproduces the saved caches with max |Δ| = 0.0. DINOv2-B was run at 224 px (518 px would take ≈ 3.6 CPU-hours). There were 330 fits and 0 errors. Silhouettes use a fixed 4,000-row sample. `figures/DINOV2_DEV_PARTITION.png`, `dev/summary.csv`.

HS3 AMI / ARI, k = 10 (predeclared), mean over 5 seeds:

| Repr. | Method | CLIP | SigLIP | DINOv2-B (224) |
|---|---|---|---|---|
| centered + L2 | k-Means | 0.451 / 0.259 | 0.523 / 0.335 | **0.538** / 0.314 |
| centered + L2 | Agglomerative | 0.512 / 0.391 | 0.542 / 0.399 | **0.561 / 0.417** |
| centered + L2 | Spectral | 0.493 / 0.273 | **0.559** / 0.311 | 0.476 / 0.165 (largest cluster 50%) |
| centered + L2 | GMM diag | 0.466 / 0.234 | **0.512** / 0.291 | 0.499 / 0.231 |
| centered + L2 | HDBSCAN (noise as cluster) | 0.401 (44% noise) | 0.394 (56%) | **0.448** (38%) |
| raw L2 | Agglomerative | 0.002 (collapse) | 0.001 (collapse) | 0.361 |
| PCA100 + L2 | Agglomerative | 0.509 | 0.515 | **0.568** / 0.421 |
| PCA50 + L2 | k-Means | 0.475 | 0.519 | **0.550** |

Other labels, centered + L2, k = 10, agglomerative: HS1 0.391 / 0.422 / 0.457; Western family 0.550 / 0.555 / 0.589; instrument name 0.444 / 0.461 / 0.480 (CLIP / SigLIP / DINOv2-B).

**Label-informed reference (k = 13 = number of benchmark HS3 classes; not blind):** DINOv2-B agglomerative 0.584 (centered) and k-Means 0.575 (PCA50+L2). The teammate's earlier methods table on the same 12,000-row subset rule (DINOv2-B, 518 px, PCA50, k = 13) reported k-Means 0.55 and average-link cosine 0.58. The protocol here differs: 224 px, L2 after PCA. The numbers are close but this is **not** an exact reproduction.

**Do rankings change with dataset size?**
- They mostly do not. On both sets DINOv2 > SigLIP > CLIP for k-Means and agglomerative; SigLIP is best with Spectral; agglomerative has the highest ARI for every backbone.
- The raw-L2 agglomerative collapse of the VLMs persists.
- Absolute scores are higher on the dev partition (13 cleaner, larger classes vs 22 on the pilot), and HDBSCAN leaves less noise (33–64%).
- These are different datasets, so the numbers are not directly comparable.

## 6. Validation

`p5_validation/checks.csv` / `summary.json` (`source/p5_verify.py`). Every pilot and dev score (HS3, plus HS1 on the pilot) is recomputed from the saved assignments. The checks also cover:
- every embedding hash, row count and finiteness;
- the pinned weight hashes;
- pilot row order against `results/phase2/inputs/image_order.csv`;
- two linear-probe reruns;
- dev/held-out disjointness, and no held-out embedding;
- the benchmark counts and rule;
- secrets and file sizes of everything committable.

## 7. Conclusions and limitations

1. **DINOv2 is the best backbone for centroid and hierarchical clustering** on both the pilot and the larger development partition. DINOv2 + agglomerative gives the best balance of AMI and ARI without a dominant cluster. SigLIP + Spectral remains the best Spectral configuration, but trades ARI for AMI.
2. **Bigger is better, last layer is best** (S < B < L; −1 > −2 ≫ −4 for S and B). Higher resolution helps agglomerative.
3. **The supervised probe gains faster than clustering as the model grows** (gap 0.13 → 0.20 on the pilot), consistent with the RQ3 hypothesis. This is descriptive only.
4. **Limitations:**
   - The pilot is explored, and all "best" configurations are post-hoc.
   - Dev results are development-only, single split, with no confidence intervals.
   - DINOv2-L and 518 px were not run on the dev partition (CPU time).
   - Near-duplicates are not screened on the full set.
   - Museum confounding is untested (the 5 museums with the most benchmark objects hold 53% of them).
   - The held-out audio-object exclusion is pending.
   - The held-out partition remains untouched for the frozen-pipeline run.
