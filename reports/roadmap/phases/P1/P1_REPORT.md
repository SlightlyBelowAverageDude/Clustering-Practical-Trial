# P1: Finish DINOv2 S/B/L at scale (RQ3)

## 1. Phase information

| Field | Value |
|---|---|
| Phase ID and name | P1  \|  Finish DINOv2 S/B/L at scale (RQ3) |
| Research questions addressed | RQ3: do larger backbones and intermediate layers improve unsupervised clustering, or mainly supervised representation quality? |
| Milestone | Week 8 (30 Nov 2026) |
| Execution date | 2026-11-19 |
| Status | **COMPLETE.** All four roadmap tasks were done on the full 12,000 objects; 121/121 independent checks pass. |
| Git branch and commit | `Aiman-testing`, working tree on top of `ce9cb0d` (P0). **Not committed.** The user commits and pushes; the commit SHA is recorded by that commit. |
| Dataset version | `MIMO-HS3-DEV12K-v1` (manifest content sha256 `6c9bf55b…`), unchanged |
| Protocol version | `FAIR-CMP-v1` (protocol content sha256 `ffff3046…`), unchanged; no protocol deviation |
| Executed by | Claude Code (agent), for the team |

## 2. Objective

On the full development benchmark, P1 compares the three DINOv2 sizes (S/14, B/14, L/14) and three depths of their CLS representation (last block, −2, −4). Everything else is held fixed: the same objects, the same image input at 224 px, the same clustering and the same metrics. Until now, size and depth had only been compared on the 2,000-image pilot.

**Scientific question:** Under fixed k = 10 and the frozen settings, does HS3 clustering agreement increase from S to B to L? Is the last layer always the best one to cluster?

**Hypothesis (from the pilot, stated before running):** S < B < L at the last layer for k-Means, Agglomerative and Spectral, and the last layer is best (pilot: 8 of 9 size × method cells).

## 3. What was already available

| Artifact | Path | Reused how | Verified in P1? |
|---|---|---|---|
| Frozen manifest, protocol, baselines | `reports/roadmap/phases/P0/artifacts/` | Unchanged | Hash + git-unchanged checks |
| DINOv2-B 224 dev embeddings, layers −1/−2/−4 | `data/dinov2/emb/dev/dinov2_b_224_L{1,2,4}.npy`, manifest `reports/dinov2/embeddings/dev_dinov2_b_224.json` | **Reused, not re-extracted**: hashes, row order, preprocessing and layer definition all match | Yes: independent recompute on 24 images |
| DINOv2 S/B/L weights (timm, LVD-142M) | `data/dinov2_model/` | Loaded offline | sha256 re-checked |
| Extractor | `reports/dinov2/source/dinov2_embed.py` | Run unchanged for S and L | Its output is independently recomputed |
| Clustering factory | `reports/week6/source/p2_benchmark.py` | Imported unchanged | Re-fit determinism checks |
| CLIP / SigLIP dev results | `P0/artifacts/BASELINE_REFERENCE.csv` | Context rows only (cached Week 6 runs, verified in P0, **not re-run**) | Equality check |
| Pilot RQ3 results | `reports/dinov2/DINOV2_BENCHMARK.md` §4 | Comparison only (different rows and classes) | Not re-checked |

**New in P1:**
- the repository `.venv`;
- DINOv2-S and DINOv2-L dev embeddings (3 layers each);
- 99 new clusterings (S/B/L × 3 layers × k-Means 5 seeds + Agglomerative + Spectral 5 seeds);
- result tables, 4 figures, and an independent verifier.

## 4. What was done

1. **Environment.** Created `.venv` in the repository from `requirements-pilot.lock.txt` with the PyTorch CPU index; the installed packages equal the lockfile exactly. `pytest tests` → 8 passed. The GPU was not used: the RTX 3050 has no NVIDIA driver loaded and torch is a CPU build, and enabling it would change the environment.
2. **Readiness.** DINOv2-S/B/L weights are local and their hashes match. 9.1 GiB of RAM was free at the start. Batch 16 was kept for every model; it is the cache chunk size, and DINOv2-L peaked at only 2.63 GiB.
3. **DINOv2-B reuse check (P1.1).** Verified the checkpoint, the weight hash, 12,000 × 768 float32 per layer, finiteness, the row-order key against the frozen manifest, the descriptor (white-square pad → timm Resize 224 bicubic → CenterCrop 224 → ImageNet normalisation; CLS + final LayerNorm), and the array hashes against P0. All valid, so no rerun.
4. **Extraction (P1.2).** `source/run_extraction.sh` ran `dinov2_embed.py --set dev --size 224 --batch 16 --threads 6` for S, then L, under GNU `time`, offline (`HF_HUB_OFFLINE=1`).
   - Layers −1/−2/−4 come from one `forward_intermediates` pass. The extractor maps blocks to requested layers explicitly and asserts that layer −1 equals the pooled output.
   - 12,000/12,000 images for each model, no failures, no restarts.
   - DINOv2-S hit a 3.85 h wall-clock stall in one progress interval; the cause is not identified, and system sleep is likely. It does not affect the embeddings, which are hash-keyed and independently recomputed. It is removed from the reported active time (§6.4).
5. **Clustering (P1.3).** `source/p1_cluster.py fit`: for each of the 9 model × layer embeddings, centered + L2, then k-Means (seeds 0–4), Agglomerative (average, cosine) and Spectral (15-NN, seeds 0–4) at k = 10, with 4 BLAS threads. Cosine silhouette on the frozen 4,000-row sample. No labels were read during fitting. `score` then computed HS3/HS1 AMI, ARI and NMI and aggregated over seeds (mean, sample SD).
6. **Validation (P1.5).** `source/p1_verify.py`, independent of the runners, ran 121 checks. The key ones:
   - **Layer mapping:** a separate re-implementation (torchvision preprocessing plus a manual loop over `model.blocks` with the final LayerNorm on each block's CLS) on 24 random dev images. Every saved layer matches its own block exactly (max |Δ| = 0.0) and differs from the other two layers by 10–22 (`artifacts/layer_mapping_check.csv`).
   - **Metrics:** all 99 assignments rescored from file.
   - **Reproduction:** the 11 DINOv2-B layer −1 fits are bit-identical to the Week 6 dev assignments; three new fits were re-fitted bit-identically.
   - **Figures:** all 96 plotted metric values and the 9 runtime x-values were checked against the CSVs.
   - **Isolation:** every cache chunk is a dev chunk; no reserved objects.
7. **Figures and tables.** `source/p1_figures.py`.

**Deviations from the plan:** none.
- The full 3 × 3 method × layer matrix was completed, so no prioritisation was needed.
- No supervised probe was run on dev; the roadmap does not require one for P1.

## 5. Experimental setup

| Item | Value |
|---|---|
| Dataset | 12,000 development objects, `MIMO-HS3-DEV12K-v1`; HS3 13 classes, all 12,000 rows scored (n_evaluated = 12,000) |
| Models | DINOv2-S/14 `timm/vit_small_patch14_dinov2.lvd142m@4610ca14` (22.1 M params, 384-d, 12 blocks); B/14 `…base…@4685c99d` (86.6 M, 768-d, 12 blocks); L/14 `…large…@4741e1ca` (304.4 M, 1,024-d, 24 blocks). Apache-2.0. Hashes in `artifacts/DINOV2_MODEL_MANIFEST.json`. |
| Image input | EXIF transpose → RGB → white square pad → Resize 224 (bicubic, antialias) → CenterCrop 224 (no-op after padding) → ImageNet mean/std. Identical descriptor for S/B/L (verified). Native timm default is 518 px; 224 px is the frozen P1 resolution. |
| Representation | CLS token after the final LayerNorm, taken at blocks −1, −2, −4 (0-based block 11/10/8 for S and B, 23/22/20 for L); float32 |
| Feature transform | centered + L2, fitted on the 12,000 rows, label-free |
| Clustering | k-Means (n_init 10, lloyd), Agglomerative (average, cosine), Spectral (15-NN, arpack, n_init 10); k = 10 |
| Seeds | 0–4 for k-Means and Spectral; Agglomerative deterministic (1 run, no SD) |
| Metrics | HS3 AMI / ARI / NMI (primary), HS1 AMI / ARI, cosine silhouette (4,000-row sample), largest cluster %, number of clusters, fit seconds |
| Software | repository `.venv`: Python 3.12.3, torch 2.15.0+cpu, timm 1.0.30, scikit-learn 1.9.1, numpy 2.5.4; OMP_NUM_THREADS = 4 for clustering |
| Hardware | i5-11400H 6C/12T, 15 GiB RAM, CPU only; torch 6 threads for S/L extraction (B: 5 threads, Week 6) |

## 6. Actual results

### 6.1 Model scaling, last layer (`artifacts/DINOV2_SBL_12K.csv`)

12,000 dev objects, centered + L2, k = 10, HS3. The mean ± SD is over 5 seeds; Agglomerative is deterministic.

| Model | Method | HS3 AMI | HS3 ARI | HS3 NMI | HS1 AMI | Silhouette | Largest cluster % (worst seed) | Clusters | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2-S | k-Means | 0.466 ± 0.000 | 0.251 ± 0.000 | 0.468 | 0.337 | 0.229 | 21.5 | 10 | 1.1 |
| DINOv2-B | k-Means | 0.538 ± 0.011 | 0.314 ± 0.006 | 0.539 | 0.421 | 0.317 | 26.1 | 10 | 1.8 |
| DINOv2-L | k-Means | 0.542 ± 0.010 | 0.317 ± 0.024 | 0.543 | 0.429 | 0.333 | 27.5 | 10 | 2.0 |
| DINOv2-S | Agglomerative | 0.555 | **0.427** | 0.556 | 0.454 | 0.170 | 33.8 | 10 | 15.8 |
| DINOv2-B | Agglomerative | 0.561 | 0.417 | 0.562 | 0.457 | 0.259 | 21.6 | 10 | 29.6 |
| DINOv2-L | Agglomerative | **0.587** | 0.391 | 0.588 | 0.474 | 0.335 | 19.3 | 10 | 39.9 |
| DINOv2-S | Spectral | 0.481 ± 0.000 | 0.200 ± 0.000 | 0.482 | 0.360 | 0.185 | 38.6 | 10 | 4.0 |
| DINOv2-B | Spectral | 0.476 ± 0.000 | 0.165 ± 0.000 | 0.477 | 0.330 | 0.177 | 50.1 | 10 | 3.5 |
| DINOv2-L | Spectral | 0.486 ± 0.000 | 0.177 ± 0.000 | 0.488 | 0.335 | 0.189 | 52.4 | 10 | 3.3 |

**Context (cached Week 6 dev runs, P0-verified, not re-run in P1):** CLIP k-Means / Agglomerative / Spectral AMI 0.451 / 0.512 / 0.493; SigLIP 0.523 / 0.542 / 0.559 (ARI SigLIP 0.335 / 0.399 / 0.311).

DINOv2-B rows are new P1 fits, bit-identical to the P0 baseline (0.538 / 0.561 / 0.476).

Silhouettes are computed in each model's own feature space. They describe how compact each partition is, but they are not a quality ranking across models.

### 6.2 Layer depth (`artifacts/DINOV2_LAYER_ABLATION.csv`)

HS3 AMI (ARI in brackets), means over seeds, k = 10, centered + L2.

| Model | Method | Layer −4 | Layer −2 | Layer −1 (last) | Best |
|---|---|---:|---:|---:|---|
| S | k-Means | 0.214 (0.108) | 0.434 (0.303) | **0.466** (0.251) | −1 |
| S | Agglomerative | 0.220 (0.146) | 0.407 (0.260) | **0.555** (0.427) | −1 |
| S | Spectral | 0.202 (0.031) | 0.461 (0.246) | **0.481** (0.200) | −1 |
| B | k-Means | 0.232 (0.120) | 0.445 (0.243) | **0.538** (0.314) | −1 |
| B | Agglomerative | 0.223 (0.143) | 0.489 (0.338) | **0.561** (0.417) | −1 |
| B | Spectral | 0.219 (0.066) | **0.492** (0.234) | 0.476 (0.165) | **−2** |
| L | k-Means | 0.430 (0.284) | 0.499 (0.293) | **0.542** (0.317) | −1 |
| L | Agglomerative | 0.387 (0.222) | 0.574 (0.393) | **0.587** (0.391) | −1 |
| L | Spectral | 0.510 (0.287) | **0.523** (0.237) | 0.486 (0.177) | **−2** |

Seed SD for every k-Means / Spectral cell is ≤ 0.011 (AMI). Largest clusters and silhouettes for every cell are in the CSV; figure: `artifacts/figures/P1_LAYER_DEPTH.png`.

### 6.3 Comparison with the 2,000-image pilot (descriptive; different rows and 22 vs 13 classes)

| Pattern | Pilot (`DINOV2_BENCHMARK.md` §4, 224 px) | 12k development (P1) |
|---|---|---|
| k-Means, last layer S → B → L | 0.459 → 0.488 → 0.508 (monotone) | 0.466 → 0.538 → 0.542 (S < B; **B ≈ L**) |
| Agglomerative S → B → L | 0.467 → 0.512 → 0.551 | 0.555 → 0.561 → 0.587 (monotone; S → B only +0.006) |
| Spectral S → B → L | 0.475 → 0.489 → 0.529 | 0.481 → 0.476 → 0.486 (**not monotone**, spread 0.010) |
| Last layer best | 8 of 9 cells (exception: L Spectral) | **7 of 9** (exceptions: B and L Spectral) |
| Layer −2 > layer −4 | 9 of 9 | 9 of 9 |
| L degrades least with depth | yes (L −4 0.35–0.44 vs S/B 0.14–0.18) | yes (L −4 0.39–0.51 vs S/B 0.20–0.23) |
| Supervised probe gap grows with size | yes (0.13 → 0.20) | **not measured on 12k** (pilot-only) |

### 6.4 Runtime and memory (`artifacts/DINOV2_RUNTIME_MEMORY.csv`)

| Model | Extraction, 12,000 images (CPU) | s / image | Peak RSS | Threads | Provenance |
|---|---:|---:|---:|---:|---|
| DINOv2-S | **12.4 min active** (raw 4 h 03 min incl. a 3.85 h stall) | 0.062 | 0.87 GiB | 6 | P1 new |
| DINOv2-B | 30.8 min | 0.154 | not recorded | 5 | Week 6 cache, reused |
| DINOv2-L | 96.5 min | 0.483 | 2.63 GiB | 6 | P1 new |

Clustering per layer −1 embedding (sum over seeds): S 41 s, B 56 s, L 66 s (k-Means + Agglomerative + Spectral). The whole 99-fit grid with silhouettes took 11 min 24 s at a peak RSS of 1.48 GiB. P0 estimates were S ≈ 11 min and L ≈ 2.0 h; the measured times are 12.4 and 96.5 min.

**Missing or failed experiments:** none.
- No DINOv2 resolution arm (518 px) was run; it is not part of P1.
- No supervised probe was run on dev.

## 7. What do these results mean?

**Did increasing model size improve clustering?** (measured) Partly, and less than the pilot suggested.
- **DINOv2-S is clearly worse than B for k-Means** (0.466 vs 0.538, a gap far larger than the seed SD). For Agglomerative, S is almost as good as B (0.555 vs 0.561, +0.006).
- **DINOv2-L improves on B for Agglomerative** (0.587 vs 0.561, +0.026). This is the highest HS3 AMI any configuration has reached on the 12k development set at k = 10, above SigLIP + Spectral (0.559).
- **For k-Means, L adds nothing measurable** (+0.004 with seed SD ≈ 0.010). Under the protocol rule (difference < 2 × SD) this is "no clear difference".
- **For Spectral, size does not matter** (0.476–0.486). All three Spectral solutions put 39–52% of the objects into one cluster.
- So S → B → L is **not consistent across methods.** It holds for Agglomerative, holds as S < B ≈ L for k-Means, and does not hold for Spectral.

**Does a larger model improve AMI but reduce ARI?** (measured) Yes, for Agglomerative: AMI rises (0.555 → 0.561 → 0.587) while ARI falls (0.427 → 0.417 → 0.391).

What the two metrics mean:
- **AMI** measures how much information the clusters share with the HS3 classes.
- **ARI** counts how many *pairs* of objects are placed consistently, so it is dominated by the large classes (flutes, reedpipes, lutes).

Interpretation: DINOv2-L separates the small classes better (higher AMI) but splits or merges some large classes differently, which costs pairwise agreement. Neither metric alone answers "better", which is why both are reported.

**More or fewer dominant clusters?** (measured) It depends on the method.
- **Agglomerative becomes more balanced with size:** the largest cluster shrinks from 33.8% (S) to 21.6% (B) to 19.3% (L).
- **Spectral becomes more dominated:** 38.6% → 50.1% → 52.4%.
- **k-Means stays in a narrow range:** 21.5–27.5%.

**Is the extra compute justified?** (measured cost; the benefit is interpretation) DINOv2-L costs about 3× B's extraction time (96.5 vs 30.8 min) and about 8× S's (12.4 min).
- For the best method it buys +0.026 AMI, at −0.026 ARI.
- On this CPU, 1.6 h is affordable once, so keeping L as a candidate is reasonable. The gain is modest, though, and it is method-specific.
- DINOv2-B remains the better cost/benefit point for k-Means.
- DINOv2-S is a cheap option *only* for Agglomerative.

**Is the last layer best?** (measured) In 7 of 9 cells.
- **Spectral prefers layer −2 for B and L** (B 0.492 vs 0.476; L 0.523 vs 0.486). For L, even layer −4 (0.510) beats the last layer.
- In those cells the earlier layers also produce higher ARI and smaller dominant clusters (L Spectral −4: ARI 0.287, largest cluster 31.8%, compared with 0.177 and 52.4% at −1).
- **Layer −2 beats −4 in all 9 cells.**
- **The best layer does depend on size:** for S and B, layer −4 is almost uninformative (AMI 0.20–0.23), while L keeps a usable signal at −4 (0.39–0.51). That is consistent with the pilot.

**Do the pilot findings hold on 12,000 objects?**
- Holds: "last layer is generally best", "−2 > −4", "L is robust to depth", and S < B < L for Agglomerative.
- Weakened: S < B < L for k-Means (B ≈ L on the larger set).
- Does not hold: S < B < L for Spectral.
- The pilot's probe-gap result was not tested here and remains pilot-only.

**What would be misleading to conclude?**
- "Bigger DINOv2 is always better for clustering": it is not, for k-Means at the top end or for Spectral.
- "DINOv2-L + Agglomerative is the best pipeline": it was chosen after seeing the dev results (post-hoc), it has lower ARI than DINOv2-S/B + Agglomerative, and it rests on one dev split.
- Any significance claim: seed SD measures clustering randomness only, not variation from re-sampling the objects. Differences of about 0.01 (for example S vs B Agglomerative, B vs L k-Means) are not established. Robustness to the data composition (museum-aware subsets, re-sampled objects) is a P5 question.

**Answer to RQ3 (for this dataset):** Larger DINOv2 backbones improve unsupervised clustering only selectively, by up to +0.03 AMI for one method and with an ARI trade-off. Intermediate layers mostly hurt, except for graph-based Spectral clustering on B and L. Together with the pilot probe result (gap 0.13 → 0.20), this is *consistent with* the hypothesis that scale mainly improves supervised representation quality. On 12k that remains a hypothesis until the probe is measured on the development set.

**Next:** P2 brings in the newer families, using DINOv2-B and DINOv2-L layer −1 (k-Means and Agglomerative) as the DINOv2 references. Spectral's sensitivity to layer depth and its collapse into one dominant cluster belong in P3 and P4.

## 8. Limitations and problems

- **Single development split, no confidence intervals;** seed SD ≠ robustness to data composition. All "best" choices are post-hoc.
- **Resolution:** all three models ran at 224 px. Their native timm resolution is 518 px, and in the pilot 518 px helped Agglomerative by about +0.03, so the absolute DINOv2 numbers may be conservative.
- **DINOv2-B extraction used 5 torch threads (Week 6), S/L used 6.** This does not affect embedding values: the independent recompute with 4 threads matches all three models exactly. It affects only the timing comparison.
- **DINOv2-S raw wall time includes an unexplained 3.85 h stall** (one progress interval; system sleep is likely). The active time is derived from the progress log.
- **No peak-memory figure for the reused DINOv2-B extraction.**
- **Supervised probe not measured on 12k,** so the "probe gap" part of RQ3 is unanswered at scale.
- **Agglomerative creates very small clusters** (smallest 6–38 objects) next to large ones. AMI rewards isolating small classes, and that interacts with the AMI/ARI trade-off described above.
- **HS3 labels are museum-assigned; three classes have 23–33 dev objects; museum confounding is untested (P5).**
- **Reserved pool (carried from P0):** untouched since the 15 Nov 2026 split, but **not never-seen**. 10 reserved objects were scored in the 29 Oct smoke test, and the teammate's pre-split exploration covered the whole benchmark (`P0/artifacts/RESERVED_PRIOR_EXPOSURE.csv`, P0 report §8). P1 processed no reserved object (verified).

## 9. Files and reproducibility

| Kind | Path |
|---|---|
| Scripts | `reports/roadmap/phases/P1/source/run_extraction.sh`, `p1_cluster.py`, `p1_figures.py`, `p1_verify.py`; extractor `reports/dinov2/source/dinov2_embed.py` (unchanged) |
| Configuration / protocol | `reports/roadmap/phases/P0/artifacts/EXPERIMENT_PROTOCOL.json` (FAIR-CMP-v1, unchanged) |
| Model manifest | `reports/roadmap/phases/P1/artifacts/DINOV2_MODEL_MANIFEST.json`; embedding manifests `reports/dinov2/embeddings/dev_dinov2_{s,b,l}_224.json` |
| Embeddings (git-ignored) | `data/dinov2/emb/dev/dinov2_{s,b,l}_224_L{1,2,4}.npy`; chunk caches `data/dinov2/cache/<fingerprint>/` |
| Result tables | `artifacts/DINOV2_SBL_12K.csv`, `DINOV2_LAYER_ABLATION.csv`, `DINOV2_RUNTIME_MEMORY.csv`, per-run `p1_runs.csv`, `p1_scores_per_run.csv` |
| Cluster assignments | `artifacts/assignments/dinov2_{s,b,l}_224_L{1,2,4}__centered_l2__{method}__k10__s{seed}.npy` (99 files, 24 KB each) |
| Figures | `artifacts/figures/P1_MODEL_SCALING.png`, `P1_LAYER_DEPTH.png`, `P1_AMI_ARI_COMPARISON.png`, `P1_RUNTIME_TRADEOFF.png`; plotted values `figure_values.json` |
| Logs | `artifacts/logs/extract_dinov2_{s,l}.{log,time}`, `cluster_fit.{log,time}`, `p1_verify.log` |
| Validation | `artifacts/P1_VALIDATION.md`, `DINOV2_EXTRACTION_VALIDATION.md`, `p1_checks.csv`, `p1_summary.json`, `layer_mapping_check.csv` |

```bash
# repository root
python3 -m venv .venv && .venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-pilot.lock.txt
bash reports/roadmap/phases/P1/source/run_extraction.sh dinov2_s dinov2_l      # ≈ 12 min + 97 min CPU; resumable
.venv/bin/python reports/roadmap/phases/P1/source/p1_cluster.py fit            # ≈ 11 min, labels not read
.venv/bin/python reports/roadmap/phases/P1/source/p1_cluster.py score
.venv/bin/python reports/roadmap/phases/P1/source/p1_figures.py
.venv/bin/python reports/roadmap/phases/P1/source/p1_verify.py                 # 121 checks, ≈ 2 min
```

## 10. Completion checklist

| # | Roadmap task | Status | Evidence |
|---|---|---|---|
| P1-005 | Reuse already validated DINOv2-B embeddings at 224 px; rerun B only if settings or alignment differ. | **Complete** | B not re-extracted. Hash, order, descriptor and layer checks pass (groups C and E); 11/11 B layer −1 fits bit-identical to Week 6. |
| P1-006 | Extract DINOv2-S and DINOv2-L embeddings for the same 12,000 development objects, using matching 224 px settings. | **Complete** | 12,000/12,000 each; same order key `112d1630…`; identical descriptor across S/B/L |
| P1-007 | Measure runtime, peak memory, dimensions, failures and preprocessing for every size; record model checkpoint IDs. | **Complete** | `DINOV2_RUNTIME_MEMORY.csv`, `DINOV2_MODEL_MANIFEST.json`. B peak memory was not recorded in Week 6 (stated). |
| P1-008 | For supported models compare last and intermediate layers (-2, -4); cluster each with identical reference settings. | **Complete** | `DINOV2_LAYER_ABLATION.csv`: 27 cells, 99 fits |

**DONE WHEN:** A matched DINOv2 S/B/L table with AMI/ARI and runtime, plus a layer-depth chart → **met** (`DINOV2_SBL_12K.csv`, `DINOV2_RUNTIME_MEMORY.csv`, `figures/P1_LAYER_DEPTH.png`).

## 11. Next phase handoff

- **Ready for P2:**
  - frozen protocol and manifest;
  - DINOv2 S/B/L dev embeddings, all 3 layers, verified;
  - DINOv2 references for comparisons: **DINOv2-B L−1** (k-Means 0.538, Agglomerative 0.561) and **DINOv2-L L−1** (k-Means 0.542, Agglomerative 0.587);
  - the repository `.venv`.
- **Read first:** this report §6–7; `P0/artifacts/EXPERIMENT_PROTOCOL.json` (`rule_for_new_encoders`); `P0/artifacts/COMPUTE_AUDIT.md` §5–6.
- **Frozen (unchanged):** FAIR-CMP-v1 and MIMO-HS3-DEV12K-v1.
- **Risks for P2:**
  - New encoders need downloads (not done) and possibly `transformers`, which is not in the lockfile, so adding it would be a documented environment change.
  - DINOv3 weights are gated behind a custom licence.
  - Qwen3-VL-Embedding at 12k is not feasible on this CPU (estimate).
  - Keep the machine from sleeping during long extractions (`systemd-inhibit` or the power settings); see the S stall.
- **May P2 begin?** Yes, after review of this report. P2 was not started.

## 12. Presentation-ready summary

**What we did:** We extracted DINOv2-S and DINOv2-L features for all 12,000 development objects, reused the verified DINOv2-B features, and clustered the last layer and two earlier layers of every model under one frozen protocol (centered + L2, k = 10, 5 seeds). That is 99 clusterings, all independently re-verified.

**Main finding:** On 12,000 objects, bigger DINOv2 helps clustering only selectively. With Agglomerative clustering, AMI rises S 0.555 → B 0.561 → L 0.587, the best k = 10 development result so far, while ARI falls (0.427 → 0.417 → 0.391). With k-Means, L is no better than B (0.542 vs 0.538 ± 0.011). With Spectral, size makes no difference (0.476–0.486).

**Scientific interpretation:** Scale improves how well small instrument classes are separated, not how well the overall grouping agrees pair-by-pair. The last layer is best in 7 of 9 cases; graph-based Spectral clustering does better on earlier layers. The pilot's clean "S < B < L" pattern does not carry over to the larger set.

**Numbers to show:**

| | k-Means AMI | Agglomerative AMI / ARI | Spectral AMI | Extraction (12k, CPU) |
|---|---:|---:|---:|---:|
| DINOv2-S | 0.466 | 0.555 / 0.427 | 0.481 | 12 min |
| DINOv2-B | 0.538 | 0.561 / 0.417 | 0.476 | 31 min |
| DINOv2-L | 0.542 | 0.587 / 0.391 | 0.486 | 97 min |

12,000 development objects, 13 HS3 classes, 224 px, CLS of the last block, centered + L2, k = 10, seeds 0–4.

**Recommended figure:** `reports/roadmap/phases/P1/artifacts/figures/P1_MODEL_SCALING.png` (backup: `P1_LAYER_DEPTH.png`)

**Remaining limitations:** development-only, one split, post-hoc best choice; no dev-scale supervised probe; 224 px only; seed SD does not cover data re-sampling; the reserved pool is untouched since the split but not never-seen.
