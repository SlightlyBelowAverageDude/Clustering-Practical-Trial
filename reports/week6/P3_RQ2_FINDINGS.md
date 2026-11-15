# P3: Robustness, AMI vs ARI, and RQ2 preprocessing (Week 6)

Protocol `P3_EXPERIMENT_PROTOCOL.json` was declared at 2026-11-14T23:30:54Z, before any P3 fit. Data: the validated MIMO pilot (2,000 images, 216 museums; HS3 = 1,549 labelled rows, 22 classes) and the CLIP ViT-B/32 and SigLIP ViT-B/16 caches. All results are **descriptive development-set results on an already-explored pilot**. Subsets overlap, so they are not independent test sets, and no significance test was run. Choosing "the best" configuration from these tables is post-hoc.

Reproducibility checks: the subset refits reproduce the saved phase-2 subset predictions exactly (**60/60**), and the preprocessing refits reproduce the phase-2 and P2 assignments exactly (**240/240**). Each subset's representation matches a fresh recomputation (centering refit inside the subset). Labels were read only after the fit and blind-k selection files were written.

## 3.1 Spectral robustness: five 80% subsets (k = 10, centered + L2 refit per subset)

Subsets: the phase-2 seeds 2026–2030, 1,600 rows each without replacement. k-Means and Spectral use seeds 0–4; agglomerative is deterministic. Source: `P3_ROBUSTNESS_RESULTS.csv`, `figures/P3_SPECTRAL_ROBUSTNESS.png`.

| Encoder | Method | HS3 AMI: mean of 5 subset means (SD across subsets; range) | Mean seed SD | HS3 ARI mean | Largest cluster % (mean / max) | Δ AMI vs k-Means (mean; subsets > 0) | Δ ARI vs k-Means (subsets > 0) |
|---|---|---|---|---|---|---|---|
| CLIP | k-Means | 0.447 (0.009; 0.438–0.462) | 0.009 | 0.312 | 14.9 / 18.1 | — | — |
| CLIP | Agglomerative | 0.465 (0.019; 0.449–0.497) | det. | **0.363** | 23.1 / 25.5 | +0.019; 4/5 | +0.051; 5/5 |
| CLIP | Spectral | **0.468** (0.019; 0.444–0.484) | 0.002 | 0.269 | 34.4 / 40.2 | +0.021; 4/5 | −0.043; 0/5 |
| SigLIP | k-Means | 0.489 (0.004; 0.484–0.494) | 0.012 | 0.348 | 14.3 / 21.4 | — | — |
| SigLIP | Agglomerative | 0.497 (0.004; 0.491–0.500) | det. | **0.361** | 27.2 / 29.3 | +0.008; 5/5 | +0.014; 4/5 |
| SigLIP | Spectral | **0.520** (0.020; 0.502–0.550) | 0.0003 | 0.296 | 34.0 / 38.0 | +0.031; 5/5 | −0.051; 0/5 |

**Answers**
- **Does Spectral consistently beat k-Means on AMI?** Mostly. It does in 9/10 encoder × subset cases (SigLIP 5/5 with 25/25 paired seeds positive; CLIP 4/5, with subset 2028 at −0.018). On ARI it never does (0/10).
- **Spectral vs agglomerative:** Spectral has higher AMI in 5/5 SigLIP subsets but only **2/5 CLIP subsets**. Agglomerative has higher ARI in **10/10** subsets. P2's "Spectral is best" therefore holds for SigLIP + AMI only.
- **Sensitivity to composition:** Spectral's AMI varies more across subsets (SD 0.019–0.020) than k-Means (0.004–0.009), even though Spectral barely changes across seeds (seed SD ≤ 0.002). The full-pilot SigLIP value of **0.544 lies near the top** of its subset range (0.502–0.550), so the subset mean of 0.520 is a more cautious summary.
- **Partition stability:** at k=10, seed 0, partition agreement (ARI) between subset fits and the full-pilot fit restricted to the same rows is lowest for Spectral: CLIP 0.63 (min 0.49), SigLIP 0.70 (min 0.57). For k-Means it is 0.70 / 0.80 and for agglomerative 0.68 / 0.81. Subset-to-subset agreement on shared rows shows the same order. Spectral is **seed-stable but composition-sensitive**.
- **Dominant cluster:** this persists. Spectral's largest cluster is 30–40% of the subset in every subset (mean 34%), against a balanced 10%.

### Blind k on subsets (`P3_BLIND_K_ROBUSTNESS.csv`, `figures/P3_BLIND_K_ROBUSTNESS.png`)

Selection files were written before any label was read. "Oracle" means the post-hoc best k chosen with labels; it is reported for reference only and is **not blind**.

| Method | Criterion | k selected over the 10 encoder × subset cases | Δ AMI vs fixed k=10 (mean; better / worse) | Mean gap to oracle | Matched oracle |
|---|---|---|---|---|---|
| k-Means | silhouette | 8 (×6), 10 (×4) | +0.003; 5 / 1 | −0.004 | 5/10 |
| Agglomerative | silhouette | 5 (×5), 8, 10 (×4) | −0.025; 1 / 5 | −0.037 | 2/10 |
| Spectral | silhouette | 5 (×3), 8 (×5), 10, 20 | −0.016; 4 / 5 | −0.038 | 1/10 |
| Spectral | eigengap | 5 (×2), 8 (×2), 10 (×2), 15 (×3), 20 | +0.002; 5 / 3 | −0.020 | 5/10 |

- **Does eigengap consistently pick a better k than silhouette?** No. It was better in 5/10 cases, equal in 2/10, and worse in 3/10. Its choices are **unstable across overlapping subsets** (k from 5 to 20; on CLIP subset 2028 it picked k=5 and lost 0.031 AMI against k=10). P2's observation (eigengap within 0.004 of the oracle in 4 configurations) does **not** hold up under resampling.
- Silhouette is reasonable for k-Means (k = 8–10, close to the oracle). For agglomerative and Spectral it tends to favour coarse partitions (k = 5).
- Neither criterion recovers the 22 HS3 classes. All selected k values are at or below 20.

## 3.2 Why AMI and ARI disagree (`p3_experiments/ami_vs_ari/`, `figures/P3_AMI_VS_ARI.png`, `figures/P3_MIXED_CLUSTER_EXAMPLES.png`)

Configuration: SigLIP, centered + L2, k=10, seed 0, from the saved P2 assignments; 1,549 labelled rows. Purity is descriptive only.

| | Spectral | Agglomerative | k-Means |
|---|---|---|---|
| HS3 AMI / ARI | **0.543** / 0.307 | 0.499 / **0.360** | 0.469 / 0.341 |
| Largest cluster (labelled rows) | 524 (33.8%) | 403 (26.0%) | 235 (15.2%) |
| …its purity / classes ≥ 5% | 0.23 / 5 | 0.21 / 5 | 0.53 / 2 |
| …share of all same-cluster pairs | **61%** | 40% | 21% |
| Pair precision / recall | 0.33 / 0.52 | 0.38 / **0.54** | **0.44** / 0.40 |
| Classes with ≥ 70% of images in one cluster | **16** | 15 | 9 |
| Classes split over ≥ 3 clusters (≥ 10% each) | 2 | 2 | 9 |
| Mutual information (nats) / cluster entropy | **1.18** / 1.93 | 1.09 / 1.91 | 1.11 / 2.25 |

**Interpretation**
- **Why Spectral has higher AMI.** Apart from one catch-all cluster, Spectral forms many nearly pure clusters: lip-reed brass 97% and 100%, lutes 91% and 99%, harps 90%, board zithers 83%. Sixteen of 22 classes are concentrated (≥70%) in a single cluster. That yields the most mutual information at almost the same cluster entropy as agglomerative, hence the higher AMI.
- **Why agglomerative has higher ARI.** ARI counts pairs of images, so it grows with the square of cluster size. Spectral's largest cluster (774 of 2,000 images; 524 labelled across 20 HS3 classes) contains 61% of all same-cluster pairs, and most of those pairs come from different classes. Pair precision falls to 0.33 and ARI falls with it. Agglomerative also has a mixed catch-all cluster (purity 0.21), but it is smaller and holds only 40% of the pairs. k-Means is the most balanced and has the highest pair precision, but it splits classes (9 classes over ≥3 clusters) and has low recall.
- **Overly large clusters?** Yes, for Spectral: about 34–39% of images in one cluster, against a balanced 10% and a largest HS3 class of 18.5%. Its AMI advantage is partly bought with one uninformative "everything small and irregular" cluster.
- **Visually similar but acoustically different instruments merged?** Yes, in the saved partition (`mixed_cluster_examples.csv`, images shown with catalogue titles and HS3 codes):
  1. **Long tubes (cluster id 8, 323 images):** reedpipes 422 (52%: cornemuse/bagpipe, clarinets, oboe) merged with edge-blown flutes 421 (46%: flute, fife, concert flute). Lip-reed instruments 423 shaped like woodwinds (“Basson russe”, “Serpent droit”) also appear here. These objects share an elongated tube form and differ in how the sound is produced. Agglomerative shows the same 421/422 merge (45%/46%), so this is not specific to Spectral.
  2. **Small hand-held objects (cluster id 1, 774 images, 20 classes):** vessel flutes, whistles and panpipes (421), castanets, bells and objects catalogued as struck idiophones (111), and rattles and cog rattles (112).
  3. In the agglomerative partition, struck drums (211, 69%) share a cluster with struck idiophones (111, 27%).

  These are descriptive observations on one seed of one configuration. They illustrate the visual-form-vs-acoustic-function hypothesis but do not establish it. Some catalogue labels look surprising (two "Tambour" records are coded 111), so label noise cannot be excluded.

## 3.3 RQ2: preprocessing ablation (`P3_PREPROCESSING_RESULTS.csv`, `figures/P3_PREPROCESSING_ABLATION.png`, `figures/P3_CLUSTER_DIAGNOSTICS.png`)

Same 2,000 rows and masks; k = 5 / 10 / 20 (principal k = 10); seeds 0–4. Representations are the hash-checked phase-2 arrays. k-Means and agglomerative results reproduce phase 2 exactly. Spectral is new for raw L2, PCA50 and PCA100 + L2.

HS3 AMI at k = 10 (mean over seeds; largest cluster %, worst seed, in brackets):

| Encoder | Method | raw L2 | centered + L2 | PCA50 | PCA50 + L2 | PCA100 + L2 |
|---|---|---|---|---|---|---|
| CLIP | k-Means | **0.449** (18.5) | 0.440 (17.5) | 0.446 (17.3) | 0.441 (16.4) | 0.439 (14.5) |
| CLIP | Agglomerative | 0.032 (**98.7**) | 0.459 (24.4) | 0.461 (21.6) | 0.461 (21.6) | **0.462** (23.0) |
| CLIP | Spectral | 0.444 (51.8) | **0.470** (33.6) | 0.443 (45.9) | 0.464 (33.4) | 0.462 (28.0) |
| SigLIP | k-Means | **0.498** (15.8) | 0.476 (13.9) | 0.499 (18.0) | 0.493 (13.9) | 0.482 (14.5) |
| SigLIP | Agglomerative | 0.025 (**98.7**) | **0.499** (28.3) | 0.498 (24.8) | 0.498 (24.8) | 0.483 (24.8) |
| SigLIP | Spectral | 0.517 (39.7) | 0.544 (38.7) | 0.515 (34.2) | 0.540 (39.4) | **0.547** (39.2) |

**Answers**
1. **Does centering help every algorithm?** No. It is decisive for **agglomerative** (+0.43 to +0.47 AMI at k=10; it prevents the ≈99% single-cluster collapse at k = 5, 10 and 20). It helps **Spectral** at k=10 (+0.027 for both encoders), but not consistently across k (SigLIP k=20: −0.012). It does nothing for or slightly hurts **k-Means** (k=10: −0.009 CLIP, −0.022 SigLIP; k=5 and 20 within ±0.007).
2. **Does PCA help?** Not by itself. For k-Means and agglomerative, PCA50 / PCA50+L2 / PCA100+L2 are within about ±0.02 of centered + L2. For Spectral, **PCA50 without re-normalisation is worse** (CLIP 0.443 vs 0.464; SigLIP 0.515 vs 0.540), and its largest cluster grows (CLIP 46%). With L2 re-normalisation PCA matches centered + L2. Most of PCA's benefit appears to come from the centering it includes; this is a hypothesis.
3. **Does preprocessing change the best algorithm?** Yes. On raw L2, agglomerative is unusable and k-Means is best for CLIP. On every centered representation, Spectral has the top AMI at k=10 for SigLIP, and for CLIP on 2 of the 4 centered representations (agglomerative on the other 2). ARI rankings differ again: at k=10, agglomerative has the top ARI in 6/8 centered encoder × representation cells (k-Means in the other 2: SigLIP PCA50 and PCA50+L2), and k-Means leads on raw L2.
4. **Can preprocessing prevent collapse?** Yes, for agglomerative, through global centering. Spectral's dominant cluster shrinks with centering (CLIP 52% → 34%) but never disappears (28–52% across representations).
5. **Consistent between CLIP and SigLIP?** The direction is the same for the large effects (the agglomerative rescue, the k-Means centering penalty, PCA50-without-L2 hurting Spectral). The small Spectral differences between PCA variants differ in sign between encoders.

**Not run:** per-museum centering. It is a different transformation from global centering. In the pilot, museums contribute 1–14 images, so a one-image museum would be centered to a zero vector. It is deferred to the larger dataset, where the research notes report that it lowered AMI (0.55 → 0.52; not recomputed here). Whitening, UMAP, padding vs crop, and background removal were also not run.

## What changed from P2

| P2 statement | After P3 |
|---|---|
| Spectral best HS3 AMI at k=10 | Holds vs k-Means in 9/10 subsets; vs agglomerative only for SigLIP (5/5), not CLIP (2/5). Full-pilot 0.544 is at the top end; subset mean 0.520. |
| Agglomerative higher ARI | Confirmed in 10/10 subsets, and explained by the dominant mixed Spectral cluster (61% of same-cluster pairs). |
| Eigengap promising (4 configurations) | Not supported: unstable choices (k = 5–20), matched the oracle in 5/10. |
| Centering is the main fix | Confirmed for agglomerative; helps Spectral at k=10 but not at every k; does not help k-Means. |
