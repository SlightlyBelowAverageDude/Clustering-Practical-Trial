# P4: Next steps (after the Week 6 milestone)

The schedule is unchanged from the Week 4 plan. Priorities within it are adjusted to what Week 6 found.

## Decisions needed from the team now

1. **Authorize the DINOv2 weight download** (S/B/L, about 1.6 GB, from `timm/vit_*_patch14_dinov2.lvd142m` on Hugging Face). This is the only blocker for RQ3.
2. **Locate or re-create the 27,348-object benchmark** (`runs/clean` IDs and the held-out split from the Week 4 protocol). Without it, the plan's target (AMI ≥ 0.68 on held-out data) cannot be tested.
3. **Agree on reporting:** AMI and ARI always together. "Best method" claims only with the metric and the k rule stated.

## Week 8 (30 Nov 2026)

- **RQ3, start:** extract DINOv2 S/B/L CLS tokens and layers −1/−2/−4 on the same 2,000 pilot rows (plan in `P3_RQ3_READINESS.md`; about 25 min of CPU in total). Run the declared grid with k-Means, agglomerative and Spectral at k=10 plus the k grid, seeds 0–4. Add the 5-fold linear-probe reference, reported separately.
- **RQ1, extend:** Spectral sensitivity to the graph (n_neighbors 10/15/30, `cluster_qr`), GMM with full/tied covariance on PCA, HDBSCAN on UMAP-10. Declare all of it before running. Repeat the subset protocol for every new method.
- **RQ2, complete:** padding vs centre-crop vs background removal (needs re-embedding), whitening, UMAP. Run per-museum centering on the larger set only (museums have 1–14 images in the pilot).
- **Stability (RQ5):** museum- and near-duplicate-aware resampling (group the 191 candidate pairs once humans confirm them). Report subset SD alongside seed SD.
- **Blind k:** compare silhouette, eigengap and a stability-based criterion on new resamples. Report the selection distribution, not a single k.

## Week 10 (14 Dec 2026)

- Finish the main development experiments on the larger benchmark's development split.
- **Select and freeze** the pipeline (encoder, representation, method, k rule) using the development split only. Write the frozen configuration to a file before touching held-out data.
- Prepare the multimodal experiments (VLM descriptions; image + audio fusion on the paired-audio objects; recount the 868-object figure).

## Week 14 (11 Jan 2027)

- Multimodal (RQ4) results, reported as exploratory and paired against image-only on the same objects.
- **One-shot held-out run** of the frozen pipeline.

## Week 16 (25 Jan 2027) / Week 17 (1 Feb 2027)

- Final presentation with demo; report and code submission.

## Housekeeping

- Complete the 180-image human annotation (`results/phase2/review/annotation_template.csv`); confirm or reject near-duplicates.
- Decide whether `reports/week6/figures/P3_MIXED_CLUSTER_EXAMPLES.png` (museum photos, CC BY-NC-SA 4.0) should be committed, or moved under the ignored `data/` like earlier galleries.
- Initialise git for the repository (it is currently not a git repository) so that the Week 6 artifacts are versioned.
