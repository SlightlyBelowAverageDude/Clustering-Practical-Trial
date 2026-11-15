# Phase 2 protocol and reproduction

All new result artifacts and source are in `results/phase2/`. No original pilot
report, script, input or result was overwritten. New downloaded model weights,
batch embedding caches and all gallery photograph bytes remain in ignored
`data/phase2/`, following the original project data-exclusion policy. The HTML
galleries and slides use relative links; keep `data/` alongside the results.

## Declared experiment

`DECLARED_PROTOCOL.json` was written before the new CLIP fits. Task 1 completed
and was independently verified before SigLIP acquisition/inference started.
The fixed comparison anchor is centered full-dimensional + L2, k=10, both methods;
it was not selected from phase 2 HS AMI. No final winner/k is claimed as an
unbiased generalization estimate. All grid comparisons are descriptive.
The same pilot was already explored in phase 1; this declaration is not an
independent preregistered test on unseen data.

- Exactly the original ordered 2,000 images and cached CLIP features. Source
  manifests, exact ordered IDs/byte hashes and evaluation masks are preserved in
  `inputs/`. Image bytes and cached arrays are rehashed. The original dataset is
  not reharvested and no additional museum image is downloaded.
- CLIP: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, immutable revision and original
  weight hash inherited from the verified original manifest; 512 dimensions.
- SigLIP: `timm/ViT-B-16-SigLIP`, revision
  `41f575766f40e752fdd1383e9565b7f02388c1c4`, 768 dimensions. This is the timm
  conversion of the original Google Big Vision/WebLI checkpoint, not a newly
  trained model. The architecture and evaluation transform come from the pinned
  `open_clip_config.json`. Weight SHA256, actual transform, code hash, Python,
  Torch/OpenCLIP/timm/Pillow versions and inference environment are recorded.
- Both images receive the existing EXIF transpose, RGB, white-square-padding
  policy, followed by each encoder's published evaluation transform. SigLIP:
  224 square, bicubic squash, channel mean/std 0.5. Padding is a shared dataset
  adaptation, separate from the published encoder transform. CPU float32,
  `eval()`, `requires_grad_(False)`, deterministic Torch, seed 42, four threads.
  Output vectors are L2-normalized. SigLIP caches are content-verified in batches
  of 16; fingerprint includes exact inputs, code, model and relevant libraries.
- Five representations: original L2; centered full-dimensional then L2;
  centered PCA50 without post-L2; PCA50 then L2; PCA100 then L2. Non-whitened,
  full-SVD PCA. Each representation and fit parameters are saved. PCA fits
  only features; no label is passed into transformation or clustering.
- K-Means: Lloyd, n_init=10, max_iter=300, tol=1e-4, seeds 0–4. Agglomeration:
  average linkage, cosine metric, deterministic one fit per configuration.
  k=5,10,20,40. 120 full fits per encoder; 240 total. All methods use every
  image in identical row order. HS1/HS2/HS3 scoring uses identical nonmissing
  masks (1,565 / 1,563 / 1,549); labels are never features or k selectors.
- AMI, ARI and NMI use scikit-learn's arithmetic averaging defaults. Record
  all cluster sizes, actual k, largest percentage, size CV, singleton count,
  normalized cluster entropy, and clustering wall time. Transformation and
  silhouette time are recorded separately. Both cosine and Euclidean
  silhouettes are exact on all 2,000 images; K-Means' method metric is
  Euclidean and average linkage's is cosine. Compare common cosine silhouettes
  cautiously: internal separation is not HS correctness. No invalid silhouette
  case occurred. Timing includes concurrent local audit/review activity and is
  not an isolated hardware performance benchmark.
- Average cosine is invariant to positive row scaling. PCA50 and PCA50+L2
  give identical arrays of cluster predictions at every k for both encoders.
  Report all 40 declared cells, and additionally remove these redundant cosine
  normalization cells when counting nonredundant comparisons. Grid cells share
  images and are not independent statistical replicates.

## Chance and stability

For seed-0 predictions of each representation/method/k, generate 20 complete-row
permutations (seeds 1000–1019), preserving exact full cluster sizes. Evaluate on
the same label masks. Save every random partition; 1,600 partitions total and
4,800 metric rows. No test-set p-value or significance claim is inferred from
this control. NMI's chance expectation need not be zero.

K-Means seed stability: all ten pairwise prediction ARI/NMI comparisons for the
five seeds of every representation/k; 400 comparisons across encoders.

Subset sensitivity: five identical paired 1,600-image samples without replacement
(80%, seeds 2026–2030), indices sorted to preserve original relative order. Sample
images without consulting labels. Use the predeclared centered-full + L2 k=10
anchor. Refit the mean inside each subset. Cross every subset with all five
K-Means seeds; average cosine is deterministic. Save indices, masks, means,
features, predictions, per-level scores, cluster sizes and runtimes. This adds 60
fits / 180 metric rows. Compare seed-0 partitions on subset intersections and
against the full-pilot fit restricted to the subset.

Report within-subset initialization SD separately from SD of five subset
seed-means. Overlapping subsets of this fixed, near-duplicate-containing pilot
do not provide independent-dataset confidence intervals. Paired encoder deltas
are computed on the same subsets; seed/sample and label-coverage effects remain
conditional on this pilot. Museum/specimen resampling and external data are next.

## Review and galleries

Seed 20261102, target 180 disjoint images, six 30-slot strata in fixed order:
possible view/duplicate pairs; museum-random; HS1-stratified; large/small failure
clusters; lexical/low-entropy/low-saturation screens; missing/multiple HS. Labels
are used here only to satisfy the explicitly requested review stratification,
never to fit clusters. Actual strata/backfills are saved. The review set includes
all five HS1 classes and 106 museums. The cluster stratum has only 14 new targeted
images after earlier strata, so 16 seeded random backfills complete its slots.

Candidate pairs: same museum + normalized title and CLIP cosine >=0.88, or
64-bit dHash Hamming <=3 and cosine >=0.90; sort by decreasing cosine, increasing
Hamming and stable original row IDs; show 15 disjoint pairs. These rules flag
similar photos/types, not confirmed duplicate objects. All 191 candidate edges
are preserved. Human confirmation is needed before grouping or exclusion.

The annotation CSV starts with all human fields blank; source fields remain
unchanged. Screening is not annotation, and 65 lexical part matches are not 65
confirmed parts. Generic placeholder, branding/detail and painting observations
are recorded separately as assistant visual observations requiring human
confirmation. The targeted sample cannot estimate population error prevalence.
Rerunning `review.py` refuses to overwrite nonempty human annotations.

Meaningful gallery: fixed original CLIP L2 K-Means k=5 seed 0, eight images
nearest each cluster mean. Failure gallery: fixed CLIP L2 average cosine k=5,
twenty seeded random giant-cluster images and all seven small-cluster members.
Neither gallery is selected by maximizing HS scores. Contact sheets are under
ignored data/, HTML and image-order CSVs are in `galleries/`.

## Execute from the original project directory

Use the existing `.venv` and `requirements-pilot.lock.txt`; no new dependency was
installed for phase 2. No Europeana API key is needed. SigLIP's pinned public
model requires network access on first use. Do not change constants or reuse
this result folder for a different protocol; version a new experiment instead.

```bash
.venv/bin/python results/phase2/source/benchmark.py clip
.venv/bin/python results/phase2/source/verify.py clip
# Inspect TASK1_RESULTS.md and report Task 1 findings before the next command.
.venv/bin/python results/phase2/source/siglip_embed.py
.venv/bin/python results/phase2/source/benchmark.py siglip_grid
.venv/bin/python results/phase2/source/verify.py siglip
.venv/bin/python results/phase2/source/evaluation.py clip
.venv/bin/python results/phase2/source/evaluation.py siglip
.venv/bin/python results/phase2/source/verify_evaluation.py clip
.venv/bin/python results/phase2/source/verify_evaluation.py siglip
.venv/bin/python results/phase2/source/review.py
.venv/bin/python results/phase2/source/quality_diagnostics.py
.venv/bin/python results/phase2/source/charts.py
.venv/bin/python results/phase2/source/report.py
.venv/bin/python -m pytest results/phase2/source/test_phase2.py -q
.venv/bin/python results/phase2/source/final_audit.py
```

Independent audits recompute every reported metric from saved predictions and
the persisted evaluation mask, every silhouette from saved representations, all
chance size constraints and deterministic permutations, refitted subset means,
and prediction-stability comparisons. Original pilot parity is exact for all 96
overlapping evaluation rows. `FINAL_VALIDATION.json` records the final checks.

No commits, pushes or new dataset downloads were performed.

Primary model sources:
[SigLIP checkpoint/configuration](https://huggingface.co/timm/ViT-B-16-SigLIP),
[original Big Vision implementation](https://github.com/google-research/big_vision),
[CLIP checkpoint](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K).
