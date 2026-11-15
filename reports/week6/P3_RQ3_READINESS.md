# P3: RQ3 model-scaling readiness (Week 6)

**RQ3.** Do larger backbones (DINOv2 ViT-S → B → L) and intermediate layers (−1, −2, −4) improve unsupervised clustering, or do they mainly raise the linear probe?

**Status: not executed. Blocked only by model weights.** No DINOv2 weights, caches or embeddings exist locally (`~/.cache/huggingface` is empty and `runs/` is absent). Following the instruction not to download large models without authorization, nothing was downloaded, and **no DINOv2 result exists for Week 6**.

## What exists

| Item | Status |
|---|---|
| Extraction code | Old spikes: `spikes/common.py` (`ENCODERS["dinov2_s"/"dinov2_b"]`, timm `vit_*_patch14_dinov2.lvd142m`) and `spikes/clean_extras.py` (`dinov2_l`). These are *not* the validated pilot pipeline: no content-hash cache, manifest or padding parity checks. |
| Validated template to copy | `results/phase2/source/siglip_embed.py`: pinned revision, weight sha256, EXIF/RGB/white-square padding, deterministic CPU inference, batch caches tied to image hashes, manifest. RQ3 should reuse this, swapping in a timm DINOv2 model. |
| Dependencies | Already installed in `.venv`: torch 2.15.0+cpu, timm 1.0.30, open_clip 3.3.0, huggingface_hub 2.3.0. **No new package needed.** |
| Intermediate layers | Verified offline with random-weight architectures: timm's `forward_intermediates(indices=[-1,-2,-4], return_prefix_tokens=True)` returns per-layer patch tokens and CLS tokens for S, B and L. |
| Inputs | The 2,000-image pilot: all image files present and hash-verified, with fixed order and masks (`results/phase2/inputs/`). The 27,348-object benchmark images and IDs are **not** local. |
| Downstream | `reports/week6/source/p2_benchmark.py` and `p3_experiments.py` take any (2000 × d) array; only a loader entry is needed. |

## Expected models (verified offline from the architecture configs; random weights)

| Model (timm) | Params | fp32 weights | Embedding dim | Blocks | Layers −1/−2/−4 | Measured CPU time*: 2,000 images | Estimate: 27,348 images |
|---|---|---|---|---|---|---|---|
| `vit_small_patch14_dinov2.lvd142m` | 21.6 M | ≈ 83 MB | 384 | 12 | 12/11/9 | ≈ 1.7 min | ≈ 23 min |
| `vit_base_patch14_dinov2.lvd142m` | 85.7 M | ≈ 327 MB | 768 | 12 | 12/11/9 | ≈ 5.6 min | ≈ 1.3 h |
| `vit_large_patch14_dinov2.lvd142m` | 303.2 M | ≈ 1.16 GB | 1,024 | 24 | 24/23/21 | ≈ 16 min | ≈ 3.7 h |

\*One forward pass with intermediates, batch 16, 224 px, 4 threads, fp32, on this shared 12-core CPU (no GPU, about 15 GB RAM, about 6 GB free). Image decoding is not included. Raw output: `p3_experiments/rq3/cpu_timing_random_weights.json`. Peak memory is dominated by the weights plus batch activations, under 3 GB for L at batch 16. **The total download is about 1.6 GB** for S + B + L. Disk has 641 GB free.

## Minimal plan for November 30 (after weights are authorized)

1. **Approval:** the team authorizes downloading the three DINOv2 checkpoints from Hugging Face (`timm/vit_{small,base,large}_patch14_dinov2.lvd142m`). Pin the revision hash and record the weight sha256 in a manifest.
2. **Extraction:** a `reports/week8/source/dinov2_embed.py` copied from `siglip_embed.py`. Apply the same padding, use DINOv2's own transform (224 px, ImageNet mean/std, bicubic), deterministic CPU, batch caches keyed by image sha256. Save the CLS token of the final layer (main), and the CLS token plus mean of patch tokens for layers −1, −2 and −4. Row order is identical to `results/phase2/inputs/image_order.csv`.
3. **Validation:** shape, finiteness and norm checks. Row-order and hash checks against the CLIP and SigLIP manifests. A rerun must be bit-identical.
4. **Clustering:** the declared P2 grid restricted to `centered_l2` and `pca50_l2`, using k-Means, agglomerative and Spectral at k=10 plus the k grid, with seeds 0–4, on identical masks. Report HS1/HS2/HS3 AMI and ARI, largest cluster % and silhouette.
5. **Probe reference (RQ3's actual question):** a stratified 5-fold logistic-regression probe on the same rows, reported as a *supervised reference*, never mixed with the unsupervised scores. Classes with fewer than 5 examples need a declared rule (merge to "other" or exclude, documented in advance).
6. **Scale-up:** the full 27,348-object benchmark is needed to compare against the research plan's 0.53 / 0.83 numbers. It requires re-acquiring the cleaned dataset IDs (`runs/clean`), which are not local.

Runtime on the pilot is under 30 minutes of CPU embedding plus about 10 minutes of clustering.
