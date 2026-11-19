# P0 compute audit and P1/P2 run schedule

Audited 2026-11-19 on the machine that holds the data. **Measured** values come from the commands or files named in each row. **Estimates** are labelled and always show how they were derived. No model weights were downloaded and no DINOv2-S/L inference was run during P0.

## 1. Hardware (measured)

| Resource | Value | Source |
|---|---|---|
| CPU | 11th Gen Intel Core i5-11400H @ 2.70 GHz (max 4.5 GHz), 6 cores / 12 logical CPUs, 12 MiB L3 | `lscpu` |
| RAM | 15 GiB total (16,070,152 kB); **≈ 6.6 GiB available** at audit time (8.7 GiB used by other programs); **no swap** | `free -h`, `/proc/meminfo` |
| GPU | NVIDIA GeForce RTX 3050 Mobile (GA107M, 4 GiB BAR, so 4 GB VRAM class) is present but **not usable**. The kernel uses `nouveau`; the NVIDIA driver/compute packages are removed (`dpkg` state `rc`); `nvidia-smi` is missing; the PyTorch build is `2.15.0+cpu` (`torch.cuda.is_available() == False`). Intel UHD (TigerLake-H GT1) iGPU is not used. | `lspci -v`, `lsmod`, `dpkg -l`, torch |
| Disk | `/home` (repo and data): 733 GB, **637 GB free**; `/`: 183 GB, 134 GB free | `df -h` |
| OS | Linux 6.8.0-149-generic x86_64, glibc 2.39 | `platform.platform()` |

**Consequence:** all P1/P2 inference is CPU-only, fp32, as in every earlier run. Using the GPU would mean installing an NVIDIA driver and a CUDA build of PyTorch. That is an environment change the team has to decide on. The device would then have to be recorded in each embedding manifest, because GPU and CPU fp32 embeddings are not bit-identical.

## 2. Software environment (measured)

| Item | State |
|---|---|
| Project venv `.venv/` (named in every reproduce command) | **Missing** from the repository folder. |
| Environment used for P0 | `~/Programming/LMU/old/CMVAL/ClusteringmeetsVisionandLanguage/Clustering-Practical-Trial-main/.venv` (the venv of the pre-move checkout). Python 3.12.3. Its package list is **identical to `requirements-pilot.lock.txt`** (56 pinned packages; the only extra is `pip 24.0`). |
| Key versions | numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.1.0, torch 2.15.0+cpu, torchvision 0.29.2+cpu, timm 1.0.30, open_clip_torch 3.3.0, pillow 13.0.0, threadpoolctl 3.7.0, pytest 9.2.0 (`artifacts/environment.json`) |
| Not installed | `hdbscan` (not needed: the protocol uses `sklearn.cluster.HDBSCAN`), `umap-learn` (needed for P4's UMAP arm), `transformers` (probably needed for SigLIP 2 / Qwen3-VL-Embedding in P2) |
| Tests | `pytest tests` → 8 passed |

Recommendation before P1: recreate `.venv` in the repository from `requirements-pilot.lock.txt` (`python3 -m venv .venv && .venv/bin/pip install -r requirements-pilot.lock.txt`), or symlink the existing venv. P0 did not do this because it would install packages.

## 3. Model weights and caches (measured)

| Asset | Location (git-ignored) | Size | Verified |
|---|---|---|---|
| DINOv2-S/14 `timm/vit_small_patch14_dinov2.lvd142m@4610ca14` | `data/dinov2_model/hub/…` | 88 MB | sha256 = `models.json` |
| DINOv2-B/14 `timm/vit_base_patch14_dinov2.lvd142m@4685c99d` | same | 346 MB | sha256 = `models.json` |
| DINOv2-L/14 `timm/vit_large_patch14_dinov2.lvd142m@4741e1ca` | same | 1.22 GB | sha256 = `models.json` |
| CLIP ViT-B/32 `laion/CLIP-ViT-B-32-laion2B-s34B-b79K@1a25a446` | `data/mimo_pilot/model/hub/…` | (578 MB folder) | sha256 = `model.json` |
| SigLIP ViT-B/16 `timm/ViT-B-16-SigLIP@41f57576` | `data/phase2/siglip_model/hub/…` | (776 MB folder) | sha256 = `model.json` |
| Dev embeddings: CLIP, SigLIP, DINOv2-B 224 (layers −1, −2, −4) | `data/dinov2/emb/dev/` | 241 MB with the pilot arrays | sha256, shape, finiteness, row order |
| Pilot embeddings: DINOv2 S/B 224+518, L 224 (3 layers each) | `data/dinov2/emb/pilot/` | (in the 241 MB above) | checked in P5 (`p5_validation`) |
| DINOv2 chunk cache | `data/dinov2/cache/` (6 fingerprints, 1,250 chunks) | 169 MB | every chunk key belongs to a dev or pilot chunk |
| Images | `data/mimo_pilot/images/` (63,473 files) | 2.4 GB | 12,000 dev files re-hashed in P0 |

Licences (Hugging Face API `cardData.license`, 2026-11-19; for every repo, head revision = pinned revision): DINOv2 S/B/L **Apache-2.0**, SigLIP **Apache-2.0**, LAION CLIP **MIT** (`model_licences_hf_api.tsv`).

**Known path issue:** `data/mimo_pilot/model/model.json` and `data/phase2/siglip_model/model.json` store `weight_path` under the old checkout location (`…/CMVAL/ClusteringmeetsVisionandLanguage/Clustering-Practical-Trial-main/…`), which no longer exists. The weights are present under the current `data/`, and their hashes match. `openclip_embed.py` would fail its weight assertion if CLIP/SigLIP had to be re-embedded. P1 is not affected (DINOv2 paths in `reports/dinov2/models.json` are current), and P1/P2 do not need to re-embed CLIP/SigLIP.

## 4. Existing runtime measurements

| Measurement | Value | Source |
|---|---|---|
| DINOv2 CPU s/img (pilot timing, fp32) | S 0.053 / B 0.148 / L 0.566 at 224 px; S 0.358 / B 1.083 / L 3.55 at 518 px | `reports/dinov2/models.json` |
| DINOv2-B 224, 12,000 dev images, 3 layers, 5 threads | **1,850.6 s** (0.154 s/img, 1.04× the pilot timing) | `reports/dinov2/embeddings/dev_dinov2_b_224.json` |
| CLIP ViT-B/32, 12,000 dev images | **657.6 s** (0.055 s/img) | `dev_clip.json` |
| SigLIP ViT-B/16, 12,000 dev images | **2,303.4 s** (0.192 s/img) | `dev_siglip.json` |
| Clustering, 12,000 rows, centered + L2, 4 threads (mean fit s) | k-Means 2.2; Agglomerative 27.2; Spectral 6.4; GMM diag 6.6; **HDBSCAN 169.9** (PCA50: 6.8) | `reports/dinov2/dev/runs.csv` |
| Whole existing dev grid per backbone (110 fits incl. k=13) | CLIP 748 s, SigLIP 702 s, DINOv2-B 572 s | same |
| P0 re-fit of the k=10 grid (3 backbones, 51 fits) | 869 s for the whole stage (rescoring + 51 re-fits + silhouettes; 14:30 min:s incl. start-up), peak RSS **1.57 GB** | `baseline_stage.log` (`/usr/bin/time -v`) |

## 5. Estimates for P1 and P2 (12,000 dev objects, this CPU, fp32)

Rule for DINOv2: pilot s/img × 12,000 × 1.04 (the overhead measured on the DINOv2-B dev run). The other rows are scaled from the closest measured model by parameter count and token count. They are marked **rough** and must be replaced by a 64-image smoke-test timing in P2.

| Job | Estimate | Basis | Feasible on this machine? |
|---|---|---|---|
| 1. DINOv2-S, 224 px, layers −1/−2/−4 | **≈ 11 min** (estimate) | 0.053 s/img | Yes |
| 2. DINOv2-B, 224 px | **not needed**: reuse the validated cache (31 min if a rerun were ever required, measured 1,850.6 s) | measured | Yes |
| 3. DINOv2-L, 224 px, layers −1/−2/−4 | **≈ 2.0 h** (estimate) | 0.566 s/img | Yes. Run it unattended. Weights (1.2 GB) plus activations fit within the ≈ 6.6 GB of free RAM (estimate ≈ 3 GB peak, to be measured in P1) |
| (optional) DINOv2 S / B / L at 518 px | ≈ 1.2 h / 3.8 h / 12.3 h (estimate) | 518-px timings | S, B: possible as a labelled resolution arm. L: not recommended on CPU |
| 4a. DINOv3 ViT-B/16 (≈ 86 M), native 224 | ≈ 25–35 min (rough) | ≈ DINOv2-B 224, 196 vs 256 patches | Yes, *if* gated licence access is granted (the DINOv3 licence and HF gating must be checked in P2) |
| 4b. SigLIP 2 ViT-B/16, 224 | ≈ 40 min (rough) | ≈ SigLIP B/16 measured 0.192 s/img | Yes |
| 4c. EVA-CLIP / EVA02-B/16, 224 | ≈ 35–45 min (rough) | ≈ ViT-B/16 cost | Yes |
| 4d. Qwen3-VL-Embedding (2 B-parameter class) | ≈ 7–20 h (rough, 2–6 s/img); fp32 weights ≈ 8 GB | parameter count | **Not feasible at 12k on this CPU with the current free RAM.** Do a smoke test only, or run it in bf16 on a GPU. Log as deferred if no GPU is available |
| Larger variants (SigLIP 2 so400m/384, EVA02-L, DINOv3-L) | several hours each (rough) | ≈ DINOv2-L or more | Only if time remains; not part of the minimum |
| 5. Primary clustering grid per new embedding (5 methods, centered + L2, k = 10, seeds 0–4) | ≈ 4–5 min (≈ 3 min of it HDBSCAN); ≈ 10–13 min including k = 13 and the PCA/L2 arms | measured dev fits | Yes. HDBSCAN is the bottleneck; 1,024-d DINOv2-L will be somewhat slower (estimate) |
| P1 core grid (S/B/L × layers −1/−2/−4 × k-Means/Agglomerative/Spectral, k = 10) | ≈ 15 min (estimate) | 9 cells × (27 + 5×2.2 + 5×6.4 s) | Yes |

Downloads needed for P2 (not done in P0): DINOv3-B (~0.35 GB), SigLIP 2-B (~0.8 GB with text tower), EVA02-B (~0.6 GB), Qwen3-VL-Embedding-2B (~4–5 GB). Sizes are approximate and must be checked in P2. There is ample disk space (637 GB free).

## 6. Feasible schedule for P1 and P2 (proposal; Week 8 deadline 30 Nov 2026)

| Slot | Work | CPU time (estimate) |
|---|---|---|
| P1, day 1 | Recreate `.venv`. Extract DINOv2-S 224 (≈ 11 min), then DINOv2-L 224 (≈ 2 h, unattended). Verify alignment (`ordered_inputs_sha256`), hashes, finiteness. Reuse DINOv2-B 224. | ≈ 2.3 h |
| P1, day 2 | P1 grid: S/B/L × layers × 3 methods at k = 10, plus the full 5-method primary grid for S and L at layer −1. Score, verify and write the P1 report. | ≈ 0.5 h |
| P2, day 3 | Licence/access check for DINOv3, SigLIP 2, EVA-CLIP and Qwen3-VL-Embedding. Download the base-size checkpoints. 64-image smoke test for all four (outputs, image-only mode, dim, finiteness, row alignment, s/img). | ≈ 0.5 h |
| P2, days 3–4 | Full 12k extraction for at least two families (SigLIP 2-B and EVA02-B, or DINOv3-B if access is granted), then the primary grid for each. Log Qwen3-VL-Embedding as smoke-test-only unless a GPU is available. | ≈ 2–2.5 h |
| Buffer | Reruns, P3–P5 start | — |

The minimum P1 + P2 compute is about 5–6 CPU-hours. That fits before Week 8, provided the long runs go unattended (overnight) and no other heavy job is competing for the ≈ 6.6 GB of free RAM.
