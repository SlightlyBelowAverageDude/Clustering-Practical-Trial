# P0: Freeze a fair comparison protocol

## 1. Phase information

| Field | Value |
|---|---|
| Phase ID and name | P0  \|  Freeze a fair comparison protocol |
| Research questions addressed | None directly. P0 is the prerequisite that makes RQ1–RQ5 comparisons fair: one data manifest and one evaluation protocol for every model family. |
| Milestone | Week 8 (30 Nov 2026) |
| Execution date | 2026-11-19 |
| Status | **COMPLETE.** All 195 P0 gate checks pass. Two pre-existing findings about the reserved pool are documented and handed to P5A/P16 (§8). They are not P0 failures. |
| Git branch and commit | `Aiman-testing`. Started from `abd235d` ("Adding final results for week 6"); this report and its artifacts are added by the commit "Complete P0 protocol freeze and establish phase reporting" (`git log -- reports/roadmap/phases/P0`). |
| Dataset version | `MIMO-HS3-DEV12K-v1`, manifest content sha256 `6c9bf55ba6992ef61f9ecbd4d578181c36b795ab3ba911b32bffa6071862d7df` |
| Protocol version | `FAIR-CMP-v1`, protocol content sha256 `ffff30463d9610e1fd1adc1e28ee544541c323b354918ef9b454f17af814826f` |
| Executed by | Claude Code (agent), for the team |

## 2. Objective

Before new encoders (DINOv2-S/L in P1; DINOv3, SigLIP 2, EVA-CLIP, Qwen3-VL-Embedding in P2) are added, P0 makes sure that every later number is computed on the same objects, in the same order, with the same labels, clustering settings and metrics. It also confirms that the Week 6 baseline numbers can be reproduced exactly.

**Scientific question:** Is the existing 12,000-object development benchmark internally consistent and separated from the reserved data, and can its baseline results (CLIP, SigLIP, DINOv2-B) be reproduced exactly from saved artifacts under one written protocol?

**Hypothesis (stated before running):** Yes. The Week 6 / P5 validation (`reports/dinov2/p5_validation/`, 812/812 checks) suggested so, but it did not re-fit the clusterings, did not re-hash the dev images and did not check the cache contents for reserved images.

## 3. What was already available

| Artifact | Path | Reused how | Verified in P0? |
|---|---|---|---|
| Benchmark and split (27,347 objects, dev / heldout / pilot-overlap) | `reports/full_dataset/benchmark/*.csv`, `manifest.json`, `stats.json` | Referenced by path + hash, not copied | Yes: hashes, counts, split rule, separation |
| Dev images (12,000 of 63,473) | `data/mimo_pilot/images/` (git-ignored) | Read-only | Yes: all 12,000 re-hashed |
| Dev embeddings CLIP / SigLIP / DINOv2-B 224 (L−1, −2, −4) | `data/dinov2/emb/dev/`, manifests `reports/dinov2/embeddings/dev_*.json` | Read-only | Yes: hash, shape, dtype, finiteness, row order |
| Dev clustering results (330 fits, 0 errors) | `reports/dinov2/dev/{runs,scores,summary}.csv`, `assignments/` | Rescored; k=10 grid re-fitted | Yes |
| Clustering factory and benchmark code | `reports/week6/source/p2_benchmark.py`, `reports/dinov2/source/*.py` | Imported unchanged | Hashes recorded in the protocol |
| Earlier protocols | `reports/dinov2/DINOV2_PROTOCOL.json`, `reports/week6/P1/P3_EXPERIMENT_PROTOCOL.json` | Consolidated into FAIR-CMP-v1 | Checked for consistency (k, methods) |
| Model weights DINOv2 S/B/L, CLIP, SigLIP | `data/dinov2_model/`, `data/mimo_pilot/model/`, `data/phase2/siglip_model/` | Not loaded for inference | sha256 re-checked |

**New in P0:** an independent verification script (`source/p0_verify.py`), the frozen dev manifest, the frozen protocol, the baseline reference table, re-fits of 51 baseline clusterings, a cache audit for reserved images, a licence check, a compute audit, and the phase reporting system (`reports/roadmap/`). No new embeddings, models, splits or experiments were created.

## 4. What was done

1. **Read the plan and the repository.** The roadmap was found as `~/Documents/LMU/Clusteringmeetsvisionandlangguage/Roadmap.docx`, saved 2026-11-19 14:45. That file name differs from the expected `Clustering_Execution_Roadmap_W8_to_Final(2).docx`, but the content is the P0–P20 plan with optional P5A. Also read: `FULL_DATASET_REPORT.md`, `DINOV2_BENCHMARK.md`, `DINOV2_PROTOCOL.json`, the Week 6 reports, `IDEAS.md` and the source of every script that produced the dev results.
2. **Environment.** The project `.venv` named in all reproduce commands is missing. The venv of the pre-move checkout (`~/Programming/LMU/old/CMVAL/…/.venv`) matches `requirements-pilot.lock.txt` exactly and was used. Nothing was installed. Existing tests: `pytest tests` → 8 passed.
3. **Dataset revalidation (74 gates).** Recomputed the dev selection rule from the benchmark row order. Compared dev rows byte-for-byte between `dev_ids.csv` and `benchmark_objects.csv`. Checked separation on id, image id, byte hash and pixel hash; the pilot-overlap flags; image files and hashes; label completeness and HS hierarchy; class and museum counts against the Week 6 tables; and embedding row alignment.
4. **Reserved-data discipline.** Labels were parsed for dev rows only. For reserved rows only `id, partition, image_id, sha256, pixel_sha256, in_explored_pilot` were loaded. Nothing reserved was embedded, clustered or scored, and no P5A split was created.
5. **Cache audit for reserved images.** Every DINOv2 cache chunk key (1,250 chunks) was matched against dev and pilot chunk keys. The per-batch CLIP caches in `data/mimo_pilot/embeddings/` were checked for reserved image ids. This found **10 reserved objects in the 29 Oct 2026 smoke test** (§8).
6. **Baseline verification (113 gates).** For the 27 baseline configurations (3 backbones × 5 methods at k=10, plus 3 × 4 at the label-informed k=13), P0 re-hashed every assignment file and recomputed HS3 AMI/ARI/NMI, HS1 AMI and the cosine silhouette from the saved assignments. It then **re-fitted all 51 k=10 clusterings** (every seed, including HDBSCAN) from the cached embeddings with the original factory and compared the labels.
7. **Protocol freeze.** Wrote `EXPERIMENT_PROTOCOL.json` (FAIR-CMP-v1), which consolidates the existing rules and adds explicit rules for seed aggregation, HDBSCAN noise, invalid labels, k policy and new encoders. Checked it against the earlier protocol, the baseline table and the weights (8 gates).
8. **Licences.** Queried the Hugging Face model API for the five checkpoints in use. Every licence is permissive, and every repository's head revision equals the pinned revision (`artifacts/model_licences_hf_api.tsv`).
9. **Compute audit.** Measured CPU, RAM, GPU, disk, software and weights. Derived run-time estimates for P1/P2 from existing measurements (`artifacts/COMPUTE_AUDIT.md`).
10. **Documentation system and roadmap.** Created `reports/roadmap/` (index, template, presentation workflow, P1–P20/P5A placeholders) and the updated Word roadmap (`docs/roadmap/`).

**Deviations from the plan:** none in substance. Re-fitting the k=10 grid (≈ 14 min CPU on development data only) goes beyond "use saved assignments". It was done because rescoring alone cannot show that the protocol reproduces the assignments.

## 5. Experimental setup (audit configuration)

| Item | Value |
|---|---|
| Dataset | 12,000 development objects (`MIMO-HS3-DEV12K-v1`); HS1/HS2/HS3/family/name valid on all 12,000 |
| Models | No inference. Cached embeddings: CLIP ViT-B/32 (512-d), SigLIP ViT-B/16 (768-d), DINOv2-B/14 224 px layer −1 (768-d) |
| Representation | centered + L2 (fitted on the 12,000 rows, label-free) |
| Clustering | `p2_benchmark.make()`: k-Means, Agglomerative (average, cosine), Spectral (15-NN), GMM (diag), HDBSCAN (mcs 20) |
| k | fixed k=10 (primary); k=13 label-informed reference (rescored only) |
| Seeds | 0–4 for k-Means/Spectral/GMM; deterministic methods 1 run |
| Metrics | AMI, ARI, NMI (HS3 primary, HS1), cosine silhouette (fixed 4,000-row sample, seed 0), largest cluster %, assigned clusters, noise %, fit seconds |
| Verification method | `source/p0_verify.py`: tolerance 1e-9 for rescored metrics, 1e-6 for silhouettes, exact equality for re-fitted labels |
| Software | Python 3.12.3, numpy 2.5.4, scikit-learn 1.9.1, torch 2.15.0+cpu, OMP_NUM_THREADS = 4 (`artifacts/environment.json`) |
| Hardware | i5-11400H 6C/12T, 15 GiB RAM, CPU only |

## 6. Actual results

### 6.1 Dataset verification (measured; `artifacts/P0_VALIDATION.md`, `artifacts/DEV_MANIFEST.json`)

| Quantity | Verified value | Reference |
|---|---:|---:|
| Cleaned photographs (`clean_records.csv`) | 59,658 | 59,658 |
| Eligible benchmark objects | 27,347 | 27,347 |
| Development objects | 12,000 (all ids unique; = `default_rng(0).choice` rule, same order) | 12,000 |
| Reserved `heldout` | 14,696 | 14,696 |
| Pilot-overlap exclusions | 651 (all are pilot objects) | 651 |
| Dev ∩ reserved on id / image id / byte sha256 / pixel sha256 | 0 / 0 / 0 / 0 | 0 |
| Explored pilot objects inside dev (allowed by the split rule) | 465 | 465 |
| Pilot objects inside reserved `heldout` | 0 | 0 |
| Dev image files present, hash-identical | 12,000 / 12,000 | |
| Images per object | exactly 1 (image ids unique) | |
| Dev rows with an empty HS1 / HS2 / HS3 / family / name | 0 / 0 / 0 / 0 / 0 | |
| HS hierarchy consistent (hs1 ⊂ hs2 ⊂ hs3 ⊂ notation) | 0 violations | |
| HS3 classes in dev | 13 (= benchmark set) | 13 |
| Museums in dev | 168; top-5 hold 52.7% | |
| Embedding row order (`ordered_inputs_sha256`) | `112d1630…` for CLIP, SigLIP and DINOv2-B | identical |

**HS1 in dev:** aerophones 7,214 · chordophones 2,609 · idiophones 1,635 · membranophones 476 · electrophones 66.
**HS3 in dev:** 421 flutes 3,103 · 422 reedpipes 2,196 · 321 lutes 2,054 · 423 labrosones 1,547 · 111 directly struck idiophones 953 · 112 indirectly struck idiophones 682 · 211 struck drums 453 · 412 free aerophones (interruptive) 368 · 314 board zithers 294 · 322 harps 261 · 513 electro-acoustic chordophones 33 · 531 analogue electronic 33 · 212 rattle drums 23.
**Top museums in dev:** "Musical instrument museum" 1,574 · Ethnological Museum Berlin 1,359 · Philharmonie de Paris 1,232 · Swedish Museum of Performing Arts 1,205 · University of Edinburgh 949.

### 6.2 Baseline results (frozen references; `artifacts/BASELINE_REFERENCE.csv`)

12,000 dev objects, centered + L2, **fixed k = 10**, HS3 (13 classes, n = 12,000). Values are means over seeds 0–4 (± sample SD) where the method is stochastic. Every row below was rescored from the saved assignments (max |Δ| = 0.0 against `reports/dinov2/dev/summary.csv`) and **re-fitted with bit-identical labels** (51/51 fits).

| Backbone | Method | HS3 AMI | HS3 ARI | HS3 NMI | HS1 AMI | Silhouette | Largest cluster % | Clusters | Noise % | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP B/32 | k-Means | 0.451 ± 0.007 | 0.259 ± 0.011 | 0.452 | 0.313 | 0.151 | 27.5 | 10 | 0 | 1.6 |
| CLIP B/32 | Agglomerative | 0.512 | 0.391 | 0.513 | 0.391 | 0.113 | 31.4 | 10 | 0 | 21.4 |
| CLIP B/32 | Spectral | 0.493 ± 0.000 | 0.273 ± 0.000 | 0.494 | 0.366 | 0.079 | 34.6 | 10 | 0 | 9.0 |
| CLIP B/32 | GMM diag | 0.466 ± 0.010 | 0.234 ± 0.023 | 0.467 | 0.336 | 0.133 | 32.8 | 10 | 0 | 4.3 |
| CLIP B/32 | HDBSCAN (noise = cluster) | 0.401 | 0.157 | 0.404 | 0.395 | 0.140 | 32.9 | 17 | 44.2 | 137.2 |
| SigLIP B/16 | k-Means | 0.523 ± 0.004 | 0.335 ± 0.013 | 0.525 | 0.383 | 0.162 | 19.9 | 10 | 0 | 2.3 |
| SigLIP B/16 | Agglomerative | 0.542 | 0.399 | 0.543 | 0.422 | 0.142 | 25.7 | 10 | 0 | 30.3 |
| SigLIP B/16 | **Spectral** | **0.559 ± 0.000** | **0.311 ± 0.000** | 0.560 | 0.417 | 0.087 | 34.7 | 10 | 0 | 6.6 |
| SigLIP B/16 | GMM diag | 0.512 ± 0.005 | 0.291 ± 0.015 | 0.513 | 0.373 | 0.143 | 24.7 | 10 | 0 | 7.2 |
| SigLIP B/16 | HDBSCAN (noise = cluster) | 0.394 | 0.089 | 0.397 | 0.286 | 0.280 | 18.6 | 23 | 55.6 | 193.1 |
| DINOv2-B 224 | k-Means | 0.538 ± 0.011 | 0.314 ± 0.006 | 0.539 | 0.421 | 0.317 | 26.1 | 10 | 0 | 1.9 |
| DINOv2-B 224 | **Agglomerative** | **0.561** | **0.417** | 0.562 | 0.457 | 0.259 | 21.6 | 10 | 0 | 29.8 |
| DINOv2-B 224 | Spectral | 0.476 ± 0.000 | 0.165 ± 0.000 | 0.477 | 0.330 | 0.177 | **50.1** | 10 | 0 | 3.4 |
| DINOv2-B 224 | GMM diag | 0.499 ± 0.015 | 0.231 ± 0.017 | 0.501 | 0.379 | 0.242 | 34.6 | 10 | 0 | 7.6 |
| DINOv2-B 224 | HDBSCAN (noise = cluster) | 0.448 | 0.178 | 0.451 | 0.412 | 0.310 | 37.3 | 24 | 38.4 | 179.3 |

HDBSCAN on assigned rows only (secondary scoring, always quoted with coverage): CLIP AMI 0.684 at 55.8% coverage; SigLIP 0.745 at 44.4%; DINOv2-B 0.665 at 61.6%. Silhouette is cosine on centered + L2 features. It is not comparable across backbones, because each backbone has its own geometry. Fit seconds are from the saved runs; the P0 re-fits took 0.92–1.23× as long.

**Label-informed k = 13 reference** (not blind; rescored, not re-fitted): DINOv2-B Agglomerative 0.584 / 0.437, k-Means 0.557 / 0.332; SigLIP Agglomerative 0.553 / 0.408, Spectral 0.541 / 0.287; CLIP Agglomerative 0.518 / 0.394, k-Means 0.469 / 0.278 (HS3 AMI / ARI).

**Week 6 headline numbers reproduced:** DINOv2-B + Agglomerative 0.561 / 0.417, SigLIP + Spectral 0.559 / 0.311 (HS3 AMI / ARI). Both are exact to three decimals and re-fitted bit-identically.

**Missing or failed experiments:** none. The PCA50/PCA100/raw-L2 dev results in `reports/dinov2/dev/summary.csv` remain valid historical results; they were rescored by the P5 validation but are not part of the frozen reference table.

### 6.3 Verification totals (`artifacts/p0_summary.json`)

| Group | Gates passed |
|---|---|
| Dataset | 74 / 74 |
| Baseline | 113 / 113 |
| Protocol | 8 / 8 |
| **Total** | **195 / 195** |

Also recorded: 3 informational values and 2 findings (§8).

## 7. What do these results mean?

**What did we learn?** (measured) The development benchmark is exactly what the Week 6 reports said: 12,000 unique objects, one image each, complete labels, 13 HS3 classes, no overlap with the reserved pool on any identifier or image hash, and every cached embedding aligned row-for-row. Every baseline number is reproducible bit-for-bit on this machine: the same embeddings, code and seeds give identical cluster labels, not merely similar scores. That is a stronger guarantee than Week 6 had. It means a later difference between two models cannot come from hidden randomness in clustering or from row misalignment.

**Did the results support the hypothesis?** Yes for the development benchmark and the baselines. There is one qualification about the reserved pool, described below.

**Did something unexpected happen?** Yes, two things.
1. (measured) A 32-image CLIP smoke test from 29 Oct 2026, run before the split existed, embedded, clustered and scored 10 objects that are now in the reserved `heldout` pool.
2. (documented, not measurable locally) The teammate's Weeks 2–4 exploration (`IDEAS.md`) reported k-means and probe scores over all 27,348 benchmark rows, which include today's reserved pool.

Interpretation: the reserved pool is **untouched since the 15 Nov 2026 split**. It is not "never seen". Early pipeline choices (DINOv2-B, padding, PCA50) were partly motivated by analyses that included those objects. The effect on a later final-test number is probably small, because those analyses were descriptive and not tuned on a held-out score. But the wording of any final-test claim has to say this.

**Why is this scientifically relevant?** RQ1–RQ3 are comparisons: a newer encoder against DINOv2-B, a large model against a small one. They are only meaningful if everything except the model is held fixed. FAIR-CMP-v1 now fixes the rows, order, labels, masks, representation, clustering settings, seeds, threads and metrics. It also records which model-specific differences cannot be equalised (native resolution, normalisation constants, output type).

**Did this phase change earlier conclusions?** It does not change any number. Two readings are sharpened (interpretation):
- At fixed k = 10, DINOv2-B + Agglomerative is the strongest balanced baseline: highest ARI (0.417), largest cluster 21.6%.
- SigLIP + Spectral reaches a similar AMI (0.559) only with a 34.7% catch-all cluster and lower ARI (0.311).
- DINOv2-B + Spectral collapses (one cluster holds 50.1% of objects, ARI 0.165). So "DINOv2 is better" holds for k-Means and Agglomerative, not for every method.

**What next?** P1 can start on the frozen protocol: extract DINOv2-S and -L at 224 px (DINOv2-B is reused), then compare sizes and layers under exactly these settings. The reserved-pool findings go to P5A, which must exclude the 10 smoke-test objects when it locks its split.

## 8. Limitations and problems

- **Findings handed on (not fixed in P0, because fixing them would change the split):**
  - F1: 10 reserved `heldout` objects were exposed in the 29 Oct smoke test. Their ids are in `artifacts/RESERVED_PRIOR_EXPOSURE.csv`. P5A must exclude them from both the validation and final-test portions.
  - F2: Pre-split teammate analyses covered the whole benchmark (`IDEAS.md` §Results; artifacts not local). Final-test claims must read "untouched since the split".
- **Held-out audio exclusion** (objects with recordings) is still not applied; see P7.
- **Near-duplicates** across dev/reserved are not screened. Only exact byte/pixel hashes were checked (0 overlap). Near-duplicate screening is P5.
- **Environment:** the in-repo `.venv` is missing. P0 used an identical relocated venv; recreate it before P1. The CLIP/SigLIP `model.json` files point to the old checkout path, so re-embedding CLIP/SigLIP would need a path remap. P1 is not affected.
- **GPU unusable** (no NVIDIA driver, CPU-only torch). Qwen3-VL-Embedding at 12k is not feasible on this CPU (estimate; see the compute audit).
- **Statistical:** one development split, no confidence intervals. Seed SD covers clustering randomness only, not sampling of objects. Silhouettes use one fixed 4,000-row sample.
- **Labels:** HS3 is a museum-assigned taxonomy. Two classes have 33 dev objects and one has 23, so per-class results for these are noisy. The top-5 museums hold 52.7% of dev objects, so museum confounding is untested (P5).
- **Vocabulary:** names for HS3 112, 412 and 531 are standard H-S names; they are not in the local MIMO vocabulary file.
- **Roadmap file name:** the local roadmap is `Roadmap.docx`, not `…(2).docx`. Its content matches the P0–P20 plan described for this phase.

## 9. Files and reproducibility

| Kind | Path |
|---|---|
| Verification script | `reports/roadmap/phases/P0/source/p0_verify.py` |
| Frozen dev manifest | `reports/roadmap/phases/P0/artifacts/DEV_MANIFEST.json` (references `reports/full_dataset/benchmark/dev_ids.csv`, sha256 `9a8d6139…4a1439`) |
| Frozen protocol | `reports/roadmap/phases/P0/artifacts/EXPERIMENT_PROTOCOL.json` |
| Baseline reference | `reports/roadmap/phases/P0/artifacts/BASELINE_REFERENCE.csv` (27 rows) |
| Re-fit evidence | `reports/roadmap/phases/P0/artifacts/baseline_refit.csv` (51 rows), `baseline_stage.log` |
| Validation | `reports/roadmap/phases/P0/artifacts/P0_VALIDATION.md`, `p0_checks.csv`, `p0_summary.json`, `checks_{dataset,baseline,protocol}.csv` |
| Reserved exposure list (ids only) | `reports/roadmap/phases/P0/artifacts/RESERVED_PRIOR_EXPOSURE.csv` |
| Compute audit | `reports/roadmap/phases/P0/artifacts/COMPUTE_AUDIT.md`, `environment.json`, `model_licences_hf_api.tsv` |
| Saved assignments (unchanged) | `reports/dinov2/dev/assignments/<backbone>__centered_l2__<method>__k10__s<seed>.npy` |
| Embeddings (git-ignored) | `data/dinov2/emb/dev/{clip,siglip,dinov2_b_224_L1}.npy`; hashes in `DEV_MANIFEST.json` |
| Roadmap | `docs/roadmap/Clustering_Execution_Roadmap_W8_to_Final.docx`; original `docs/roadmap/source/…_ORIGINAL.docx`; edit script `docs/roadmap/source/update_roadmap_docx.py` |

```bash
# from the repository root, with an environment built from requirements-pilot.lock.txt
python reports/roadmap/phases/P0/source/p0_verify.py            # all stages, ≈ 15 min CPU (re-fits dominate)
python reports/roadmap/phases/P0/source/p0_verify.py dataset     # ≈ 10 s
python -m pytest tests                                          # 8 tests
```

Re-running regenerates the artifacts. `frozen_utc` / `declared_utc` change; `manifest_content_sha256` and `protocol_content_sha256` exclude those timestamps and stay the same.

## 10. Completion checklist

Copied from the execution roadmap, P0.

| # | Roadmap task | Status | Evidence |
|---|---|---|---|
| P0-001 | Revalidate 12,000 development IDs, HS3/HS1 labels, museum IDs and object grouping; verify no held-out access. | **Complete** | 74/74 dataset gates. No reserved labels were read, and nothing reserved was embedded or scored in P0. The two pre-existing exposures (F1, F2) are documented. |
| P0-002 | Fix identical evaluation masks, seeds, reference k=10, model input accounting and metric definitions (AMI, ARI, NMI, silhouette). | **Complete** | `EXPERIMENT_PROTOCOL.json` (FAIR-CMP-v1); 8/8 protocol gates |
| P0-003 | Document existing DINOv2-B/CLIP/SigLIP scores, embedding versions, preprocessors and licenses as immutable baselines. | **Complete** | `BASELINE_REFERENCE.csv` (27 rows, rescored, 51/51 k=10 fits re-fitted bit-identically); checkpoints, transforms and licences in the protocol and `model_licences_hf_api.tsv` |
| P0-004 | Audit available compute, model storage and total inference budget; create a feasible model-run schedule. | **Complete** | `COMPUTE_AUDIT.md` §1–§6 |

**DONE WHEN:** A frozen dev manifest, experiment config and a reference results table exist → **met** (`DEV_MANIFEST.json`, `EXPERIMENT_PROTOCOL.json`, `BASELINE_REFERENCE.csv`).

## 11. Next phase handoff

- **Ready for P1:** the frozen manifest, protocol and baselines; validated DINOv2-B 224 embeddings for layers −1, −2 and −4 (reuse them, no rerun needed); DINOv2-S/L weights present and hash-verified; `dinov2_embed.py --set dev --size 224` already supports the dev set.
- **Read first:** this report; `artifacts/EXPERIMENT_PROTOCOL.json`; `artifacts/COMPUTE_AUDIT.md` §5–6; `reports/dinov2/DINOV2_BENCHMARK.md` §4 (pilot RQ3 results to compare against).
- **Frozen:** `MIMO-HS3-DEV12K-v1` and `FAIR-CMP-v1`: rows and order, label masks, centered + L2, k = 10, method hyperparameters, seeds 0–4, 4 BLAS threads, metric definitions, HDBSCAN noise rule, seed aggregation.
- **Before P1 runs:** (1) recreate `.venv` from `requirements-pilot.lock.txt` (or point to the relocated venv); (2) make sure ≈ 3 GB of RAM is free for DINOv2-L (estimate).
- **Open risks:** F1/F2 (P5A, P16); audio exclusion (P7); museum confounding (P5); near-duplicates (P5); no GPU for Qwen3-VL-Embedding (P2).
- **May P1 begin?** Yes, after the team has reviewed this report. P1 was not started.

## 12. Presentation-ready summary

**What we did:** We froze one development dataset (12,000 MIMO objects, 13 HS3 classes) and one comparison protocol for all coming encoder experiments. We independently re-verified the data, the split and the Week 6 baselines: 195 automated checks, including a bit-identical re-fit of all 51 k=10 baseline clusterings.

**Main result:** All Week 6 development baselines reproduce exactly. DINOv2-B + Agglomerative gives HS3 AMI 0.561 / ARI 0.417, and SigLIP + Spectral gives 0.559 / 0.311 (12,000 objects, centered + L2, k = 10). Development and reserved data share no object or image.

**Scientific interpretation:** Later model comparisons (DINOv2 S/B/L, newer encoders) can be attributed to the model alone; data, clustering and scoring are held fixed and reproducible.

**Limitations:** Development-only numbers (one split, no CIs). 10 reserved objects were seen in an early smoke test, and the teammate's pre-split exploration covered the whole benchmark, so the final test is "untouched since the split", not never-seen. CPU-only compute.

**Recommended figure:** `reports/dinov2/figures/DINOV2_DEV_PARTITION.png` (existing; development-partition comparison of the three backbones).

**Slide-ready numbers:**

| Number | Value | Context | Source |
|---|---|---|---|
| Dev objects / HS3 classes / museums | 12,000 / 13 / 168 | MIMO-HS3-DEV12K-v1 | `DEV_MANIFEST.json` |
| Reserved / pilot-excluded | 14,696 / 651 | 0 overlap on id, image and pixel hash | `P0_VALIDATION.md` |
| Best balanced baseline | DINOv2-B + Agglomerative: AMI 0.561, ARI 0.417, largest cluster 21.6% | centered + L2, k = 10, deterministic | `BASELINE_REFERENCE.csv` |
| SigLIP + Spectral | AMI 0.559, ARI 0.311, largest cluster 34.7% | same, 5 seeds | same |
| k-Means (CLIP / SigLIP / DINOv2-B) | AMI 0.451 / 0.523 / 0.538 | same, 5 seeds | same |
| Reproducibility | 195/195 checks; 51/51 re-fits bit-identical | | `p0_summary.json`, `baseline_refit.csv` |
| P1 compute estimate | DINOv2-S ≈ 11 min, DINOv2-L ≈ 2.0 h (CPU) | estimates from pilot timings | `COMPUTE_AUDIT.md` |
