# Research reports: map

Everything here was produced by the scripts in each folder's `source/`, run from the repository root with the pilot environment (`requirements-pilot.lock.txt`). Images, embeddings and model weights live in the git-ignored `data/`; this folder holds the code, protocols, result tables, figures and validation evidence.

| Folder | What | Start with |
|---|---|---|
| `week6/presentation/final/` | **Final Week 6 deck** (full MIMO collection + 12,000-object development benchmark), EN/DE speaker notes, slide sources, validation (224/224) | `week6/presentation/final/Week6_Final_FullMIMO_StatusUpdate.pdf` |
| `week6/` | P1 foundation audit, P2 RQ1 benchmark (5 clustering families × CLIP/SigLIP, blind k), P3 robustness, AMI-vs-ARI, RQ2 preprocessing, P4 validation (299/299) and the 16 Nov slides | `week6/P4_WEEK6_FINAL_REPORT.md`, `week6/presentation/` |
| `full_dataset/` | Reconstructed full MIMO dataset: 63,473 images, 59,658 cleaned photographs, 27,347-object benchmark (13 HS3 classes), object-level dev/held-out split | `full_dataset/FULL_DATASET_REPORT.md` |
| `dinov2/` | DINOv2 S/B/L (official weights), three-backbone comparison, RQ3 size/layer/resolution, linear-probe gap, development-partition benchmark, P5 validation | `dinov2/DINOV2_BENCHMARK.md` |
| `integration/` | How the Week 6 work was merged into this branch; what was preserved, added and excluded | `integration/INTEGRATION_REPORT.md` |

Older pilot and phase-2 results are in `../results/` (`pilot/`, `smoke/`, `phase2/`); see `../PIPELINE.md`, `../FIRST_RESULTS.md` and `../results/phase2/PROTOCOL.md`.

**Held-out rule:** `full_dataset/benchmark/heldout_ids_DO_NOT_USE_FOR_SELECTION.csv` lists the reserved objects. Do not embed, cluster or score them until the pipeline is frozen (Week 10 plan).
