# Integration report: Week 6 research (P1–P4) into `Aiman-testing`

**Target:** `https://github.com/SlightlyBelowAverageDude/Clustering-Practical-Trial`, branch `Aiman-testing`.

**Base:** commit `423f6ba` ("Add the name-closeness tests"). At integration time `Aiman-testing` was identical to `main` (63 tracked files).

**Source of the integrated work:** the local workspace `ClusteringmeetsVisionandLanguage/Clustering-Practical-Trial-main/`. It was an untracked copy of the same project, with the pilot pipeline, the phase-2 results and the Week 6 reports added. A second folder, `ClusteringmeetsVisionandLanguage/instrument_data_starter/`, held the course's data-starter scripts.

## 1. Teammate files: all preserved unchanged

All 63 tracked files keep their content. The one exception is `.gitignore`, which was extended, not replaced (see §3).

These were checked byte-for-byte against `origin/Aiman-testing`:
- `EXPLAINER.md`, `IDEAS.md`, `.gitattributes`
- all 59 files in `spikes/`, including the four that existed only on the branch: `rock_bottom.py`, `test_name_similarity.py`, `test_name_similarity2.py`, `test_name_similarity3.py`.

## 2. Conflicts and how they were resolved

| File | Situation | Resolution |
|---|---|---|
| `spikes/mimo_harvest.py` | The workspace version had been rewritten to read `EUROPEANA_API_KEY` from the environment and delegate to `pilot_acquire.py`. The teammate's version uses the public `api2demo` key. | **Teammate version kept unchanged.** The keyed version was added as `spikes/mimo_harvest_keyed.py`, with the same CSV outputs (`data/mimo/records.csv`, `hs_vocab.csv`). `PIPELINE.md` still describes the keyed entry point under its old name; use `mimo_harvest_keyed.py`. Note: `pilot_benchmark.py` stores a provenance hash of `spikes/mimo_harvest.py`; a rerun will record the teammate file's hash. Nothing asserts on it. |
| `.gitignore` | The workspace added `.env` rules. | Merged: all teammate rules kept; added secrets (`.env`, `.env.*`, `!.env.example`), model weights (`*.safetensors`, `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`) and caches. Precise exceptions make the research evidence logs under `results/` and `reports/` trackable. Verified with `git check-ignore -v`. |

## 3. Added files

| Path | Content |
|---|---|
| `PIPELINE.md`, `FIRST_RESULTS.md`, `requirements-pilot.lock.txt`, `.env.example` | Pilot pipeline documentation, first results, pinned dependency lock, key template (no key) |
| `spikes/pilot_acquire.py`, `pilot_benchmark.py`, `pilot_validate.py`, `pilot_report.py`, `spikes/mimo_harvest_keyed.py` | Validated, resumable pilot acquisition and benchmark |
| `tests/test_pilot.py` | Pilot tests |
| `results/pilot/`, `results/smoke/`, `results/phase2/` | Pilot and phase-2 (CLIP/SigLIP ablation) results, sources, logs, figures, galleries (HTML only; photos stay in ignored `data/`) |
| `reports/week6/` | P1 audit, P2 RQ1 benchmark, P3 robustness/AMI-ARI/RQ2/RQ3-readiness, P4 validation, Week 6 slides (PPTX/PDF), speaker notes, all scripts, assignments and figures |
| `reports/week6/figures/ATTRIBUTION.md` | New: per-image CC BY-NC-SA 4.0 attribution for the 23 museum thumbnails in one figure and on slide 5 |
| `instrument_data_starter/` | The course data-starter scripts (`README.md`, `fetch_met_instruments.py`, `mimo_harvest_with_key.py`); they contain no key |
| `reports/full_dataset/`, `reports/dinov2/`, `reports/integration/` | P5 work (this phase) |

## 4. Not committed (intentionally)

| What | Why | Where it is |
|---|---|---|
| `.env` (Europeana key) | Secret | Local only; ignored. Key never printed or logged |
| `data/` | About 63,000 museum images (≈ GBs), CLIP/SigLIP/DINOv2 caches, model weights (≈ 1.6 GB DINOv2) | Local only; reproducible with the documented commands. Sizes and hashes are in `reports/full_dataset/manifest.json` and `reports/dinov2/embeddings/*.json` |
| `runs/`, `dl/` | Intermediate teammate-pipeline outputs and Wikidata dumps (regenerable) | Local only. Curated copies of the needed tables are in `reports/full_dataset/` |

## 5. Validation after integration (run inside the clone)

| Check | Result |
|---|---|
| Existing tests (`tests/test_pilot.py`, `results/phase2/source/test_phase2.py`) | **11 / 11 pass** |
| Week 6 independent verification (`reports/week6/source/p4_verify.py`) | **299 / 299 checks pass**; all 26 hashed P1–P4 artifacts byte-identical after the move |
| Imports of teammate and pilot modules (`common`, `labels`, `mimo_fetch`, `mimo_full`, `mimo_clean`, `mimo_names`, `clean_analysis`, `rock_bottom`, `test_name_similarity`, `method_compare`, `pilot_acquire`, `pilot_benchmark`, `mimo_harvest`, `mimo_harvest_keyed`) | All import |
| Syntax check of every `spikes/*.py` | Pass |
| Secret scan (real key value searched in all files to be committed) | Not found. The only `wskey=` literals are the public `api2demo` demo key in teammate code and a fake value in a test |
| Largest committed file | 24 MB (`reports/full_dataset/clean_records.csv`); no file above GitHub's 50 MB warning limit |
| P5 independent verification (`reports/dinov2/source/p5_verify.py`) | **812 / 812 checks pass**: DINOv2/CLIP/SigLIP scores recomputed from assignments, embedding and weight hashes, pilot row order, dev/held-out isolation, benchmark counts, secret scan, file sizes |
| Tracked teammate files changed | Only `.gitignore` (+19 lines, additions only); verified with `git diff --stat` |
| Commit size | about 3,630 new files, ≈ 212 MB (results, reports, small assignment arrays); no images, weights or `.env` |

## 6. Known path caveat

Several historical manifests (`results/pilot/embedding_manifest.json`, `results/phase2/**/embedding_manifest.json`, `data/*/model.json`) record **absolute paths** from the original workspace (`.../ClusteringmeetsVisionandLanguage/Clustering-Practical-Trial-main/...`). They are kept as historical evidence and were not rewritten. The new P5 scripts map these paths to the repository root. Phase-2 scripts that assert on `array_path` need the original workspace to exist, or the path to be adjusted, on another machine.
