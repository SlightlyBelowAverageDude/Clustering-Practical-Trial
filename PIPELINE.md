# MIMO acquisition and first CLIP benchmark

The existing spike collection is preserved. `spikes/mimo_harvest.py` now uses
the environment-key harvester in `spikes/pilot_acquire.py`, integrating the
correction in `instrument_data_starter/mimo_harvest_with_key.py`. The new pilot
uses `data/mimo_pilot/` so it does not mix files with previous research runs.
Historical claims in `EXPLAINER.md` are context, not results of this run.

## Reproduction

Run from `Clustering-Practical-Trial-main/`. Configure `EUROPEANA_API_KEY` in the
environment, or in the ignored local `.env`. Environment values take precedence.
Do not put the key on the command line. Requests exception URLs are suppressed;
response key echoes and credential query parameters are removed before caching.

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.15.0+cpu torchvision==0.29.2+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-pilot.lock.txt
.venv/bin/python spikes/pilot_acquire.py --verify-only
.venv/bin/python spikes/pilot_acquire.py --max-records 500 --n 32 --tag smoke
.venv/bin/python spikes/pilot_benchmark.py --tag smoke --ks 5 10 --batch-size 8
.venv/bin/python spikes/pilot_acquire.py --n 2000 --tag pilot
.venv/bin/python spikes/pilot_benchmark.py --tag pilot
.venv/bin/python spikes/pilot_validate.py --tag pilot
.venv/bin/python spikes/pilot_report.py
.venv/bin/python -m pytest tests/test_pilot.py -q
```

Repeat acquisition with the same command to resume cached pages and validated
downloads. Vocabulary errors are retried on rerun. To keep the exact original
metadata snapshot and avoid an API request, use `--reuse-metadata`. Download
failures are retried on rerun; service changes can affect which candidates fill
the target. Preserve the saved sample CSV and image hashes to reproduce the exact
benchmark. Do not run acquisition concurrently against the same output tag.

## Protocol

- Query: all Europeana records with provider `MIMO - Musical Instrument Museums
  Online` and `TYPE:IMAGE`; cursor pagination, 100 records per page. The whole
  metadata snapshot is harvested before selecting the full pilot. Throttling,
  bounded retries, `Retry-After`, atomic page checkpoints and unique record IDs
  handle network failures and interrupted runs. Cursor continuation is checked.
- Sampling seed 42. Records are ordered by SHA256(seed:record ID) within museum,
  then selected in museum round-robin order, replacing invalid or duplicate
  images until the target is reached. This overrepresents small museums. It is
  not an estimate of the overall MIMO distribution. Missing HS labels remain.
- Use Europeana previews first, then source-image fallbacks. Keep the actual URL,
  original bytes, byte SHA256, decoded RGB SHA256, dimensions, Europeana record ID,
  and SHA256(record ID + URL) image identifier. Four download workers; results
  consumed in candidate order. Validate decoding, minimum dimensions, and reject
  near-uniform images. Exact pixel and URL duplicates do not enter the sample.
  Failed URLs, duplicate candidates and unused prefetches remain in the ledger.
- Resolve every source HS URI using MIMO Skosmos; URI numeric IDs are not HS
  codes. Retain all source codes. HS levels use the first one, two or three
  hierarchy digits before `-` modifiers and the first branch before `+`. If
  multiple concepts disagree at a level, or any concept is unresolved, that
  label is missing for evaluation. Source titles are instrument names; they are
  not inferred or used as features. Origin is retained verbatim, not equated to
  provider country or automatically interpreted as manufacture location.
- Frozen OpenCLIP ViT-B/32, `laion2b_s34b_b79k` weights. Immutable Hub revision,
  weight SHA256, official preprocessing configuration and installed versions
  are saved in the embedding manifest. EXIF transpose, RGB, white square padding
  to preserve long instruments, official 224-pixel evaluation transform. CPU
  float32, deterministic inference, seed 42, four compute threads by default.
- Raw representation: L2-normalized 512-dimensional embeddings. PCA variant:
  centered, non-whitened PCA with 50 components (bounded by sample size), full
  deterministic SVD, then L2 normalization. PCA fits every sampled image without
  labels. Retained variance and PCA components are saved.
- K-Means: Euclidean Lloyd algorithm, `n_init=10`, `max_iter=300`, `tol=1e-4`,
  seeds 0, 1, 2. Average-link agglomeration: cosine distance, deterministic, one
  run per configuration. Fixed k in 5, 10, 20, 40; smoke uses 5 and 10. These
  values were set before seeing scores and are not selected from label counts.
- Every clustering method uses every sampled image in the same order. Labels
  never enter embedding, PCA, or clustering. Evaluate each prediction on the
  nonmissing-label mask for HS1, HS2, HS3, preserving that mask for all methods.
  No minimum class-frequency filtering. Save AMI, ARI, NMI, evaluated counts,
  actual k, seeds, clustering runtime and representation preprocessing runtime.
- Each image's bytes are rehashed before embedding. Cache keys include model,
  preprocessing, package versions, image ID and content hash; array hashes,
  dimensions, finiteness and norms are checked. Missing or mismatched cache
  provenance is rejected. Aggregate input hashes include the exact row order.
- `scores.csv`, `assignments.csv`, `image_order.csv`, manifests, environment,
  completion record and `comparison.png` are under `results/<tag>/`. Comparison
  bars show mean +/- one standard deviation over K-Means seeds; agglomeration
  is a single deterministic run. Gallery uses raw embeddings, K-Means, smallest
  preselected k and first seed, independently of label scores; it shows images
  nearest each cluster mean with source and rights attribution. Its thumbnails
  remain in ignored `data/`; the HTML links to them using relative paths.

The pilot uses stricter cache validation than the old spikes and explicitly
requests average linkage. The old `common.cluster_and_score` still defaults to
Ward and must not be used to reproduce this benchmark.

## Sources

- [Europeana Search API documentation](https://europeana.atlassian.net/wiki/spaces/EF/pages/2385739812)
- [Official frozen model repository](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)
- [OpenCLIP implementation](https://github.com/mlfoundations/open_clip)
- MIMO vocabulary service: `https://vocabulary.mimo-international.com/rest/v1/HornbostelAndSachs/data`
