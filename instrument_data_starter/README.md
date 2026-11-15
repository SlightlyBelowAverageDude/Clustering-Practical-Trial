# Instrument Clustering: Original data sources and a small Met pilot

## 1. MIMO: primary dataset

Official Europeana collection: https://www.europeana.eu/en/collections/organisation/1595-mimo-musical-instrument-museums-online

The colleague's repository already contains the download pipeline:

- `spikes/mimo_harvest.py`: queries Europeana, writes `data/mimo/records.csv` and `data/mimo/hs_vocab.csv`.
- `spikes/mimo_fetch.py`: samples and downloads preview images, writes `data/mimo/sample.csv` and `data/mimo/images/`.
- `spikes/mimo_audio_fetch.py`: audio extension.

**Required fix:** The colleague's `spikes/mimo_harvest.py` hard-codes the demo API key. This ZIP includes a corrected version, `mimo_harvest_with_key.py`, which reads `EUROPEANA_API_KEY` from the environment. Europeana requires a personal API key (free registration): https://pro.europeana.eu/page/get-api. Never commit keys.

Copy `mimo_harvest_with_key.py` into your project's `spikes/` folder (renaming it to `mimo_harvest.py` after backing up the old file). On Linux/macOS:

```bash
pip install pandas requests
export EUROPEANA_API_KEY="your-key-here"
python spikes/mimo_harvest.py
python spikes/mimo_fetch.py --n 2000 --per-provider 300 --per-hs1 600
```

This is the **full MIMO harvest** before sampling and can take time. If the API returns errors, inspect its status and response rather than assuming the demo key works. The existing script has not been live-tested in this environment.

Individual MIMO image rights must be preserved from the source metadata. Metadata is not the same as permission to redistribute image files.

## 2. The Metropolitan Museum of Art: public-domain pilot

The Met's official dataset: https://huggingface.co/datasets/metmuseum/openaccess

A **compact 25.3 MB metadata file** is available as `openaccess.csv.gz`. The full image dataset is hundreds of gigabytes and is **not** necessary for a small pilot.

Run the accompanying script from the project root:

```bash
pip install pandas requests
python fetch_met_instruments.py --per-class 40 --download-images
```

Or point to a downloaded official `MetObjects.csv`:

```bash
python fetch_met_instruments.py --metadata-file dl/MetObjects.csv --per-class 40 --download-images
```

Outputs:

- `data/met_instruments/objects.csv`
- `data/met_instruments/images/*.jpg`

It filters for `department=Musical Instruments`, `isPublicDomain=True`, and Hornbostel-Sachs-style classes. Some class groups may be small; actual counts are reported after downloading metadata. The script can resume downloads. The network-based path has not been live-tested here, since this environment cannot reach the museum APIs.

## 3. Precomputed Met embeddings

Official model embeddings can be joined by `objectID` to the Met metadata. They cover the entire Met image collection, not only instruments:

- SigLIP 2: https://huggingface.co/datasets/metmuseum/openaccess-embeddings-siglip2 (about 1.12 GB)
- OpenCLIP: https://huggingface.co/datasets/metmuseum/openaccess-embeddings-openclip-vitg14 (about 1.26 GB)
- DINOv2 giant: https://huggingface.co/datasets/metmuseum/openaccess-embeddings-dinov2-giant (about 1.5 GB)

The colleague's repository has a related implementation at `spikes/instruments_met_large.py`. Note that DINOv2 is a visual encoder, not a vision-language model; include CLIP or SigLIP to satisfy the core of Topic 1.

## Important for benchmarking

- Never use labels to generate clusters. Labels are for evaluation.
- Keep stable object IDs and image IDs. Multiple images per instrument must not silently count as independent objects.
- Preserve the per-image rights information.
- Compare different methods on identical rows and encoder settings.
- MIMO numeric Hornbostel-Sachs codes and the Met's textual classification fields are not automatically identical; document mappings explicitly.
