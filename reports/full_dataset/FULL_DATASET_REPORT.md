# Full MIMO dataset: reconstruction report (P5)

Built 15 Nov 2026 with the project's existing Europeana API configuration: the key is read from the git-ignored `.env`; it is never printed, logged or committed. Every count below was recomputed from the download, not copied from the research plan. Machine-readable versions: `stats.json`, `manifest.json`, `class_distributions.csv`, `museum_distribution.csv`. Figures: `figures/FULL_DATASET_{FUNNEL,CLASSES,MUSEUMS}.png`.

## 1. Pipeline and reuse

| Step | Script | Reuse |
|---|---|---|
| API check + metadata | `spikes/pilot_acquire.py` (`verify`, cached resumable `harvest`, `resolve_vocab`) via `full_acquire.py` | Existing key-redacting harvester. The live count on 15 Nov 2026 is **63,478** records, identical to the cached 636-page snapshot (29 Oct 2026), so the snapshot was reused |
| Images | `reports/full_dataset/source/full_acquire.py` → `pilot_acquire.download_one` | Europeana preview first, source-image fallback. Decode, size and near-uniform validation; byte + pixel sha256; content-addressed files (the 2,062 pilot images were reused). Bounded retries with exponential backoff and `Retry-After`. Ledger written in chunks, so reruns resume and only retry failures (the rerun retried the 5 failures; still failing) |
| Names / families | `wikidata_names.py` → `spikes/mimo_names.py` (**teammate, unchanged**) | The two Wikidata tables `mimo_names.py` reads (`dl/wd_mimo_*.csv`) were never committed and are re-queried (Wikidata SPARQL, 15 Nov 2026; queries in `wikidata_query.json`) |
| Cleaning | `spikes/mimo_clean.py` (**teammate, unchanged**) | Placeholder, shared-photo copies, instrument parts |
| Benchmark + split | `full_build.py` | Benchmark rule copied verbatim from `spikes/clean_analysis.py` step 2; development subset drawn exactly as step 6 |

**Teammate code discrepancy, documented rather than silently changed:** the docstring of `spikes/mimo_full.py` says uncoded records stay in the set, and the documented counts (59,661 cleaned, 40,697 coded) require that. The committed filter, however, keeps only coded rows. The reconstruction follows the **documented** protocol: all records with a valid image are kept.

## 2. Counts: recomputed vs research plan

| Stage | Recomputed | Plan | Note |
|---|---|---|---|
| Europeana MIMO image records | **63,478** | 63,478 | identical |
| Records with an HS concept URI | 41,210 | 41,210 (`mimo_full.py` docstring) | identical |
| Valid downloaded images | **63,473** | — | 5 failed: preview rejected (`ValueError`: too small or undecodable) and source host SSL error, on both tries |
| Excluded: Europeana placeholder | 247 | 247 | identical. Audit: the most frequent image (293×293) occurs 247×; the next most frequent only 28× |
| Excluded: copies of a shared photo | 556 | 556 | identical (later byte-identical copies) |
| Excluded: instrument parts | 3,012 | 3,014 | bows 1,513, mouthpieces 1,434; by title: drumsticks, beaters, … |
| **Cleaned photographs** | **59,658** | 59,661 | −3 (the 5 undownloadable records, net of their exclusions) |
| Cleaned, with HS code | **40,694** | 40,697 | −3 |
| Cleaned, with instrument name / family | 40,229 / 34,268 | 40,230 / 34,098 | the family tree comes from today's Wikidata (more items now reach a top family) |
| **Benchmark objects** (HS3 + family + name, classes ≥ 30) | **27,347** | 27,348 | −1 |
| Benchmark classes HS1 / HS2 / **HS3** / HS4 / HS5 / HS6 | 5 / 8 / **13** / 19 / 33 / 38 | – / 8 / **13** / – | **13 HS3 classes reproduced** |
| Benchmark family / name classes | 6 / 129 | 6 / 129 | identical |
| Benchmark rows with HS4 / HS5 / HS6 | 26,180 / 23,532 / 21,475 | 26,181 / 23,501 / 21,445 | small Wikidata-driven differences |

The reconstruction is faithful to within ±3 objects at every stage. The remaining differences come from 5 unreachable images and from live Wikidata edits since October; forcing the old numbers was not attempted.

## 3. Class distributions

- **HS1, cleaned set:** aerophones 20,515; no HS code 18,964 (31.8% missing); chordophones 10,790; idiophones 6,850; membranophones 2,307; electrophones 232.
- **HS3, benchmark:** 421 (7,039), 422 (4,943), 321 (4,623), 423 (3,602), 111 (2,171), 112 (1,516), 211 (1,069), 412 (866), 314 (712), 322 (591), 531 (89), 513 (86), 212 (40). The largest class holds 25.7%. HS1 in the benchmark: aerophones 60.2%.
- **Family, benchmark:** woodwind 12,179, string 5,348, percussion 4,793, brass 3,610, keyboard 1,329, electronic 88.
- **Missing data, cleaned set:** HS1 18,964; HS3 19,035; name 19,429; family 25,390; region 24,241. Ambiguous multi-concept records (2–3 HS URIs: 1,030) follow the teammate rule: the first code is used (`common.hs_level`). The pilot instead treated disagreeing concepts as missing; both rules are documented.

## 4. Museums, duplicates and multiple images

- 216 museums in the cleaned set and 195 in the benchmark. **The 5 largest museums hold 53% of the benchmark** (museum confounding risk; see the figure). Coverage varies strongly: Tervuren contributes 8,435 records but few benchmark rows (largely uncoded).
- **Exact duplicates:** 62,671 distinct pixel hashes among 63,473 images. The byte-identical copies are removed by the cleaning step (placeholder + shared photos). **No identical pixels cross the development/held-out boundary** (checked on pixel sha256).
- **Near duplicates** (same object photographed twice, or different objects on one shared background) are *not* detected on the full set. The pilot screen found 191 candidate pairs in 2,000 images, so near-duplicate leakage between partitions remains possible. Detecting it requires embedding the held-out partition, which was deliberately not done.
- **Multiple images per object:** each Europeana record is one museum object. One image per record is used (the first preview, otherwise the source image); object IDs (`id`) are preserved in every table.

## 5. Benchmark split (object level)

The original split IDs (`runs/clean/methods_subset_ids.csv`) are not available. The replacement is deterministic and follows the original rule:

| Partition | Objects | Rule |
|---|---|---|
| `dev` | **12,000** | `np.random.default_rng(0).choice(27,347, 12,000)`, sorted, exactly the `clean_analysis.py` methods-subset rule; 465 of these were already explored in the pilot |
| `heldout` | **14,696** | all remaining benchmark objects… |
| `excluded_pilot_overlap` | 651 | …except those already explored with labels in the 2,000-image pilot (removed from held-out) |

All 13 HS3 classes appear in both partitions (smallest class: dev 23, held-out 17). The Week 4 plan also removes objects with audio recordings from held-out. The audio object list is not available locally, so this is **not yet applied**: a to-do before the held-out run. Held-out IDs are in `benchmark/heldout_ids_DO_NOT_USE_FOR_SELECTION.csv`; they were not embedded, clustered or scored.

## 6. Rights and provenance

- 63,477 of 63,478 records are **CC BY-NC-SA 4.0**; 1 is In Copyright (`InC-OW-EU`) and is not in the benchmark. All 27,347 benchmark objects are CC BY-NC-SA 4.0.
- Provenance per object: Europeana record ID, holding museum (`provider`), image URL, byte and pixel sha256 (`data/mimo_full/download_ledger.csv` locally; benchmark rows in `benchmark/benchmark_objects.csv`).
- **Images are not committed.** The 63,473 images (2.40 GB) stay in the git-ignored `data/mimo_pilot/images/`, with hashes listed in the ledger. No remote backup exists; they are reproducible with the commands below.

## 7. Reproduce

```bash
# .env must contain EUROPEANA_API_KEY=... (never commit it)
.venv/bin/python reports/full_dataset/source/full_acquire.py --workers 6   # resumable; reruns retry failures only
.venv/bin/python reports/full_dataset/source/wikidata_names.py             # dl/wd_mimo_*.csv
.venv/bin/python reports/full_dataset/source/full_build.py                 # runs teammate mimo_names.py + mimo_clean.py, builds benchmark + split
.venv/bin/python reports/full_dataset/source/full_figures.py
```

Runtime here: about 35 min download (6 workers, 63,473 images, 2.4 GB), about 2 min cleaning and benchmark.
