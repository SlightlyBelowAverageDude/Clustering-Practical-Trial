# Week 6 final deck: slide sources

Deck: `Week6_Final_FullMIMO_StatusUpdate.pptx` / `.pdf` (A4 landscape, 10 main slides + 5 backup). Generator: `source/build_final_deck.js`, which uses the same theme and helpers as the earlier Week 6 deck (`reports/week6/source/build_deck.js`). Notes: `source/notes_en.js`. **Every number on the slides and in the notes is read from `final_claims.json`**, which `source/final_verify.py` recomputes from the saved development assignments, `dev_ids.csv` labels and the dataset tables. The earlier deck (`../Week6_Clustering_StatusUpdate.*`) is unchanged.

**Common context:**
- Development benchmark: 12,000 objects, 13 HS3 classes (`reports/full_dataset/benchmark/dev_ids.csv`).
- Encoders: CLIP ViT-B/32, SigLIP ViT-B/16, DINOv2-B/14 at **224 px**.
- Default configuration: centered + L2, k = 10 (predeclared). k-Means, Spectral and GMM are the mean of seeds 0–4.
- Development results only; the held-out set was not used.

| Slide | Content | Source |
|---|---|---|
| 1 Title | RQ wording; collection size | `final_claims.json` → dataset |
| 2 Progress | 63,478 / 59,658 / 27,347 / 12,000 / 14,696; DINOv2 scope | `reports/full_dataset/stats.json`, `benchmark/benchmark_objects.csv`, `reports/dinov2/embeddings/*.json` |
| 3 Dataset | Funnel incl. 63,473 valid, 651 excluded; 13 / 6 / 129 classes; dev HS3 bar chart `figures/FINAL_DEV_HS3_CLASSES.png` | `data/mimo_full/download_ledger.csv` (local), `reports/full_dataset/{clean_records.csv, benchmark/benchmark_objects.csv}` |
| 4 Pipeline | Encoders, preprocessing, methods, metrics | `reports/dinov2/DINOV2_PROTOCOL.json`, `reports/dinov2/embeddings/dev_*.json` |
| 5 RQ1 backbones | `figures/FINAL_RQ1_BACKBONES.png`; 0.561, 0.538, 0.559 vs 0.476, ARI 0.335 vs 0.314 | `reports/dinov2/dev/{summary,scores}.csv`, assignments `reports/dinov2/dev/assignments/` |
| 6 RQ1 methods | `figures/FINAL_RQ1_AMI_VS_ARI.png`; 0.561/0.417, 0.559/0.311, 0.476/0.165, largest clusters 21.6% / 34.7% / 50.1% | same |
| 7 RQ2 | `figures/FINAL_RQ2_PREPROCESSING.png`; 0.002→0.512, 0.001→0.542, 0.361→0.561, 0.568; 99.8% / 68.6% largest cluster | same (agglomerative, k = 10) |
| 8 Limitations | 12,000 / 14,696 / 59,658; 53% top-5 museums; audio exclusion pending | `final_claims.json` dataset; `reports/full_dataset/FULL_DATASET_REPORT.md` |
| 9 Roadmap | Week 4 schedule | Week 4 deck; `reports/week6/P4_NEXT_STEPS.md` |
| B1 | `reports/dinov2/figures/DINOV2_DEV_PARTITION.png` | `reports/dinov2/dev/summary.csv` |
| B2 | k = 10 vs k = 13 table (label-informed reference) | `final_claims.json` scores `…|k13` |
| B3 | `reports/full_dataset/figures/FULL_DATASET_MUSEUMS.png` | `reports/full_dataset/museum_distribution.csv` |
| B4 | `reports/full_dataset/figures/FULL_DATASET_FUNNEL.png` | `reports/full_dataset/stats.json` |
| B5 | `reports/dinov2/figures/DINOV2_SCALING.png`: **exploratory, 2,000-image pilot**, labelled as such | `reports/dinov2/pilot/summary.csv` |

**Pilot material removed from the main slides:** all 2,000-image pilot charts and numbers (Week 6 v1 slides 4–7). This includes the museum-photo contact sheet, so **the final deck contains no museum photographs**, and no image attribution is needed for it. The only pilot chart (B5) is a backup slide marked exploratory.
