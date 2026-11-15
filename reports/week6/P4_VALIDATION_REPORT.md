# P4: Independent validation report (Week 6)

Script: `reports/week6/source/p4_verify.py`. It deliberately does **not** reuse the P2/P3 aggregation code (`p2_summarize.py`, `p3_experiments.py robust_score`). It loads the saved assignment `.npy` files, the hash-checked representations and the phase-2 evaluation masks (whose sha256 is checked against `results/phase2/inputs/input_identity.json`), recomputes every metric with plain scikit-learn calls, and compares the results with the reported CSVs.

Outputs: `p4_validation/checks.csv` (one row per check), `p4_validation/presentation_claims.json` (the only number source the deck generator reads), and `p4_validation/artifact_hashes.json` (sha256 of every report, CSV, JSON and figure).

**Result: 299 / 299 checks pass** (tolerance 1e-9 for metrics; exact for selected k and counts).

| # | Check | Rows | Result |
|---|---|---|---|
| 1 | Best Spectral HS3 AMI: SigLIP centered + L2, k=10, mean of 5 seeds = 0.544 (SD 0.001); CLIP 0.470 | 2 | pass |
| 2 | k-Means, agglomerative and GMM comparisons at k=10 for both encoders: AMI mean, ARI mean, AMI seed SD, largest cluster % | 30 | pass |
| 3 | SigLIP vs CLIP count, recomputed from the 448 k-requiring assignment files: **51 / 56** cells | 2 | pass |
| 4 | Blind k (P2): cosine silhouettes recomputed from the saved representations for all 448 k-requiring runs; maximum reselected; selected k and HS3 AMI at that k match for all 16 curves | 32 | pass |
| 5 | AMI vs ARI decomposition (pair precision and recall, share of pairs in the largest cluster, largest-cluster purity) | 8 | pass |
| 6 | Preprocessing results: AMI and largest-cluster % for all 90 rows (2 encoders × 5 representations × 3 methods × 3 k) | 180 | pass |
| 7 | Subset robustness: AMI for all 30 encoder × subset × method rows, using subset rows and masks; 4 subset blind-k silhouette reselections | 34 | pass |
| 8 | Numerical precision: in the float64 sensitivity run, Spectral > k-Means AMI and agglomerative > Spectral ARI at k=10 still hold for both encoders | 4 | pass |
| 9 | Original pilot snapshot (59 files), phase-2 sources (11) and phase-2 reports (4) unchanged; dataset 216 museums, 22 HS3 classes | 5 | pass |
| | **Total** | **299** | **299 pass** |

Earlier, independent of this script:
- **P1** recomputed all 816 historical metric rows exactly from saved assignments (`P1_BASELINE_RESULTS.csv`).
- **P2** reproduced the phase-2 k-Means and agglomerative assignments exactly on all 216 overlapping rows (`experiments/phase2_consistency_check.csv`).
- **P3** reproduced 60/60 phase-2 subset predictions and 240/240 earlier assignments.

## Figure and deck checks

- All P2/P3 figures are drawn only from the saved CSV/NPY files by `p2_figures.py` and `p3_figures.py`. The numbers in figure subtitles are computed from those files (for example the 4/5 and 5/5 counts and the eigengap k sets), not typed by hand. Figure hashes are in `artifact_hashes.json`.
- **Deck:** every decimal number in the slide text and notes was cross-matched against the verified claims and result CSVs. 81 decimals were found. All match except dates, speaking times, and the "≈1.6 GB" download size, which comes from the parameter counts in `p3_experiments/rq3/`. `validate.py` (OOXML schema/relationships) passes, and the PDF export was rendered to images and inspected slide by slide.
- **Visual QA fixes applied:** overlapping card headings (slides 5 and 8), a caption overflowing into the footer (slide 3), and figure label collisions (P2/P3 figures).

## Statistical reporting rules applied

- Stochastic methods are reported as mean ± SD over seeds 0–4. No best-seed selection.
- No significance claims. Seed agreement and subset agreement are reported as descriptive consistency only: seeds are not independent samples, and the subsets overlap (1,600 of the same 2,000 rows).
- "Best" configurations (SigLIP + Spectral) were identified **after** examining development results. This is stated on the slides (slide 4 caption) and in the notes.
- The research plan's 0.53 / 0.83 / 0.68 are never compared with pilot numbers (different dataset, class count, encoder and k rule).

## Known caveats not resolved by validation

- Validation proves that the numbers are reproducible and correctly computed. It does **not** establish generalization: the pilot was explored, nothing is held out, and there is no museum- or specimen-level resampling.
- Some catalogue HS labels look inconsistent (for example "Tambour" coded 111). Labels were used as recorded, and the 180-image human review is still blank.
- The 868 paired-audio count on slide 8 is quoted from Week 4 / `IDEAS.md` and was **not** recomputed.
