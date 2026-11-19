# P5A: OPTIONAL in Week 8, alternatively Week 10 (new-data validation)

**Status: NOT STARTED.** No work has been executed for this phase; there are no results here yet.

Milestone: WEEK 8 | 30 NOV 2026

Roadmap note: Scheduling: Option A, complete P5A during Week 8 after the bias/leakage checks. Option B, defer P5A to Week 10, before P13 pipeline freeze. It is NOT required to begin P6. This validation must not consume the protected final test.

## Roadmap tasks (verbatim from the execution roadmap)

- [ ] Before using reserved objects, audit IDs and original audio exclusions; detect exact/near duplicates and overlaps with the 12k development and 2k pilot records. If these checks are unresolved, defer P5A to Week 10.
- [ ] Before scoring any reserved object, lock an object-level split: a validation portion (e.g., ~7,000 after exclusions) for Week 8 OR Week 10 and a separate protected final-test portion for Week 14. Save exact counts and hashes.
- [ ] Predeclare a small set of backbones, preprocessing methods, clustering algorithms, seeds and the k-selection rule using development data. Never choose configurations by looking at the protected final labels or scores.
- [ ] Run the predeclared candidates on the newly designated validation objects in the chosen week. Record HS3 AMI/ARI/NMI, coverage, cluster sizes, museum distribution, run times and exact model/input settings.
- [ ] Compare 12k development with new-data validation using matching definitions. Report model-ranking changes, class/museum shifts and uncertainty; do not rebrand validation results as an untouched final test.
- [ ] Save a validation report, immutable manifests, checksums and configs. Keep all remaining protected final-test scores inaccessible until the pipeline is frozen and the Week 14 one-shot evaluation begins.

**DONE WHEN:** P5A has been completed in Week 8 OR Week 10, with a documented new-data comparison and an untouched Week 14 final-test split. If deferred in Week 8, mark 'deferred to Week 10' rather than 'done'.

## Carried in from P0 (must be handled before any reserved object is scored)

- 10 reserved `heldout` objects were embedded, clustered and scored in the 29 Oct 2026 CLIP smoke test (before the split existed). Their identifiers are in [`../P0/artifacts/RESERVED_PRIOR_EXPOSURE.csv`](../P0/artifacts/RESERVED_PRIOR_EXPOSURE.csv). Exclude them from both the validation portion and the protected final test when the split is locked.
- The teammate's Weeks 2–4 exploration (`IDEAS.md`, `spikes/clean_analysis.py`) scored k-means and a linear probe on all 27,348 benchmark rows, which include today's reserved pool. The reserved pool is therefore *untouched since the 15 Nov 2026 split*, not never-seen. State this wording in the P5A and P16 reports.
- The audio-object exclusion for the reserved pool is still not applied (see P7).

## When this phase is executed

1. Copy [`../../PHASE_REPORT_TEMPLATE.md`](../../PHASE_REPORT_TEMPLATE.md) to `P5A_REPORT.md` in this folder and fill in all 12 sections.
2. Put generated evidence (CSV, JSON, logs, figures) in `artifacts/` and scripts in `source/`.
3. Follow the frozen protocol [`../P0/artifacts/EXPERIMENT_PROTOCOL.json`](../P0/artifacts/EXPERIMENT_PROTOCOL.json) and compare against [`../P0/artifacts/BASELINE_REFERENCE.csv`](../P0/artifacts/BASELINE_REFERENCE.csv).
4. Update the status row in [`../../README.md`](../../README.md).
