# Progress presentations: workflow

No presentation has been made from the roadmap phases yet. The team will put its PowerPoint template in a dedicated folder later. Until it is there, nothing in this folder should be turned into slides.

## Milestones

| Milestone | Date | Phases that feed it | Phase that builds the deck |
|---|---|---|---|
| Week 8 | 30 Nov 2026 | P0–P7, plus P5A if it ran in Week 8 | P8 |
| Week 10 | 14 Dec 2026 | P9–P13, plus P5A if it was deferred | Week 10 exit checklist |
| Week 14 | 11 Jan 2027 | P14–P16 | P17 |
| Final | 25 Jan 2027 (talk + demo), 1 Feb 2027 (report) | everything | P19 (talk), P20 (report) |

## Workflow

1. **Finish a research phase.** Run it under the frozen protocol in [`../phases/P0/artifacts/EXPERIMENT_PROTOCOL.json`](../phases/P0/artifacts/EXPERIMENT_PROTOCOL.json).
2. **Validate its results.** Recompute the headline numbers from saved assignments or score files, using a script, before anything is written up.
3. **Write the phase report** at `../phases/<PHASE_ID>/<PHASE_ID>_REPORT.md`, using [`../PHASE_REPORT_TEMPLATE.md`](../PHASE_REPORT_TEMPLATE.md).
4. **Extract the presentation-ready summary** (section 12 of the report). It is the only text that moves from a report onto slides.
5. **When a milestone is reached, combine the section-12 summaries** of every phase completed for that milestone, in a file such as `presentations/<milestone>/SLIDE_SOURCES.md`. Each slide gets one row: slide number, claim, exact number, phase report, artifact path.
6. **Build the deck from the template the team provides.** Do not design a new layout. Put the template in `presentations/template/` (or the folder the team chooses) and record its file name and hash in the milestone's `SLIDE_SOURCES.md`.
7. **Add verified plots, tables and interpretation.** Use only figures that exist under `reports/` and are named in a phase report. Where a figure is regenerated, commit the script that made it.
8. **Write speaker notes and source references.** Every slide gets notes that say what was done, the result, what it means and the limitation, plus the artifact path for each number.
9. **Check the deck with a script.** Every number on a slide must be found in, or recomputed from, the cited artifact (as `reports/week6/presentation/final/source/final_verify.py` did for Week 6). Commit the check output next to the deck.

## Rules

- Every slide must trace back to a phase report and a saved artifact.
- Planned, running or deferred work is labelled as such. A result is never claimed before its phase report says COMPLETE (or PARTIAL, with the missing part named).
- Development results (MIMO-HS3-DEV12K-v1), optional P5A validation results and the Week 14 protected final test always appear in separately labelled tables or panels.
- Fixed k = 10 and the label-informed k = 13 reference are never shown as blind-k results.
- Slides show the sample size and the dataset version for every number.
- Week 6 material in `reports/week6/presentation/` is historical. It is not edited; new decks go in new folders.
