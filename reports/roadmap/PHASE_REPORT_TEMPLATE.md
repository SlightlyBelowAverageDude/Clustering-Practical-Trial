# <PHASE_ID>: <Phase name exactly as in the execution roadmap>

<!--
How to use this template
- Copy to reports/roadmap/phases/<PHASE_ID>/<PHASE_ID>_REPORT.md once the phase has actually been executed.
- Keep all 12 sections, in this order. If a section does not apply, write "Not applicable" and give the reason.
- Every number must come from a saved file listed in section 9. Quote the exact value; never retype a number from memory or from slides.
- Label everything that is not a direct measurement: "Estimate", "Interpretation" or "Hypothesis".
- Do not delete negative or failed results. Missing experiments are listed, not hidden.
- Development results (12,000-object manifest MIMO-HS3-DEV12K-v1), optional P5A validation results and the
  Week 14 protected final test are never mixed in one table without a column that says which is which.
- Remove these comments in the finished report.
-->

## 1. Phase information

| Field | Value |
|---|---|
| Phase ID and name | <P#  \|  name> |
| Research questions addressed | <RQ1–RQ5, from the roadmap heading> |
| Milestone | <Week 8 (30 Nov 2026) / Week 10 (14 Dec 2026) / Week 14 (11 Jan 2027) / Final (25 Jan + 1 Feb 2027)> |
| Execution date(s) | <YYYY-MM-DD – YYYY-MM-DD> |
| Status | <NOT STARTED / IN PROGRESS / COMPLETE / PARTIAL / BLOCKED / DEFERRED> |
| Git branch and commit | <branch> @ <commit SHA of the commit that adds this report's results> |
| Dataset version | <manifest_id + manifest_content_sha256, e.g. MIMO-HS3-DEV12K-v1> |
| Protocol version | <protocol_id + protocol_content_sha256, e.g. FAIR-CMP-v1> |
| Executed by | <person / agent> |

## 2. Objective

<What this phase was supposed to investigate, in two or three sentences.>

**Scientific question:** <the one question this phase answers, phrased so the answer can be yes/no or a number.>

**Hypothesis (stated before running):** <expected outcome, or "none: descriptive phase".>

## 3. What was already available

| Artifact | Path | Reused how | Verified? |
|---|---|---|---|
| <code / data / embeddings / earlier results> | <path> | <as-is / extended / re-run> | <hash checked, rescored, …> |

**New work in this phase:** <one sentence that separates reused material from what this phase produced.>

## 4. What was done

<Numbered list of the steps actually executed, in order. Include data checks, models run, methods tested, preprocessing,
tests, and every configuration change against the frozen protocol (with the reason). Do not copy the planned task list.>

1. …
2. …

**Deviations from the plan:** <none / list with reason>

## 5. Experimental setup

| Item | Value |
|---|---|
| Dataset and object counts | <manifest id, n objects, n evaluated per label level> |
| Models and checkpoints | <repo@revision, weight sha256, params, embedding dim> |
| Image input | <adaptation, native vs used resolution, transform> |
| Feature representation | <e.g. centered_l2> |
| Clustering algorithms and hyperparameters | <exact settings or "per FAIR-CMP-v1"> |
| k policy | <fixed k=10 / label-informed k=13 (reference) / blind k rule> |
| Random seeds | <list> |
| Evaluation metrics | <AMI, ARI, NMI, silhouette, largest cluster %, assigned clusters, noise %, runtime, seed SD> |
| Software environment | <python + key package versions or lockfile hash> |
| Hardware | <CPU / GPU / threads> |

<For non-experimental phases: describe the audit configuration and the verification method instead.>

## 6. Actual results

<Tables with exact values from saved outputs. State the source file under every table.>

| Configuration | HS3 AMI (mean ± SD) | HS3 ARI | HS3 NMI | Silhouette | Largest cluster % | Assigned clusters | Noise % | Fit s |
|---|---|---|---|---|---|---|---|---|
| … | | | | | | | | |

Source: `<path to CSV>`

**Comparison with frozen baselines** (`reports/roadmap/phases/P0/artifacts/BASELINE_REFERENCE.csv`): <Δ against the matching baseline row>

**Stability:** <seed SD, partition agreement>

**Missing or failed experiments:** <list with reason, or "none">

## 7. What do these results mean?

<The most important section. Write for a reader who knows the project but not this phase. Explain technical terms.
Mark every sentence that is not a direct measurement as Interpretation or Hypothesis.>

- **What did we learn?**
- **Did the results support the hypothesis?**
- **Did anything unexpected happen?**
- **Why is this scientifically relevant?**
- **Does this change any earlier conclusion?**
- **What should be investigated next?**

## 8. Limitations and problems

- **Missing experiments:**
- **Data-quality issues:**
- **Unverified assumptions:**
- **Computational limits:**
- **Potential bias (museum, class imbalance, duplicates):**
- **Statistical limits (single split, no CIs, post-hoc choices):**
- **Open questions:**

## 9. Files and reproducibility

| Kind | Path |
|---|---|
| Scripts | |
| Configurations / protocol | |
| Manifests | |
| Saved scores | |
| Cluster assignments | |
| Figures | |
| Logs | |
| Validation report | |

```bash
# exact commands, run from the repository root, in order
```

Large files kept outside Git (embeddings, weights, images): <paths + sha256 recorded in the tracked manifests>

## 10. Completion checklist

Copied from the execution roadmap (`docs/roadmap/Clustering_Execution_Roadmap_W8_to_Final.docx`).

| # | Roadmap task | Status | Evidence |
|---|---|---|---|
| 1 | <task text> | <Complete / Partial / Blocked / Not started / Deferred> | <file, check, number> |

**DONE WHEN:** <criterion text> → <met / not met, and why>

## 11. Next phase handoff

- **Ready for the next phase:**
- **Files the next agent should read first:**
- **Frozen configurations (do not change without a new protocol id):**
- **Open risks and unresolved problems:**
- **May the next phase begin?** <Yes / Yes, with conditions / No — reason>

The next phase is not started automatically; it waits for review of this report.

## 12. Presentation-ready summary

**What we did:** <two or three sentences>

**Main result:** <the single most important verified finding, with its number>

**Scientific interpretation:** <what the result tells us>

**Limitations:** <what must be disclosed on the slide>

**Recommended figure:** `<exact path to an existing figure, or "none yet">`

**Slide-ready numbers:**

| Number | Value | Context (dataset, k, representation, seeds) | Source |
|---|---|---|---|
| | | | |
