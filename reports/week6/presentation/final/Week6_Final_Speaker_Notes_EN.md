# Week 6 final presentation: English speaker notes

**Deck:** Week6_Final_FullMIMO_StatusUpdate.pptx (10 main slides + 5 backup) · **Talk:** 16 Nov 2026 · **Planned time:** 9:30 (slides 1–9) plus questions.

Terminology used throughout: **full MIMO collection** = 59,658 cleaned photographs; **selected benchmark** = 27,347 eligible objects; **development benchmark** = the 12,000 objects used in every experiment shown; **held-out set** = 14,696 reserved objects, never used. All scores are development results (HS3, 13 classes, k = 10, centered + L2 unless stated).

## Slide 1 · Title

[0:40] What: our Week 6 status. Since Week 4 the project moved from a small 2,000-image pilot to the full MIMO collection. Our question stays the same: can unsupervised clustering of frozen image embeddings recover the expert Hornbostel–Sachs (HS) classification, which groups instruments by how they produce sound, while photographs mostly show form? Today we show (1) the reconstructed dataset, (2) the first controlled comparison of three backbones and five clustering methods on a 12,000-object development benchmark. Limitation up front: these are development results; the held-out set has not been touched.

## Slide 2 · 1 · What we accomplished since Week 4

[1:00] What: progress since Week 4. How: we re-downloaded the MIMO collection through the Europeana API with our project key, reran the teammate's cleaning and labelling scripts unchanged, and rebuilt the selected benchmark with the original rule. Meaning: 63,478 records gave 59,658 cleaned photographs and 27,347 benchmark objects, within 1–3 objects of the Week 4 numbers. We installed official DINOv2 weights in three sizes; all three were tested on the pilot, but only DINOv2-B was run on the 12,000-object development benchmark. Relevance: we now evaluate on the data the project was designed for. Validation: an independent script recomputed every score in this talk from the saved cluster assignments. Limitation: no held-out results yet.

## Slide 3 · 2 · Dataset and evaluation protocol

[1:10] What: the dataset funnel and protocol. How: from 63,478 Europeana records we downloaded 63,473 valid images (5 unreachable). Cleaning removed 247 placeholder images, 556 copies of shared photos and 3,012 instrument parts, leaving 59,658 photographs: the full MIMO collection. The selected benchmark keeps objects that have an HS3 code, a Western family and an instrument name, with every class at least 30 objects: 27,347 objects, 13 HS3 classes, 6 families, 129 names. We split it at object level: 12,000 development objects (the teammate's original 12,000-row rule), 14,696 held out, and 651 removed because we had already looked at them in the pilot. Meaning: all experiments today use only the development set; the held-out set is reserved for one final run of a frozen pipeline. Limitation: flutes, reedpipes and lutes dominate; the smallest classes have only 23 development objects; the audio-object exclusion from the held-out set is still pending.

## Slide 4 · 3 · Our working benchmark pipeline

[1:00] What: the pipeline. How: each photo is padded to a white square so long instruments are not cropped, then embedded by a frozen encoder: CLIP ViT-B/32, SigLIP ViT-B/16 or DINOv2-B/14 at 224 px. No model is trained. We then optionally center the embeddings (subtract the mean vector), L2-normalise them, or reduce them with PCA, and cluster them with five methods at a predeclared k = 10. Only at the end do we compare the clusters with the HS3 labels using AMI, ARI and NMI. Relevance: the labels never influence the clusters, so the scores measure how much of the expert taxonomy appears on its own. All comparisons use the same 12,000 objects, so differences come from the method, not the data. Limitation: these are not scores on all 59,658 photographs, and DINOv2 on the development set was run at 224 px only.

## Slide 5 · 4 · RQ1: Which backbone performs best?

[1:20] What: RQ1, backbone comparison. How: centered + L2 embeddings, k = 10, 12,000 development objects; k-Means is the mean of 5 seeds. Meaning: AMI is 0 for a random grouping and 1 for perfect agreement with the 13 HS3 classes. DINOv2-B gives the highest AMI with k-Means (0.538 vs 0.523 SigLIP, 0.451 CLIP) and with agglomerative clustering (0.561 vs 0.542 vs 0.512). Relevance: DINOv2 was trained without text, on images only, and still recovers the expert classes best with these two methods. Limitation: this does not hold for every method: with Spectral clustering SigLIP is clearly better (0.559 vs 0.476), and for k-Means, ARI favours SigLIP. One development split, no significance test, no confidence interval.

## Slide 6 · 5 · RQ1: Does the clustering method matter?

[1:20] What: RQ1, does the method matter? How: each point is one backbone and one method on the same development objects; x = AMI, y = ARI. Meaning: DINOv2-B + agglomerative (AMI 0.561, ARI 0.417) and SigLIP + Spectral (AMI 0.559, ARI 0.311) are tied on AMI but differ strongly on ARI. ARI counts pairs of images that are grouped together, so one large mixed cluster costs a lot; AMI measures shared information and is more tolerant. DINOv2-B + Spectral is the warning case: 50.1% of all objects end up in one cluster, ARI 0.165. Relevance: choosing a method by one metric can hide a degenerate partition, so we report AMI, ARI and cluster sizes together. DINOv2-B + agglomerative is the most balanced candidate (largest cluster 21.6%), but only under this configuration (k = 10, centered + L2); we do not call it universally best. Limitation: development data; the choice was made after seeing these results.

## Slide 7 · 6 · RQ2: Does preprocessing matter?

[1:10] What: RQ2, preprocessing. How: same 12,000 objects, agglomerative clustering with average linkage and cosine distance, k = 10; only the representation changes. Meaning: on raw L2-normalised embeddings, CLIP and SigLIP collapse: 99.8% of all objects land in one cluster and AMI is 0.002 / 0.001. The reason is that all raw embeddings point in a similar direction, so cosine distances are compressed. Subtracting the global mean fixes this: 0.512 and 0.542. DINOv2-B is more robust raw (0.361, 68.6% in one cluster) and reaches 0.568 with PCA100 + L2. Relevance: a simple, label-free preprocessing step can matter more than the choice of algorithm. Limitation: the best preprocessing depends on backbone and method; per-museum centering is not tested yet.

## Slide 8 · 7 · Current problems and research limitations

[1:00] What: open problems. (1) Only the development benchmark is evaluated; the held-out set is untouched by design. (2) The collection has 59,658 photographs, but clustering so far uses 12,000 objects, so we must not claim results for the whole collection. (3) Scaling: DINOv2-B was evaluated at scale; S/B/L sizes and layer depth were only explored on the 2,000-image pilot (backup slide). (4) Museum confounding: the five largest museums hold 53% of benchmark objects, so clusters could partly reflect photo style. (5) Exact duplicates are removed, but near-duplicates are not screened yet. (6) The original protocol removes objects with audio recordings from the held-out set; that list is not prepared yet. Relevance: all scores are development estimates and probably optimistic.

## Slide 9 · 8 · Next steps and research roadmap

[0:50] What: the roadmap, unchanged from Week 4. Week 8: extend DINOv2 on the development benchmark, test model size and intermediate layers at scale, more preprocessing, museum effects and near-duplicates, and label-free choice of k. Week 10: finish development experiments, freeze the pipeline, finish the held-out split. Week 14: multimodal experiments and one single held-out run with the frozen pipeline. Weeks 16 and 17: final presentation with demo, report and code. Priority: validate the development findings at scale before we freeze anything.

## Slide 10 · Closing

Thank you. Backup slides: all five methods with AMI and ARI; the label-informed k = 13 reference; museum distribution; reconstruction vs plan counts; exploratory pilot RQ3 scaling.

## Slide 11 · Backup A · All five methods (development benchmark)

All five methods on the development benchmark, AMI left, ARI right. Same configuration as slides 5 and 6.

## Slide 12 · Backup B · Label-informed k = 13 reference

k = 13 equals the number of HS3 classes in the benchmark, so it uses label information and is only a reference. DINOv2-B + agglomerative reaches 0.584 at k = 13 versus 0.561 at k = 10. Our headline stays the predeclared k = 10.

## Slide 13 · Backup C · Museum distribution

Museum distribution: 216 museums in the cleaned collection, 195 in the benchmark; the five largest hold 53% of benchmark objects. Tervuren contributes many photos but few benchmark objects because most of its records have no HS code.

## Slide 14 · Backup D · Reconstruction vs plan counts

Our recomputed counts against the research plan: within 1–3 objects at every stage. Differences come from 5 unreachable images and live Wikidata edits since October.

## Slide 15 · Backup E · Exploratory pilot RQ3 scaling

EXPLORATORY, 2,000-image pilot only (22 HS3 classes, different from the development benchmark). On the pilot, DINOv2 S < B < L at the last layer, and earlier layers are much worse for S and B. Do not compare these numbers with the development scores; full-scale RQ3 validation is future work.

## Likely questions and short answers

**Q: Are these results on all 60,000 photographs?**
No. The full MIMO collection has 59,658 cleaned photographs. The benchmark keeps the 27,347 objects that have an HS3 code, a family and a name in classes of at least 30. All scores today are on the 12,000-object development part of that benchmark.

**Q: Is DINOv2 the best backbone?**
For k-Means and agglomerative clustering, yes, on the development benchmark (AMI 0.538 and 0.561). Not for Spectral clustering, where SigLIP is best (0.559 vs 0.476). One split, no significance test.

**Q: Why do you trust 0.561 when SigLIP + Spectral gets 0.559?**
We don't claim it is better on AMI; the two are tied. DINOv2-B + agglomerative has clearly higher ARI (0.417 vs 0.311) and no oversized cluster (21.6% vs 34.7%), so it is the more balanced candidate under this configuration.

**Q: Why k = 10 when there are 13 HS3 classes?**
k = 10 was declared before the experiments, so it doesn't use label information. k = 13 is shown in the backup only as a label-informed reference (DINOv2-B + agglomerative 0.584). Choosing k without labels is still open.

**Q: Which DINOv2 resolution did you use?**
224 px on the development benchmark. The 518 px runs (the model's default) were only done on the 2,000-image pilot and are not mixed with today's numbers.

**Q: Did you compare DINOv2 S, B and L at scale?**
No. All three sizes and layers −1, −2, −4 were compared only on the pilot (backup, exploratory). Only DINOv2-B was evaluated on the 12,000 development objects. That is a Week 8 task.

**Q: Could the clusters just reflect museums?**
Possibly. The five largest museums hold 53% of benchmark objects, and photo style differs by museum. We have not tested this yet; museum-level analysis is planned for Week 8.

**Q: How do you know the numbers are right?**
A separate verification script recomputed every score in the deck from the saved cluster assignments and the label files (224 of 224 checks pass), checked the dataset counts and confirmed that no held-out object was embedded or scored.

**Q: What is the 0.53 / 0.83 / 0.68 from the plan?**
The research plan's DINOv2 baseline, linear-probe reference and target come from the earlier analysis. Our development benchmark reproduces its dataset rule. However, the held-out comparison and the probe have not been rerun at scale, so we don't compare against those numbers today.
