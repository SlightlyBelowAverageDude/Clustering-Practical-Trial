# Week 6 speaker notes: Visual Form vs. Acoustic Function in Instrument Clustering

Talk: 16 November 2026, about 10 minutes (slides 1–9) plus questions. Slide 10 closes the talk; slides 11–14 are backup.
Timing plan: 0:45 · 1:00 · 1:00 · 1:30 · 1:15 · 1:15 · 1:15 · 1:00 · 1:00 = 10:00.
Every number below comes from `reports/week6/p4_validation/presentation_claims.json` (independently recomputed; 299/299 checks pass). Sources per slide: `Week6_Slide_Sources.md`.

## Slide 1: Topic 1 · Week 6 Update · 16.11.2026

(~45 s) Week 4 we presented the plan; today we show a verified working benchmark. Our question: can unsupervised clustering of frozen image embeddings recover the expert Hornbostel–Sachs (HS) taxonomy, which classifies instruments by how they produce sound, while images mainly show form. Today's results are all on a 2,000-image development pilot drawn from MIMO/Europeana. They are development results, not held-out estimates: I will say this several times because it matters for how to read every number.

## Slide 2: 1 · What we accomplished since Week 4

(~1 min) Since Week 4 we first audited what already existed. Every earlier score, 816 rows, was recomputed exactly from the saved cluster assignments, and the dataset counts were confirmed: 2,000 images from 216 museums, 1,549 with an HS level-3 label, 22 classes. On top of that we ran P2 (five clustering families, two backbones, blind k) and P3 (subset robustness and a preprocessing ablation): 1,556 fits in total. Labels were never used to fit clusters or to choose k; they enter only in a separate scoring step, after the selection files are written. Finally, a separate script re-derived every number in this talk from the raw assignment files: 299 of 299 checks pass. What is not done: DINOv2 and the large 27k benchmark.

## Slide 3: 2 · Our working benchmark pipeline

(~1 min) The pipeline is the one from Week 4, now verified end to end. Images are padded to a white square so long instruments are not cropped. Two frozen encoders, CLIP and SigLIP. Five representations, from raw L2 to centering and PCA. Five clustering families plus four blind k criteria. Evaluation with AMI, NMI and ARI against HS levels 1 to 3: labels enter only here. The protocol was written to a file before any fit. Important change from the plan: the large 27,348-object benchmark with 13 HS3 classes and its held-out split is not available locally, so all results today are on the 2,000-image pilot with 22 HS3 classes. That is why we do NOT compare our numbers with the plan's 0.53 baseline, 0.83 probe or 0.68 target: different data, different classes, different encoder. DINOv2 code is ready, but the weights were not downloaded yet.

## Slide 4: 3 · RQ1: Which clustering method works best?

(~1.5 min) Left: HS3 AMI against the number of clusters, CLIP left panel, SigLIP right. AMI is 0 for random and 1 for perfect agreement with the expert classes. At our predeclared k = 10, SigLIP + Spectral clustering on a 15-nearest-neighbour graph reaches 0.544 (mean of 5 seeds), against 0.476 for k-Means. But look at ARI in the table: Agglomerative has 0.360, Spectral only 0.309. Agglomerative wins on ARI in all 10 subset runs. So the ranking depends on the metric; the next slide explains why. Backbone: SigLIP beats CLIP in 51 of 56 matched comparisons, the clearest result we have. HDBSCAN is not competitive: it leaves 59–70% of images as noise. Caveat: the 0.544 is from the full pilot; on five 80% subsets Spectral averages 0.520, and we picked Spectral after seeing these results.
Likely question: 'Is Spectral significantly better?' No significance test: the comparisons share images. Spectral beats k-Means in 9/10 subset runs, but beats Agglomerative only for SigLIP (5/5), not for CLIP (2/5).

## Slide 5: 4 · Why AMI and ARI disagree: form over function

(~75 s) This is the interpretation slide. Top: images from the two most mixed Spectral clusters, sampled at random within their main classes. Row 1 is a catch-all cluster of small, hand-held objects: vessel flutes, panpipes and whistles (HS 421, edge-blown), castanets and bells (111), rattles (112). That one cluster has 774 images, 0.23 purity, and contains 61% of all pairs of images that share a cluster. ARI counts pairs, so it grows with the square of cluster size: this one cluster pulls ARI down. AMI is information-based: Spectral's other clusters are very pure (brass, lutes, harps), which gives the most mutual information, so AMI goes up. Row 2 is the most interesting for our project: long tubes. Clarinets, oboes and bagpipes (reedpipes) sit with flutes and fifes, and even a 'Russian bassoon' and a serpent, which are lip-reed (brass) instruments. The images group by visual form; Hornbostel–Sachs groups by how the sound is made. Caveat: one partition, descriptive; a couple of catalogue codes look odd (two 'Tambour' records coded as struck idiophones), so some label noise is likely.

## Slide 6: 5 · Can we choose k without labels?

(~75 s) Can we choose k blindly? On the full pilot, maximum silhouette chose k = 8 or 10 for k-Means, which is close to the best possible. For Spectral it chose k = 5 for both backbones and lost 0.07 to 0.09 AMI against k = 10: silhouette favours coarse partitions. In P2 the spectral eigengap looked promising, but on four configurations only. So in P3 we repeated it on five 80% subsets (the plot). Each row is one subset: blue dot = silhouette choice, orange square = eigengap choice, black bar = fixed k = 10, open circle = the best k if we had used labels, shown only as a reference. Eigengap jumps between k = 5 and 20, and beats silhouette in only 5 of 10 cases. Honest conclusion: no blind k rule is reliable yet. Note that HS3 has 22 classes, but only 13 with at least 10 images, so even the 'true' k is not well defined on this pilot.
Likely question: 'Why not just use 22?' That uses label information. Our protocol keeps k fixed at a predeclared 10 or chooses it blindly.

## Slide 7: 6 · RQ2: Does preprocessing matter?

(~75 s) RQ2: same rows, same masks, same k = 10, only the representation changes. Top row AMI, bottom row ARI. Strongest result: on raw L2-normalised embeddings, average-link agglomerative collapses, with about 99% of images in one cluster and AMI 0.03. Simply subtracting the global mean fixes it: 0.46 for CLIP and 0.50 for SigLIP. The reason is that raw embeddings all point in a similar direction (mean-vector norm about 0.7), so cosine distances are compressed. But centering is not a universal improvement: k-Means gets slightly worse, and Spectral gains only at k = 10. PCA adds nothing beyond the centering it includes, and PCA50 without re-normalisation hurts Spectral. So preprocessing interacts with the algorithm: it changes which method looks best. Global centering is not the same as per-museum centering; we did not run the latter because museums contribute only 1 to 14 images each in the pilot.

## Slide 8: 7 · Current problems and research limitations

(~1 min) Six open problems. One: the pilot was explored in earlier phases, so every number is a development estimate and probably optimistic; Spectral's 0.544 is near the top of its subset range. Two: the large benchmark and its held-out split are not available locally, which is why we cannot test the 0.68 target today. Three: DINOv2 is not evaluated; we only need authorization to download about 1.6 GB of weights. Four: class imbalance (aerophones are half of HS1) and dominant clusters make AMI and ARI disagree. Five: museums contribute only 1 to 14 images each, and we flagged 191 possible near-duplicate pairs that need human confirmation; overlapping subsets cannot give confidence intervals. Six: for the multimodal RQ4, only 868 objects have their own recording.

## Slide 9: 8 · Next steps and research roadmap

(~1 min) The schedule is unchanged from Week 4. For Week 8 we extend RQ1, finish the RQ2 ablations, start RQ3 model scaling and extend the stability analyses. What changes in priority is the order: first, the DINOv2 weights, which is all RQ3 needs, and the linear-probe reference. Second, the large benchmark and its held-out split, so the freeze in Week 10 is not based on the explored pilot. Third, we will always report AMI and ARI together, given today's metric disagreement. Week 10: freeze the pipeline; Week 14: multimodal and the one-shot held-out run; Week 16 the final presentation with demo, Week 17 the report and code.

## Slide 10: Thank you for listening

Backup slides follow: AMI vs ARI heatmaps, subset robustness, cluster-size diagnostics, backbone comparison.

## Slide 11: Backup · Why AMI and ARI disagree (cluster × class composition)

Each row is a cluster (sorted by labelled size), each column one of the 12 most frequent HS3 classes; the cell is the class share within the cluster. Spectral: one large mixed cluster (C1) plus many pure clusters. Agglomerative: more medium mixed clusters, smaller catch-all.

## Slide 12: Backup · Spectral robustness on five 80% subsets

Five 1,600-image subsets (phase-2 seeds 2026–2030), centering refit inside each subset. Spectral > k-Means AMI in 4/5 CLIP and 5/5 SigLIP subsets; Spectral > Agglomerative AMI in 2/5 CLIP and 5/5 SigLIP; Agglomerative > Spectral ARI in 10/10.

## Slide 13: Backup · Cluster-size diagnostics

Largest cluster share at k = 10 per representation (worst seed) and the cluster-size profile for SigLIP centered + L2. Raw-L2 Agglomerative collapses; Spectral keeps one dominant cluster under every representation.

## Slide 14: Backup · Backbone comparison at k = 10

CLIP vs SigLIP on identical rows and masks. This compares complete encoder bundles (architecture, patch size, training data, loss, transform), so the gain cannot be attributed to the pretraining loss alone.

## Likely questions from the professor

**Q: Is Spectral significantly better than k-Means?**  
No significance test was run, because all comparisons share the same images and the subsets overlap. At k=10, Spectral beats k-Means in 9/10 subset runs (43/50 paired seeds; all 25 for SigLIP). Against agglomerative, it wins only for SigLIP (5/5 subsets), not for CLIP (2/5). Spectral was also chosen after seeing development results.

**Q: Your 0.544 is close to the 0.53 baseline from the plan. Have you matched DINOv2?**  
No, the numbers are not comparable. The plan's 0.53/0.83 come from DINOv2 on a 27,348-object set with 13 HS3 classes and k = number of classes. Our pilot has 22 HS3 classes (many tiny), uses CLIP/SigLIP, and k=10. The plan's own table even gives 0.55 at HS3 (0.53 is HS2), so we flagged that baseline as ambiguous.

**Q: Why not set k = 22?**  
That uses label information. We use a predeclared k=10 or a blind criterion. Only 13 of the 22 classes have ≥10 images, so the "true" k is not well defined on the pilot anyway.

**Q: Which metric should we trust, AMI or ARI?**  
Both, reported together. ARI counts pairs, so it penalises one large mixed cluster quadratically. AMI rewards pure clusters and is less sensitive to a single catch-all cluster. Spectral is better on AMI; agglomerative is better on ARI in 10/10 subsets.

**Q: Why does raw-L2 agglomerative collapse?**  
Raw embeddings share a strong common direction (mean-vector norm ≈ 0.70 for CLIP and 0.77 for SigLIP), which compresses cosine distances. Average linkage then merges almost everything. Subtracting the global mean removes that direction.

**Q: Is the "form over function" interpretation proven?**  
No. The mixed clusters (flutes with reedpipes; small hand-held objects) are descriptive observations from one partition. Testing it needs the paired audio (RQ4) and expert review of the catalogue labels, some of which look noisy.

**Q: Why was DINOv2 not run?**  
No local weights. We did not download about 1.6 GB of models without team authorization. The code path, dimensions and CPU timing are checked (DINOv2-B ≈ 6 min for the pilot), so this is the first Week 8 task.

**Q: Is HDBSCAN useless?**  
On these embeddings it leaves 59–70% of images as noise. Its high score on the assigned images is matched or beaten by k-Means scored on the same images (8/8), so its apparent advantage comes from which images it keeps. Running it on a UMAP projection is a reasonable next test.

**Q: Could museum style drive the clusters?**  
Possibly. The pilot has only 1–14 images per museum, so we could not test museum effects reliably. Museum-level resampling and per-museum analyses are planned on the larger set.

