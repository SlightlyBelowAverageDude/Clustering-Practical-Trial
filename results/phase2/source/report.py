"""Generate progress prose and slide content exclusively from finished experiments."""
import json
import numpy as np
import pandas as pd
from benchmark import OUT,ROOT,csv,write_json,now

def table(headers,rows):
    def cell(v):return str(v).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(cell(x) for x in row)+' |' for row in rows]])

def main():
    for enc in ['clip','siglip']:
        for rel in [f'{enc}/validation.json',f'evaluation/{enc}/validation.json']:
            assert json.loads((OUT/rel).read_text())['success'],rel
    summaries={e:pd.read_csv(OUT/e/'summary.csv') for e in ['clip','siglip']}
    pair=pd.read_csv(OUT/'paired_encoder_comparison.csv');grid=pd.read_csv(OUT/'paired_grid_summary.csv')
    model=json.loads((OUT/'siglip/embedding_manifest.json').read_text())
    quality=json.loads((OUT/'review/data_quality_summary.json').read_text());review=json.loads((OUT/'review/summary.json').read_text())
    def anchor(enc,method):
        d=summaries[enc];return d[d.label.eq('hs3')&d.representation.eq('centered_l2')&d.method.eq(method)&d.requested_k.eq(10)].iloc[0]
    anchors=[]
    for enc in ['clip','siglip']:
        for method in ['kmeans','average_cosine']:
            r=anchor(enc,method);ami=f'{r.AMI_mean:.4f}'+(f' ± {r.AMI_seed_sd:.4f}' if method=='kmeans' else '')
            anchors.append([enc.upper(),method,ami,f'{r.ARI_mean:.4f}',f'{r.NMI_mean:.4f}',f'{r.largest_cluster_pct:.2f}',f'{r.silhouette_cosine:.4f}'])
    uncertainties=[];subset_means={};stability=[]
    for enc in ['clip','siglip']:
        u=pd.read_csv(OUT/'evaluation'/enc/'uncertainty_summary.csv');seed=pd.read_csv(OUT/'evaluation'/enc/'seed_partition_stability.csv');overlap=pd.read_csv(OUT/'evaluation'/enc/'subset_overlap_stability.csv')
        subs=pd.read_csv(OUT/'evaluation'/enc/'subset_scores.csv');s=subs[subs.label.eq('hs3')].groupby(['method','subset_seed']).AMI.mean().reset_index();subset_means[enc]=s
        for r in u[u.label.eq('hs3')].itertuples():
            within='not applicable' if r.method=='average_cosine' else f'{r.mean_within_subset_seed_sd:.4f}'
            uncertainties.append([enc.upper(),r.method,within,f'{r.sd_of_subset_seed_means:.4f}'])
        g=seed[seed.representation.eq('centered_l2')&seed.k.eq(10)]
        stability.append({'encoder':enc,'anchor_kmeans_seed_partition_ARI':g.partition_ARI.mean(),**{f'{m}_subset_overlap_ARI':overlap.loc[overlap.method.eq(m),'partition_ARI'].mean() for m in ['kmeans','average_cosine']}})
    csv(OUT/'stability_summary.csv',stability)
    paired_sub=subset_means['clip'].merge(subset_means['siglip'],on=['method','subset_seed'],suffixes=('_clip','_siglip'));paired_sub['AMI_delta_siglip_minus_clip']=paired_sub.AMI_siglip-paired_sub.AMI_clip;csv(OUT/'paired_subset_deltas.csv',paired_sub)
    pos=int(pair.AMI_delta_siglip_minus_clip.gt(0).sum());non=grid[grid.remove_cosine_scale_duplicate.eq(True)]
    non_cells=int(non.cells.sum());non_pos=int(non.positive_cells.sum())
    delta_k=anchor('siglip','kmeans').AMI_mean-anchor('clip','kmeans').AMI_mean
    delta_a=anchor('siglip','average_cosine').AMI_mean-anchor('clip','average_cosine').AMI_mean
    subset_sentences=[]
    for method,g in paired_sub.groupby('method'):
        subset_sentences.append(f'{method}: mean paired delta {g.AMI_delta_siglip_minus_clip.mean():+.4f}, positive in {g.AMI_delta_siglip_minus_clip.gt(0).sum()}/5 subsets')
    lines=['# Next update: Clustering meets Vision and Language','', 'Progress for 2 November 2026. Completed experiments only. Original pilot artifacts remain unchanged.','',
           '## Question and design','',
           'Do frozen vision-language embeddings group instrument photographs according to the expert Hornbostel–Sachs taxonomy, and which preprocessing choices change that answer? Same 2,000 images / 216 museums; HS3 evaluation uses the same 1,549 labeled rows. All images, including missing-label records, enter clustering.','',
           'Five representations: original L2; centered full-dimensional + L2; PCA50; PCA50 + L2; PCA100 + L2. K-Means uses five seeds, ten restarts; average linkage uses cosine distance. Fixed k=5/10/20/40. Two encoders give 240 full-grid fits / 720 score rows. Five paired 80% image subsets at a fixed centered k=10 anchor add 60 fits; 1,600 size-preserving random partitions provide chance controls. In total, all 5,700 AMI/ARI/NMI rows and 480 full-sample silhouette values were independently recomputed from saved predictions. The pilot was already explored in phase 1; declaring this phase’s comparisons does not convert it into a held-out test.','',
           '## Main result: centering explains the collapse rescue','',
           'For CLIP average linkage at k=5, raw L2 has HS3 AMI **0.0035** and a **99.65%** largest cluster. Centering in all 512 dimensions gives **0.3793** and **37.10%**, without dimensionality reduction. PCA50+L2 gives 0.3942 and 32.40%. At k=10, centered full-dimensional AMI 0.4591 is close to PCA50 0.4610. PCA50 helps at some other cuts, but PCA100 does not improve consistently: there is no general dimensionality-reduction benefit established here.','',
           'For cosine average linkage, PCA50 partitions are exactly identical with and without post-L2 at every k, for both encoders. Positive row scaling cannot affect cosine distance. K-Means changes with normalization, but CLIP preprocessing gains are small and inconsistent across k. [Full Task 1 results](TASK1_RESULTS.md).','',
           '![Preprocessing ablation](figures/preprocessing_ablation.png)','',
           '## CLIP versus SigLIP: matched configurations','',
           f'SigLIP: `{model["model"]["repo"]}`, revision `{model["model"]["revision"]}`, 768 dimensions; CLIP: LAION ViT-B/32, 512 dimensions. Both frozen, CPU float32, four threads. Both receive the original EXIF/RGB white-square-padding policy, then each checkpoint’s published evaluation transform. SigLIP uses 224-pixel bicubic squash and mean/std 0.5. This compares complete encoder/preprocessing bundles, not the pretraining loss alone. [Model manifest](siglip/embedding_manifest.json).','',
           'The centered full-dimensional k=10 anchor was declared before the new phase 2 fits, independently of their HS scores:','',
           table(['Encoder','Method','HS3 AMI (mean ± seed SD)','ARI','NMI','Largest cluster %','Cosine silhouette'],anchors),'',
           f'SigLIP’s paired HS3 AMI is higher in **{pos}/40** declared cells ({non_pos}/{non_cells} after removing cosine’s duplicate normalization condition). Mean grid deltas: K-Means {grid[(grid.method=="kmeans")&(~grid.remove_cosine_scale_duplicate)].mean_paired_AMI_delta.iloc[0]:+.4f}; average cosine {grid[(grid.method=="average_cosine")&(~grid.remove_cosine_scale_duplicate)].mean_paired_AMI_delta.iloc[0]:+.4f}. These equal-weight grid summaries reuse the same images; cells are not independent replicates. At the fixed anchor, deltas are {delta_k:+.4f} and {delta_a:+.4f}. Average-linkage anchor ARI is slightly lower for SigLIP (0.3598 vs 0.3662), so improvement is not unanimous across metrics. No best method/k was selected and reported as an unbiased estimate.','',
           '![Matched encoder comparison](figures/clip_siglip_comparison.png)','',
           '## Chance and stability','',
           'Twenty prediction permutations per seed-0 configuration preserve full cluster sizes. Their AMI is near zero; observed anchor partitions exceed this chance control. Raw NMI has a positive chance baseline; AMI/ARI are adjusted. [Chance results](evaluation/) and [figure](figures/chance_baseline.png).','',
           table(['Encoder','Method','Mean within-subset initialization SD','SD of subset seed-means'],uncertainties),'',
           'Each of five subsets contains 1,600 images sampled without labels, with centering refit within that subset; identical subsets are used for both encoders. ' + '; '.join(subset_sentences) + '. Seed SD measures initialization. Variation across subset means measures conditional composition sensitivity: overlapping subsets of one museum-balanced pilot do not supply an independent-dataset confidence interval. Near duplicates may inflate stability. At the fixed anchor, SigLIP has higher K-Means AMI seed SD despite stronger average partition agreement; score variability and partition stability are different quantities. [Saved subset deltas](paired_subset_deltas.csv).','',
           '## Data quality and limitations','',
           f'- HS3 is missing for 451/2,000 images (22.55%). Aerophones are {quality["hs1_majority_fraction_among_labeled"]:.1%} of HS1-labeled images. Seven museums have no HS3 labels; 71 have complete coverage in this small sample. Museum contributions are only 1–14 images. Evaluation may favor better-catalogued instruments and cannot establish cross-museum generalization.',
           f'- The 180-image [review gallery](galleries/human_review_180.html) spans {review["review_museums"]} museums and all five HS1 classes, with missing/conflicting-label and outlier strata. A [blank annotation template](review/annotation_template.csv) and [codebook](review/ANNOTATION_CODEBOOK.md) are ready. No human annotations, source corrections or exclusions were applied.',
           f'- Lexical screening flags 65 possible parts records, with {quality["hs3_missing_fraction_parts_flagged"]:.1%} missing HS3 versus {quality["hs3_missing_fraction_not_flagged"]:.1%} in the rest. These are screening flags, not prevalence estimates: “embouchure” can describe an entire flute. The enriched review sample cannot estimate a collection-wide error rate.',
           f'- Visual inspection found a generic placeholder at row 237 (HS3 421), piano branding/detail at row 1834 (HS3 314), and a painting at row 1955 (no HS). {review["candidate_pair_count"]} near-duplicate/view candidate pairs were flagged; 15 disjoint pairs are shown for confirmation. Exact pixel deduplication alone does not ensure independent objects. [Observations](review/visual_observations.csv).',
           '- Both encoders had image-text supervision before this experiment. Clustering is unsupervised, not necessarily representation pretraining. Internal silhouettes describe geometry, not taxonomy correctness. No held-out population estimate, expert-validated clean benchmark or new-dataset confidence interval is claimed. Wall times are recorded but include concurrent local audit/review activity, so they are not an isolated speed benchmark.','',
           '[Meaningful grouping gallery](galleries/meaningful_grouping.html) · [Failure gallery](galleries/failure_case.html) · [Cluster-size imbalance](figures/cluster_size_imbalance.png)','',
           '## Next steps','',
           '1. Complete human review, confirm object/view groups, and release an explicit curated version. Re-run the fixed grid on it as a declared sensitivity analysis; retain this uncurated result.',
           '2. Acquire a focused cross-museum sample with at least 100 coded images per selected museum. Group near duplicates/specimens before splitting. Fit PCA and K-Means on training museums only, freeze the configuration using separate development data, then score unseen museums. Declare an out-of-sample rule separately for agglomeration.',
           '3. Use museum/specimen-level resampling and external evaluation for population uncertainty. Investigate hierarchical/multilabel HS scoring and crop/padding sensitivity after data-quality controls.','',
           '[Protocol and reproduction](PROTOCOL.md) · [Per-run CLIP scores](clip/scores.csv) · [Per-run SigLIP scores](siglip/scores.csv) · [Paired grid](paired_encoder_comparison.csv). No commit or push was performed.','',
           'Model source: [SigLIP checkpoint and published configuration](https://huggingface.co/timm/ViT-B-16-SigLIP); converted from the original Google Big Vision checkpoint. All numerical findings above are from this local run.','']
    (OUT/'NEXT_UPDATE.md').write_text('\n'.join(lines))
    slides=['# Clustering meets Vision and Language','## Topic 1 · Progress, 2 November 2026','',
            '**Question:** Do clusters of museum-instrument photos recover Hornbostel–Sachs categories?','',
            '**Finding:** The geometry’s origin matters more than the first PCA comparison suggested.','',
            '---','', '## 1. Controlled comparison','',
            '- Fixed 2,000 MIMO images, 216 museums; HS3 labels for 1,549.',
            '- Frozen CLIP B/32 and SigLIP B/16; identical image order and evaluation masks.',
            '- Five representations; K-Means and cosine average linkage; k=5,10,20,40.',
            '- Five K-Means seeds. 240 full-grid fits; all metrics independently checked.',
            '- Labels are used for scoring, never to fit embeddings, PCA or clusters.','',
            'Speaker note: The pilot is museum-balanced and uncurated. These are descriptive results on a fixed dataset, not unseen-museum generalization.','',
            '---','', '## 2. What actually fixes the CLIP failure?','',
            table(['Average linkage, k=5','HS3 AMI','Largest cluster'],[['Raw L2','0.0035','99.65%'],['Centered full + L2','0.3793','37.10%'],['PCA50 + L2','0.3942','32.40%']]),'',
            '![Ablation](figures/preprocessing_ablation.png)','',
            '- Centering rescues the clustering without reducing dimension.',
            '- PCA50 with/without L2 gives identical cosine partitions; no normalization benefit.',
            '- PCA dimensionality has no consistent monotonic benefit.','',
            '---','', '## 3. A second encoder, compared fairly','',
            '![CLIP versus SigLIP](figures/clip_siglip_comparison.png)','',
            f'- Fixed centered-full + L2, k=10: K-Means AMI {anchor("clip","kmeans").AMI_mean:.3f} → {anchor("siglip","kmeans").AMI_mean:.3f}; average linkage {anchor("clip","average_cosine").AMI_mean:.3f} → {anchor("siglip","average_cosine").AMI_mean:.3f}.',
            f'- SigLIP improves HS3 AMI in {pos}/40 matched cells; no claim from a single maximum.',
            '- Average-linkage anchor ARI is slightly lower for SigLIP. Encoder/preprocessing bundles differ; this does not isolate the sigmoid loss.','',
            '---','', '## 4. Chance and variability','',
            '![Chance baseline](figures/chance_baseline.png)','',
            '- Size-preserving random partitions have AMI near zero.',
            '- Initialization and image composition are different uncertainty sources.',
            '- CLIP K-Means: within-subset seed SD 0.0094; SD across subset means 0.0091.',
            '- Five overlapping 80% subsets are a sensitivity check, not a future-museum confidence interval.',
            '- AMI stability can differ from partition agreement.','',
            '---','', '## 5. The data are a research problem too','',
            '- 22.55% lack HS3; class and museum-label coverage are uneven.',
            '- Placeholder with an HS label; piano-brand detail; painting; near-duplicate/view candidates.',
            f'- Review set: 180 images / {review["review_museums"]} museums; blank annotation CSV; no labels silently repaired.',
            '- The raw-average-link failure has one 1,993-image group and small outlier clusters.','',
            '[Review gallery](galleries/human_review_180.html) · [Meaningful grouping](galleries/meaningful_grouping.html) · [Failure case](galleries/failure_case.html)','',
            '![Imbalance](figures/cluster_size_imbalance.png)','',
            '---','', '## 6. Next experiments and take-home','',
            '1. Expert review and versioned removal of placeholders/parts; specimen-aware duplicate grouping.',
            '2. Focused sample with ≥100 coded images per museum; separate development and held-out museums.',
            '3. Freeze preprocessing/k before held-out scoring; fit PCA on training museums only.',
            '4. Museum/specimen-level uncertainty and hierarchical HS evaluation.','',
            '**Take-home:** Centering fixes a major geometric failure. SigLIP improves many matched AMI comparisons on this pilot. Data quality and external validation remain the barriers to a broader claim.','',
            '---','', '## Appendix: illustrative shape grouping','',
            '![Meaningful grouping](../../data/phase2/gallery/meaningful_grouping.jpg)','',
            'Fixed CLIP raw-L2 K-Means k=5, seed 0; nearest-centroid examples, chosen without label scores.','',
            '---','', '## Appendix: a failure that AMI alone does not explain','',
            '![Failure case](../../data/phase2/gallery/failure_case.jpg)','',
            'Fixed CLIP raw-L2 cosine average linkage k=5. Random examples from the giant cluster plus all small-cluster members.','',
            '---','', '## Reproducibility and references','',
            '- Immutable inputs, model revision/weight hashes, saved predictions and masks.',
            '- All 5,700 metric rows and 480 silhouettes independently recomputed.',
            '- Original pilot unchanged; new outputs confined to phase2 and ignored model/photo caches.',
            '- [Detailed results](NEXT_UPDATE.md) · [Protocol](PROTOCOL.md)',
            '- [SigLIP checkpoint](https://huggingface.co/timm/ViT-B-16-SigLIP) · [CLIP checkpoint](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)','']
    (OUT/'SLIDES_NOV2.md').write_text('\n'.join(slides))
    print('NEXT_UPDATE.md and SLIDES_NOV2.md generated from completed, verified runs',flush=True)

if __name__=='__main__':main()
