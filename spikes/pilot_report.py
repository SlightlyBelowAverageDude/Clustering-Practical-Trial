"""Build FIRST_RESULTS.md from saved, audited observations only."""
import json

import pandas as pd
import numpy as np

from pilot_acquire import DATA, ROOT, now


def table(headers, rows):
    def cell(x):
        return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |', *['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows]])


def main():
    out = ROOT / 'results/pilot'
    report = json.loads((out / 'dataset_report.json').read_text())
    harvest = json.loads((DATA / 'harvest.json').read_text())
    validation = json.loads((out / 'validation.json').read_text())
    completion = json.loads((out / 'completion.json').read_text())
    evidence = json.loads((out / 'embedding_manifest.json').read_text())
    scores = pd.read_csv(out / 'scores.csv')
    sample = pd.read_csv(DATA / 'sample_pilot.csv', dtype=str, keep_default_na=False)
    metadata = pd.read_csv(DATA / 'metadata.csv', dtype=str, keep_default_na=False)
    ledger = pd.read_csv(DATA / 'download_ledger_pilot.csv', dtype=str, keep_default_na=False)
    env = json.loads((out / 'environment.json').read_text())
    assert validation['success'] and completion['success']
    lines = ['# First results: Clustering meets Vision and Language', '', f'LMU Topic 1 · executed 29 October 2026 · report generated {now()}.', '',
             f'The real Europeana/MIMO pipeline completed: **{len(sample):,} unique validated pilot images**, frozen OpenCLIP embeddings, and **{completion["clustering_runs"]} clustering runs**. All {len(scores)} metric rows were independently recomputed from saved predictions. No other dataset was substituted. Historical numbers in `EXPLAINER.md` were not reused as observations.', '',
             '## Acquisition and dataset', '', f'Europeana verification returned {harvest["total_results"]:,} MIMO image records. Harvested {harvest["records"]:,} unique metadata records across {harvest["pages"]} cursor pages; complete snapshot: {harvest["complete"]}. Metadata includes stable record IDs, all HS concept URIs, resolved codes, source titles/instrument names, museums, provider, image URLs, origin/place strings, provider country, dates, creators and rights where supplied.', '',
             f'Pilot: {len(sample):,} images from **{sample.museum.nunique()} museums**, {sample.hs1[sample.hs1.ne("")].nunique()} represented HS1 classes, {sample.hs2[sample.hs2.ne("")].nunique()} HS2 classes and {sample.hs3[sample.hs3.ne("")].nunique()} HS3 classes. Metadata snapshot includes {metadata.museum.nunique()} museums. Sampling uses seed 42, a deterministic museum round-robin, and stable hashed ordering within museums. Missing labels are retained; every clustering sees the same 2,000 rows.', '',
             table(['Outcome', 'Actual count'], [['Selected unique valid images', len(sample)], ['Candidate records processed', len(ledger)], *[[k, v] for k, v in report['statuses'].items()], ['Distinct image files stored, including smoke and unused candidates', len(list((DATA / 'images').glob('*.img')))], ['Exact duplicate pixels in selected pilot', len(sample) - sample.pixel_sha256.nunique()]]), '',
             f'Individual URL failures before fallback: {sum(len(json.loads(v)) for v in ledger.url_failures)}. Candidate outcomes and every failed URL are preserved in `data/mimo_pilot/download_ledger_pilot.csv`. Successful decoding does not establish that a photo depicts a complete instrument. Stored files can exceed 2,000 because smoke images, duplicate candidates and unused prefetches are retained. All photo bytes, including gallery thumbnails, remain inside ignored `data/`.', '',
             '### Missing fields', '', table(['Field', 'Missing in pilot', 'Missing in full metadata'], [[k, v, report['missing_metadata'][k]] for k, v in report['missing_selected'].items()]), '',
             'Missing HS labels include absent concepts, unresolved codes, codes too short for a level, and conflicts between multiple source concepts. Evaluation excludes only the missing label for the level being scored; embedding, PCA and clustering still use every image. The label mask is identical across all methods.', '',
             '### HS1 distribution', '', table(['HS1 code', 'Family', 'Pilot images'], [[k or '(missing)', {'1': 'Idiophones', '2': 'Membranophones', '3': 'Chordophones', '4': 'Aerophones', '5': 'Electrophones'}.get(k, 'Missing/ambiguous'), v] for k, v in sample.hs1.value_counts().items()]), '',
             '### Most common HS3 labels', '', table(['HS3', 'Pilot images'], [[k or '(missing)', v] for k, v in sample.hs3.value_counts().head(20).items()]), '',
             'Full class distributions: [HS1](results/pilot/hs1_distribution.csv), [HS2](results/pilot/hs2_distribution.csv), [HS3](results/pilot/hs3_distribution.csv). No class-frequency cutoff was applied.', '',
             '### Museum coverage', '', table(['Museum (15 largest pilot contributions)', 'Images'], [[k, v] for k, v in sample.museum.value_counts().head(15).items()]), '',
             'All represented museums and counts: [museum distribution](results/pilot/museum_distribution.csv). The museum-balanced sampling policy overrepresents small collections and is not representative of the overall MIMO collection.', '',
             '### Names, origin, rights and image size', '',
             table(['Common source title / instrument name', 'Images'], [[k, v] for k, v in sample.title.value_counts().head(10).items()]), '',
             f'{sample.origin.ne("").sum()} pilot records contain origin/place strings. These are retained verbatim and may include manufacture, use or enriched geographic context; provider country is kept separately. Titles are multilingual source descriptions rather than standardized instrument categories.', '',
             table(['Rights statement', 'Images'], [[k or '(missing)', v] for k, v in sample.rights.value_counts().items()]), '',
             f'Image dimensions: median width {sample.width.astype(int).median():.0f} px, median height {sample.height.astype(int).median():.0f} px; {sum(max(int(w), int(h)) / min(int(w), int(h)) > 2 for w, h in zip(sample.width, sample.height))} images have aspect ratio greater than 2. Europeana previews are preferred over source-image fallbacks; white square padding preserves long instruments before model resizing.', '',
             '## Experiments and actual results', '',
             'The smallest working end-to-end run used 32 images from the first 500 metadata records (two museums), producing 16 clustering runs and 48 metric rows. It is a technical smoke test, not a representative performance estimate. All 32 embeddings were verified and reused on rerun. The first model invocation failed due to preprocessing keyword names in the installed OpenCLIP version; this was fixed before any embedding or score was produced.', '',
             f'Full pilot model: **OpenCLIP ViT-B/32**, LAION-2B `laion2b_s34b_b79k`, 512 dimensions, frozen, CPU float32, no fine-tuning. Hub revision `{evidence["model"]["revision"]}`; model SHA256 `{evidence["model"]["weight_sha256"]}`. New pilot embeddings: {evidence["new_images"]}; verified cache hits: {evidence["cache_hits"]}; embedding wall time excluding prior model download and input/model hash checks: {evidence["embedding_seconds"]:.2f} s.', '',
             'Representations: L2-normalized embeddings, and centered non-whitened PCA to 50 components followed by L2 normalization. PCA uses full deterministic SVD and fits all images without labels. K-Means uses Euclidean Lloyd updates, 10 restarts, seeds 0/1/2. Agglomerative clustering uses average linkage and cosine distance, one deterministic run. Fixed k = 5, 10, 20, 40 was specified before scoring; class counts were not passed to clustering.', '',
             f'PCA retained {scores.loc[scores.representation.eq("pca50_l2"), "pca_variance_retained"].iloc[0]:.2%} of embedding variance. Per-level evaluated images: ' + ', '.join(f'{col.upper()} {sample[col].ne("").sum()}' for col in ['hs1', 'hs2', 'hs3']) + '.', '',
             'K-Means entries below average all three seeds; ± is one seed standard deviation. Agglomeration is one run. Runtime is mean clustering wall time per run; PCA preprocessing is reported separately in the CSV. Scores do not include missing labels as a class.', '']
    rows = []
    for (rep, method, k), group in scores.groupby(['representation', 'method', 'requested_k'], sort=True):
        hs3 = group[group.label == 'hs3']
        def fmt(label, metric='AMI'):
            values = group.loc[group.label == label, metric]
            return f'{values.mean():.4f}' + (f' ± {values.std(ddof=1):.4f}' if len(values) > 1 else '')
        rows.append([rep, method, k, fmt('hs1'), fmt('hs2'), fmt('hs3'), f'{hs3.ARI.mean():.4f}', f'{hs3.NMI.mean():.4f}', f'{hs3.cluster_seconds.mean():.3f}'])
    lines += [table(['Representation', 'Method', 'k', 'HS1 AMI', 'HS2 AMI', 'HS3 AMI', 'HS3 ARI', 'HS3 NMI', 'Seconds'], rows), '',
              'Every per-seed AMI, ARI, NMI, cluster count, sample count and runtime: [scores.csv](results/pilot/scores.csv). Saved predictions and [assignments](results/pilot/assignments.csv) allow exact metric recomputation.', '',
              '![Clustering comparison](results/pilot/comparison.png)', '',
              '[Cluster image gallery](results/pilot/cluster_gallery.html). Gallery selection was fixed independently of label scores: raw L2 embeddings, K-Means, k=5, seed 0, ten images nearest each cluster mean. Keep the repository alongside `data/` so its thumbnail links work.', '',
              '## Validation and reproducibility', '',
              f'- {validation["images_redecoded_and_hashed"]} images redecoded and byte/pixel hashes verified; selected record IDs, image IDs and pixel hashes are unique.',
              f'- Exact row order checked for all {validation["identical_order_verified_runs"]} runs. All {validation["metric_rows_independently_recomputed"]} score rows independently recomputed from persisted predictions.',
              '- Cache reuse requires matching image bytes, stable image ID, model weights, preprocessing specification and relevant library versions. Cache array hashes, shape, finiteness and L2 norms are checked. Missing provenance is rejected.',
              f'- Input fingerprint: `{evidence["input_key"]}`. Full ordered IDs and hashes: [image_order.csv](results/pilot/image_order.csv).',
              f'- Credential scan: {validation["credential_scan_text_files"]} saved text files checked; configured key was not found. `.env`, `data/`, and `.venv/` are excluded from Git.',
              '- Eight targeted safeguard tests passed: credential stripping, HS hierarchy/modifiers/conflicts, deterministic sampling order, decoded-pixel deduplication/invalid images, cache rejection, row-order hashing, rate-limit/error safety, and cursor checkpoint resumption/record deduplication.',
              '- The original `mimo_harvest.py` entry point integrates the API-key correction and retains compatibility exports for older spikes. Existing embedding/clustering spikes are preserved; this benchmark uses its own strict cache and explicit average linkage.', '',
              table(['Software', 'Executed version'], [[k, env['packages'].get(k, '')] for k in ['torch', 'torchvision', 'open_clip_torch', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'pillow', 'matplotlib']]), '',
              '[Environment and parameters](results/pilot/environment.json), [embedding manifest](results/pilot/embedding_manifest.json), [validation evidence](results/pilot/validation.json), [dependency lock](requirements-pilot.lock.txt), and [commands/protocol](PIPELINE.md). Compute: CPU with four inference/math threads; embedding seed 42, sampling seed 42, K-Means seeds 0/1/2; average linkage and full-SVD PCA deterministic.', '',
              '## Data-quality issues and limits', '',
              '- Labels are source catalogue annotations, not unquestionable ground truth. In the smoke sample, record `/09102/_CM_0156121` is titled a pakhavaj drum and visually shows a drum, but its only source HS URI `/24` resolves to `111.241.1`, individual gongs. Record `/09102/_CM_0156125` carries both a gong and drum concept; its conflicting HS1 label is excluded. No titles were used to repair labels.',
              '- The first pilot includes instrument parts (for example violin bows), historical representations and possible placeholders. Decode/dimension/variance checks reject broken files, not every semantic data error. No claim is made that all 2,000 images show complete instruments. Inspect the gallery and curate a versioned complete-instrument subset next.',
              '- Exact byte/pixel and URL deduplication cannot remove different views of one object, resized copies or near duplicates. Shared catalogue images and subtle placeholders require perceptual and record-level checks.',
              '- Multiple HS concepts that disagree are masked. Compound `+` notations use their first branch before slicing; audit composite instruments and hierarchical/multilabel scoring next. HS2/HS3 labels with few examples are included, which increases score uncertainty.',
              '- CLIP was pretrained with image-text supervision. Clustering is unsupervised and receives no dataset labels, but this does not establish that the pretrained representation had no instrument-name exposure.',
              '- Museum photographic style, backgrounds, rulers, acquisition conditions, uneven label completeness and geographic bias may affect clusters. Museum-balanced sampling does not remove these confounds. No cross-museum generalization experiment was run in this first benchmark.',
              '- Seed variation measures K-Means initialization stability, not statistical uncertainty over museums or specimens. The pilot provides no confidence interval for future collections. PCA and clustering are transductive on this fixed sample.',
              '- Live source metadata and thumbnail services can change. Preserve raw cached search pages, image bytes and manifests; rerunning live acquisition without the snapshot may yield a different replacement sample after failures.', '',
              '## Recommended next experiments', '',
              '1. Curate a versioned object-only dataset: audit labels, remove parts/representations/placeholders, deduplicate near-identical photos and identify multiple views of one specimen. Compare coverage and scores before/after curation.',
              '2. Run frozen **SigLIP** (and optionally SigLIP 2) on exactly the same ordered input manifest and parameter grid; use encoder-specific preprocessing and hashed caches. Compare with this CLIP baseline without selecting k from HS labels.',
              '3. Run **cross-museum evaluation**: first report within-museum clustering on museums with enough coded instruments and per-museum label coverage. For a held-out-museum K-Means test, fit PCA and cluster centers using other museums only, assign held-out images to frozen centers, and score the held-out labels. Remove cross-museum near duplicates and match class coverage. Report museum-balanced averages, per-museum scores and bootstrap intervals over museums. Average linkage requires a separately declared out-of-sample assignment rule.',
              '4. Compare padding with center crops and higher-resolution source images; inspect errors for elongated instruments. Add DINOv2 as a non-language-supervised representation control.',
              '5. Extend to hierarchical and multilabel HS evaluation and normalized instrument-name categories. Add random-partition baselines, cluster stability and uncertainty intervals; keep descriptive exploration distinct from fixed evaluation.', '',
              '## References', '',
              '- [Europeana Search API](https://europeana.atlassian.net/wiki/spaces/EF/pages/2385739812)',
              '- [Official frozen OpenCLIP model](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)',
              '- [OpenCLIP implementation](https://github.com/mlfoundations/open_clip)', '']
    means = scores[scores.label.eq('hs3')].groupby(['representation', 'method', 'requested_k'])[['AMI', 'ARI', 'NMI']].mean()
    best = means.AMI.idxmax()
    values = means.loc[best]
    raw_largest = int(np.bincount(np.load(out / 'l2_average_cosine_k5_s0.npy')).max())
    pca_largest = int(np.bincount(np.load(out / 'pca50_l2_average_cosine_k5_s0.npy')).max())
    interpretation = ['### Observed pattern', '',
                      f'The highest HS3 AMI among the tested configurations was **{values.AMI:.4f}** for `{best[0]}`, `{best[1]}`, k={best[2]} (ARI {values.ARI:.4f}, NMI {values.NMI:.4f}). This is exploratory selection from this grid, not a held-out estimate or a claim of general superiority.', '',
                      f'Raw L2 average linkage produced highly uneven clusters: its largest k=5 cluster contains {raw_largest:,} of 2,000 images. Under PCA plus L2 normalization, the largest k=5 cluster contains {pca_largest:,}. Both computations succeeded, but the raw-representation result is practically degenerate. PCA here also centers and renormalizes the representation, so the gain cannot be attributed to reducing dimension alone. Add a centered full-dimensional control and a PCA-without-renormalization ablation next.', '',
                      'A 64-image systematic contact sheet and the 50-image cluster gallery were visually inspected. The comparison plot was inspected for legibility and all gallery thumbnail paths were verified. These checks are not an exhaustive semantic review of all 2,000 photographs.', '']
    marker = lines.index('![Clustering comparison](results/pilot/comparison.png)')
    lines[marker:marker] = interpretation
    failure_path = out / 'download_failure_diagnosis.json'
    if failure_path.exists():
        diagnosis = json.loads(failure_path.read_text())
        marker = lines.index('### Missing fields')
        lines[marker:marker] = [f'The one failed candidate, `{diagnosis["record_id"]}` ({diagnosis["museum"]}), returned a decodable {diagnosis["width"]}×{diagnosis["height"]} preview, rejected because its shortest side is below the declared 32-pixel threshold. Its source-image fallback encountered an SSL error. It was replaced by the next valid candidate; this was a size-policy rejection rather than a corrupt JPEG. [Measured diagnosis](results/pilot/download_failure_diagnosis.json).', '']
    museum_counts = sample.museum.value_counts()
    marker = lines.index('## Recommended next experiments')
    lines[marker:marker] = [f'This broad museum pilot has {museum_counts.min()}–{museum_counts.max()} images per museum (median {museum_counts.median():.0f}). Within-museum scores would be unstable at this size. Build an additional focused sample with at least 100 coded images per selected large museum before interpreting within-museum or held-out-museum comparisons.', '']
    summary = scores.groupby(['representation', 'method', 'requested_k', 'label']).agg(
        AMI_mean=('AMI', 'mean'), AMI_std=('AMI', 'std'), ARI_mean=('ARI', 'mean'),
        NMI_mean=('NMI', 'mean'), mean_cluster_seconds=('cluster_seconds', 'mean'),
        n_evaluated=('n_evaluated', 'first'), runs=('run_id', 'nunique')).reset_index()
    summary.to_csv(out / 'scores_summary.csv', index=False)
    cluster_sizes = []
    for run_id in scores.run_id.unique():
        counts = np.bincount(np.load(out / (run_id + '.npy')))
        cluster_sizes.extend({'run_id': run_id, 'cluster': i, 'images': int(c)} for i, c in enumerate(counts))
    pd.DataFrame(cluster_sizes).to_csv(out / 'cluster_sizes.csv', index=False)
    (ROOT / 'FIRST_RESULTS.md').write_text('\n'.join(lines))
    print('Wrote FIRST_RESULTS.md from audited actual results.')


if __name__ == '__main__':
    main()
