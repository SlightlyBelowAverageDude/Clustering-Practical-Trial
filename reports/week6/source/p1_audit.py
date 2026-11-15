"""Week 6 / P1: foundation audit. Read-only with respect to all earlier artifacts.

Recomputes dataset counts, verifies embedding caches and their row alignment, and
independently recomputes AMI/NMI/ARI/silhouette from every saved assignment of the
original pilot (results/pilot) and phase 2 (results/phase2/{clip,siglip}).
Outputs go only to reports/week6/.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import hashlib, json, sys, time, platform, importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI,
                             normalized_mutual_info_score as NMI, silhouette_score, pairwise_distances)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/week6'
DATA = ROOT / 'data/mimo_pilot'
P2 = ROOT / 'results/phase2'
LABELS = ['hs1', 'hs2', 'hs3']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def load_sample():
    df = pd.read_csv(DATA / 'sample_pilot.csv', dtype=str, keep_default_na=False)
    return df


def dataset_report(df):
    rep = {'dataset': 'MIMO/Europeana museum-balanced pilot (tag "pilot")',
           'source_file': 'data/mimo_pilot/sample_pilot.csv',
           'source_sha256': sha(DATA / 'sample_pilot.csv'),
           'n_rows': len(df),
           'unique_record_ids': int(df.id.nunique()),
           'unique_image_ids': int(df.image_id.nunique()),
           'unique_byte_sha256': int(df.sha256.nunique()),
           'unique_pixel_sha256': int(df.pixel_sha256.nunique()),
           'duplicate_record_rows': int(df.id.duplicated().sum()),
           'duplicate_byte_rows': int(df.sha256.duplicated().sum()),
           'duplicate_pixel_rows': int(df.pixel_sha256.duplicated().sum())}
    # image files present and byte hashes match
    ok = 0
    for r in df.itertuples():
        p = DATA / r.file
        if p.exists() and sha(p) == r.sha256:
            ok += 1
    rep['image_files_present_and_hash_verified'] = ok
    levels = {}
    for lab in LABELS:
        valid = df[lab].ne('')
        vc = df.loc[valid, lab].value_counts()
        levels[lab] = {'valid': int(valid.sum()), 'missing': int((~valid).sum()),
                       'missing_pct': round(100 * (~valid).mean(), 2),
                       'classes': int(vc.size),
                       'largest_class': str(vc.index[0]), 'largest_class_n': int(vc.iloc[0]),
                       'largest_class_pct_of_valid': round(100 * vc.iloc[0] / valid.sum(), 2),
                       'singleton_classes': int((vc == 1).sum()),
                       'classes_with_ge_10': int((vc >= 10).sum()),
                       'distribution': {str(k): int(v) for k, v in vc.items()}}
    rep['labels'] = levels
    rep['hs_multiple_true'] = int(df.hs_multiple.eq('True').sum())
    mus = df.museum.value_counts()
    rep['museums'] = {'distinct': int(mus.size), 'min_images': int(mus.min()), 'median_images': float(mus.median()),
                      'max_images': int(mus.max()),
                      'museums_without_any_hs3': int((df.groupby('museum').hs3.apply(lambda s: s.ne('').sum()) == 0).sum()),
                      'top10': {str(k): int(v) for k, v in mus.head(10).items()}}
    # near-duplicate screening from phase 2 review (not confirmed)
    pairs = pd.read_csv(P2 / 'review/possible_duplicate_pairs.csv')
    rep['near_duplicate_candidates'] = {'source': 'results/phase2/review/possible_duplicate_pairs.csv',
                                        'candidate_pairs': int(len(pairs)),
                                        'rows_involved': int(len(set(pairs.row_a) | set(pairs.row_b))),
                                        'status': 'screening flags only; not human-confirmed; no exclusion applied'}
    return rep


def check_alignment(df):
    res = {}
    order = [{'id': r.id, 'image_id': r.image_id, 'sha256': r.sha256} for r in df.itertuples()]
    clip_m = json.loads((ROOT / 'results/pilot/embedding_manifest.json').read_text())
    sig_m = json.loads((P2 / 'siglip/embedding_manifest.json').read_text())
    res['clip_manifest_inputs_match_sample_order'] = clip_m['descriptor']['inputs'] == order
    res['siglip_manifest_inputs_match_sample_order'] = sig_m['inputs'] == order
    io = pd.read_csv(P2 / 'inputs/image_order.csv', dtype=str, keep_default_na=False)
    res['phase2_image_order_matches_sample'] = io.image_id.tolist() == df.image_id.tolist() and io.sha256.tolist() == df.sha256.tolist()
    masks = pd.read_csv(P2 / 'inputs/evaluation_masks.csv', dtype=str, keep_default_na=False)
    res['phase2_masks_match_sample_labels'] = all(masks[l].tolist() == df[l].tolist() for l in LABELS) and masks.image_id.tolist() == df.image_id.tolist()
    res['phase2_mask_counts'] = {l: int((masks[l + '_valid'] == 'True').sum()) for l in LABELS}
    po = pd.read_csv(ROOT / 'results/pilot/image_order.csv', dtype=str, keep_default_na=False)
    res['pilot_image_order_matches_sample'] = po.image_id.tolist() == df.image_id.tolist()
    return res, clip_m, sig_m


def embedding_checks(clip_m, sig_m):
    out = []
    def describe(name, path, expected_sha=None, model=None):
        X = np.load(path, allow_pickle=False)
        norms = np.linalg.norm(X, axis=1)
        h = sha(path)
        out.append({'name': name, 'path': str(Path(path).relative_to(ROOT)), 'model': model,
                    'shape': list(X.shape), 'dtype': str(X.dtype),
                    'nan': int(np.isnan(X).sum()), 'inf': int(np.isinf(X).sum()),
                    'row_norm_min': float(norms.min()), 'row_norm_max': float(norms.max()),
                    'l2_normalized': bool(np.allclose(norms, 1, atol=1e-4)),
                    'mean_vector_norm': float(np.linalg.norm(X.mean(0))),
                    'centered': bool(np.linalg.norm(X.mean(0)) < 1e-3),
                    'sha256': h, 'expected_sha256': expected_sha,
                    'hash_ok': (h == expected_sha) if expected_sha else None})
        return X
    describe('clip_raw', clip_m['array_path'], clip_m['array_sha256'], 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K (OpenCLIP)')
    describe('siglip_raw', sig_m['array_path'], sig_m['array_sha256'], 'timm/ViT-B-16-SigLIP @41f5757')
    for enc in ['clip', 'siglip']:
        for js in sorted((P2 / enc / 'representations').glob('*.json')):
            meta = json.loads(js.read_text())
            describe(f'{enc}_{js.stem}', js.with_suffix('.npy'), meta['array_sha256'], enc)
            out[-1]['variance_retained'] = meta.get('variance_retained')
    return out


def silhouette_cos(Z, pred, D):
    if len(np.unique(pred)) < 2:
        return np.nan
    return silhouette_score(D, pred, metric='precomputed')


def recompute_phase2(df):
    rows = []
    masks = {l: df[l].ne('').to_numpy() for l in LABELS}
    for enc in ['clip', 'siglip']:
        sc = pd.read_csv(P2 / enc / 'scores.csv')
        runs = pd.read_csv(P2 / enc / 'runs.csv')
        reps = {}
        for r in runs.itertuples():
            if r.representation not in reps:
                Z = np.load(P2 / enc / 'representations' / f'{r.representation}.npy')
                D = pairwise_distances(Z, metric='cosine'); np.maximum(D, 0, out=D); np.fill_diagonal(D, 0)
                reps[r.representation] = D
            pred_path = P2 / enc / 'predictions' / f'{r.run_id}.npy'
            pred = np.load(pred_path)
            assert len(pred) == len(df)
            sil = silhouette_cos(None, pred, reps[r.representation])
            for lab in LABELS:
                m = masks[lab]
                y = df.loc[m, lab].to_numpy(); p = pred[m]
                rep_row = sc[(sc.run_id == r.run_id) & (sc.label == lab)].iloc[0]
                a, n, ar = AMI(y, p), NMI(y, p), ARI(y, p)
                rows.append({'source': f'results/phase2/{enc}', 'run_id': r.run_id, 'dataset': 'mimo_pilot_2000',
                             'encoder': enc, 'representation': r.representation, 'method': r.method,
                             'requested_k': r.requested_k, 'actual_k': len(np.unique(pred)), 'seed': r.seed,
                             'label': lab, 'n_evaluated': int(m.sum()), 'label_classes': len(np.unique(y)),
                             'AMI_recomputed': a, 'AMI_reported': rep_row.AMI, 'NMI_recomputed': n, 'NMI_reported': rep_row.NMI,
                             'ARI_recomputed': ar, 'ARI_reported': rep_row.ARI,
                             'silhouette_cosine_recomputed': sil, 'silhouette_cosine_reported': rep_row.silhouette_cosine,
                             'predictions_sha256_ok': sha(pred_path) == r.predictions_sha256,
                             'largest_cluster_pct': 100 * np.bincount(pred).max() / len(pred)})
        print(enc, 'recomputed', flush=True)
    return rows


def recompute_pilot(df):
    rows = []
    sc = pd.read_csv(ROOT / 'results/pilot/scores.csv')
    masks = {l: df[l].ne('').to_numpy() for l in LABELS}
    for r in sc.itertuples():
        pred = np.load(ROOT / 'results/pilot' / f'{r.run_id}.npy')
        m = masks[r.label]; y = df.loc[m, r.label].to_numpy(); p = pred[m]
        rows.append({'source': 'results/pilot', 'run_id': r.run_id, 'dataset': 'mimo_pilot_2000', 'encoder': 'clip',
                     'representation': r.representation, 'method': r.method, 'requested_k': r.requested_k,
                     'actual_k': len(np.unique(pred)), 'seed': r.seed, 'label': r.label, 'n_evaluated': int(m.sum()),
                     'label_classes': len(np.unique(y)),
                     'AMI_recomputed': AMI(y, p), 'AMI_reported': r.AMI, 'NMI_recomputed': NMI(y, p), 'NMI_reported': r.NMI,
                     'ARI_recomputed': ARI(y, p), 'ARI_reported': r.ARI,
                     'silhouette_cosine_recomputed': np.nan, 'silhouette_cosine_reported': np.nan,
                     'predictions_sha256_ok': None, 'largest_cluster_pct': 100 * np.bincount(pred).max() / len(pred)})
    return rows


def full_metadata_context():
    md = pd.read_csv(ROOT / 'data/mimo/records.csv', dtype=str, keep_default_na=False, usecols=lambda c: c in ['id', 'museum', 'hs3', 'hs1'])
    md2 = pd.read_csv(DATA / 'metadata.csv', dtype=str, keep_default_na=False, usecols=['id', 'museum', 'hs1', 'hs3'])
    return {'data/mimo/records.csv (older spike harvest)': {'rows': len(md), 'unique_ids': int(md.id.nunique()),
                                                            'museums': int(md.museum.nunique()) if 'museum' in md else None},
            'data/mimo_pilot/metadata.csv (pilot metadata snapshot)': {'rows': len(md2), 'unique_ids': int(md2.id.nunique()),
                                                                       'museums': int(md2.museum.nunique()),
                                                                       'hs1_valid': int(md2.hs1.ne('').sum()),
                                                                       'hs3_valid': int(md2.hs3.ne('').sum())},
            'cleaned_59661_photos': 'NOT PRESENT LOCALLY: referenced in EXPLAINER.md/IDEAS.md (runs/clean/...), the runs/ directory does not exist in this repository',
            'scored_27348_objects': 'NOT PRESENT LOCALLY: referenced in EXPLAINER.md/IDEAS.md; no ids, embeddings or assignments available',
            'reserved_final_holdout': 'No held-out split file exists in the repository. The 2,000-image pilot has already been explored in phase 1/2 and is a development set; it must not be described as held-out.'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    df = load_sample()
    rep = dataset_report(df)
    align, clip_m, sig_m = check_alignment(df)
    rep['alignment'] = align
    rep['embeddings'] = embedding_checks(clip_m, sig_m)
    rep['dinov2'] = {'available_locally': False,
                     'searched': ['data/', 'results/', '~/.cache/huggingface', 'runs/ (absent)', 'filesystem search for *dinov2*'],
                     'extraction_code': 'spikes/common.py ENCODERS (timm vit_base_patch14_dinov2.lvd142m) — old spike pipeline, not the validated pilot pipeline',
                     'status': 'BLOCKED: requires downloading ~350 MB DINOv2-B weights from Hugging Face; not downloaded without explicit approval. CPU inference on 2,000 images is otherwise feasible (~5-10 min).'}
    rep['larger_benchmark'] = full_metadata_context()
    rep['generated_utc'] = now()
    (OUT / 'P1_DATASET_REPORT.json').write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k not in ('labels', 'embeddings')}, indent=1)[:3000])

    rows = recompute_pilot(df) + recompute_phase2(df)
    res = pd.DataFrame(rows)
    for m in ['AMI', 'NMI', 'ARI', 'silhouette_cosine']:
        res[m + '_absdiff'] = (res[m + '_recomputed'] - res[m + '_reported']).abs()
    res['status'] = np.where((res[['AMI_absdiff', 'NMI_absdiff', 'ARI_absdiff']].max(axis=1) < 1e-9)
                             & (res.silhouette_cosine_absdiff.fillna(0) < 1e-5), 'reproduced', 'MISMATCH')
    ext = pd.DataFrame([
        {'source': 'EXPLAINER.md / IDEAS.md (research plan)', 'run_id': 'dinov2b_pca50_kmeans_k=n_classes', 'dataset': 'cleaned MIMO 27,348 scored objects (not local)',
         'encoder': 'dinov2_b', 'representation': 'pad_pca50', 'method': 'kmeans', 'label': 'hs3', 'label_classes': 13,
         'AMI_reported': 0.55, 'status': 'NOT_REPRODUCIBLE_LOCALLY: dataset ids, embeddings and assignments absent (runs/ missing); IDEAS.md lists HS3=0.55, HS2=0.53, and 0.53 at HS3 in the 40,697-object depth table — the "0.53 at HS3" figure is ambiguous'},
        {'source': 'EXPLAINER.md / IDEAS.md (research plan)', 'run_id': 'dinov2b_logreg_probe_5fold', 'dataset': 'cleaned MIMO 27,348 scored objects (not local)',
         'encoder': 'dinov2_b', 'representation': 'pad', 'method': 'supervised_linear_probe', 'label': 'hs3', 'label_classes': 13,
         'AMI_reported': 0.83, 'status': 'NOT_REPRODUCIBLE_LOCALLY: supervised reference; dataset/embeddings absent'}])
    res = pd.concat([res, ext], ignore_index=True)
    res.to_csv(OUT / 'P1_BASELINE_RESULTS.csv', index=False)
    loc = res[res.source.str.startswith('results')]
    print('rows', len(loc), 'mismatches', int((loc.status != 'reproduced').sum()),
          'max AMI diff', loc.AMI_absdiff.max(), 'max sil diff', loc.silhouette_cosine_absdiff.max(),
          'pred hash ok', loc.predictions_sha256_ok.dropna().all(), 'secs', round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    main()
