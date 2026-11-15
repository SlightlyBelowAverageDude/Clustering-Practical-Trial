"""Phase 2: frozen-input ablations. All label access is confined to scoring."""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/phase2'
DATA = ROOT / 'data/mimo_pilot'
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data/phase2/matplotlib'))
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                            normalized_mutual_info_score, silhouette_score, pairwise_distances)
from threadpoolctl import threadpool_limits

REPS = ['l2', 'centered_l2', 'pca50', 'pca50_l2', 'pca100_l2']
KS = [5, 10, 20, 40]
SEEDS = [0, 1, 2, 3, 4]
LABELS = ['hs1', 'hs2', 'hs3']

def now(): return datetime.now(timezone.utc).isoformat()
def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
def digest(x): return hashlib.sha256(x).hexdigest()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.part'); tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)); tmp.replace(path)
def csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.part'); pd.DataFrame(rows).to_csv(tmp, index=False); tmp.replace(path)
def array(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.part')
    with tmp.open('wb') as f: np.save(f, x, allow_pickle=False)
    tmp.replace(path)

def snapshot():
    path = OUT / 'inputs/original_snapshot.json'
    if path.exists():
        old = json.loads(path.read_text())
        for rel, checksum in old['files'].items():
            assert sha(ROOT / rel) == checksum, 'Original changed: ' + rel
        return
    files = list((ROOT / 'results/pilot').rglob('*'))
    files += [ROOT / 'FIRST_RESULTS.md', ROOT / 'PIPELINE.md', DATA / 'sample_pilot.csv']
    files += list((ROOT / 'spikes').glob('pilot*.py'))
    write_json(path, {'recorded_utc': now(), 'files': {str(p.relative_to(ROOT)): sha(p) for p in files if p.is_file()}})

def inputs(check_images=True):
    snapshot()
    manifest = json.loads((ROOT / 'results/pilot/embedding_manifest.json').read_text())
    df = pd.read_csv(DATA / 'sample_pilot.csv', dtype=str, keep_default_na=False)
    assert len(df) == 2000 and df.image_id.is_unique and df.pixel_sha256.is_unique
    order = [{'id': r.id, 'image_id': r.image_id, 'sha256': r.sha256} for r in df.itertuples()]
    assert order == manifest['descriptor']['inputs']
    assert digest(canonical(manifest['descriptor']).encode()) == manifest['input_key']
    if check_images:
        for r in df.itertuples(): assert sha(DATA / r.file) == r.sha256
    assert sha(manifest['array_path']) == manifest['array_sha256']
    X = np.load(manifest['array_path'], allow_pickle=False)
    assert X.shape == (2000, 512) and np.isfinite(X).all()
    assert np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-4)
    if not (OUT / 'inputs/image_order.csv').exists():
        csv(OUT / 'inputs/image_order.csv', [{'row': i, **r} for i, r in enumerate(order)])
        masks = df[['image_id', *LABELS]].copy(); masks.insert(0, 'row', range(len(df)))
        for label in LABELS: masks[label + '_valid'] = masks[label].ne('')
        csv(OUT / 'inputs/evaluation_masks.csv', masks)
        write_json(OUT / 'inputs/clip_embedding_manifest.json', manifest)
        write_json(OUT / 'inputs/input_identity.json', {'ordered_images_sha256': digest(canonical(order).encode()), 'sample_csv_sha256': sha(DATA / 'sample_pilot.csv'), 'masks_sha256': sha(OUT / 'inputs/evaluation_masks.csv'), 'original_clip_input_key': manifest['input_key']})
    return df, X, manifest

def environment(stage):
    write_json(OUT / f'{stage}_environment.json', {'utc': now(), 'python': sys.version, 'platform': platform.platform(), 'threads': 4, 'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}, 'command': sys.argv, 'source_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}})

def transform(X, name, path):
    started = time.perf_counter()
    params = {'representation': name, 'fit_rows': len(X), 'input_shape': list(X.shape)}
    if name == 'l2': Z = normalize(X)
    elif name == 'centered_l2':
        mean = X.mean(axis=0); Z = normalize(X - mean)
        np.savez(path.with_suffix('.npz'), mean=mean)
    else:
        d = 100 if name == 'pca100_l2' else 50
        pca = PCA(n_components=d, svd_solver='full', whiten=False)
        Z = pca.fit_transform(X)
        params['variance_retained'] = float(pca.explained_variance_ratio_.sum())
        np.savez(path.with_suffix('.npz'), mean=pca.mean_, components=pca.components_, explained_variance_ratio=pca.explained_variance_ratio_)
        if name.endswith('_l2'): Z = normalize(Z)
    assert np.isfinite(Z).all() and (np.linalg.norm(Z, axis=1) > 1e-8).all()
    seconds = time.perf_counter() - started
    array(path, Z)
    params.update({'seconds': seconds, 'array_sha256': sha(path), 'shape': list(Z.shape)})
    write_json(path.with_suffix('.json'), params)
    return Z, params

def estimator(method, k, seed):
    if method == 'kmeans':
        return KMeans(n_clusters=k, n_init=10, max_iter=300, tol=1e-4, random_state=seed, algorithm='lloyd')
    return AgglomerativeClustering(n_clusters=k, linkage='average', metric='cosine')

def metrics(y, pred):
    if len(set(y)) < 2: return {'AMI': np.nan, 'ARI': np.nan, 'NMI': np.nan}
    return {'AMI': adjusted_mutual_info_score(y, pred), 'ARI': adjusted_rand_score(y, pred), 'NMI': normalized_mutual_info_score(y, pred)}

def size_stats(pred):
    counts = np.unique(pred, return_counts=True)[1]; p = counts / len(pred)
    return {'actual_k': len(counts), 'largest_cluster_pct': 100 * p.max(), 'smallest_cluster_n': counts.min(), 'singleton_clusters': int((counts == 1).sum()), 'normalized_cluster_entropy': float(-(p * np.log(p)).sum() / np.log(len(counts))), 'size_cv': float(counts.std() / counts.mean())}

def run_id(encoder, rep, method, k, seed): return f'{encoder}__{rep}__{method}__k{k}__s{seed}'

def full_grid(encoder, X, df):
    out = OUT / encoder; out.mkdir(parents=True, exist_ok=True)
    rows, runs, sizes = [], [], []
    for rep in REPS:
        Z, params = transform(X, rep, out / 'representations' / (rep + '.npy'))
        distances = {}
        for metric in ['cosine', 'euclidean']:
            D = pairwise_distances(Z, metric=metric, n_jobs=1); np.maximum(D, 0, out=D); np.fill_diagonal(D, 0)
            distances[metric] = D
        for method in ['kmeans', 'average_cosine']:
            for k in KS:
                for seed in SEEDS if method == 'kmeans' else [0]:
                    rid = run_id(encoder, rep, method, k, seed)
                    started = time.perf_counter(); pred = estimator(method, k, seed).fit_predict(Z); runtime = time.perf_counter() - started
                    path = out / 'predictions' / (rid + '.npy'); array(path, pred)
                    start_sil = time.perf_counter()
                    sc = silhouette_score(distances['cosine'], pred, metric='precomputed')
                    se = silhouette_score(distances['euclidean'], pred, metric='precomputed')
                    common = {'run_id': rid, 'encoder': encoder, 'representation': rep, 'method': method, 'requested_k': k, 'seed': seed, 'n_images': len(df), 'dimensions': Z.shape[1], 'cluster_seconds': runtime, 'transform_seconds': params['seconds'], 'silhouette_cosine': sc, 'silhouette_euclidean': se, 'silhouette_method_metric': se if method == 'kmeans' else sc, 'silhouette_seconds': time.perf_counter() - start_sil, 'predictions_sha256': sha(path), 'representation_sha256': params['array_sha256'], **size_stats(pred)}
                    runs.append(common)
                    cs = np.bincount(pred)
                    sizes.extend({'run_id': rid, 'cluster': i, 'images': int(c), 'percentage': 100 * c / len(df)} for i, c in enumerate(cs))
                    for label in LABELS:
                        mask = df[label].ne('').to_numpy()
                        rows.append({**common, 'label': label, 'n_evaluated': int(mask.sum()), 'label_classes': df.loc[mask, label].nunique(), **metrics(df.loc[mask, label], pred[mask])})
                print(encoder, rep, method, 'k', k, 'complete', flush=True)
                csv(out / 'scores.csv', rows); csv(out / 'runs.csv', runs); csv(out / 'cluster_sizes.csv', sizes)
    summary = pd.DataFrame(rows).groupby(['encoder', 'representation', 'method', 'requested_k', 'label']).agg(AMI_mean=('AMI', 'mean'), AMI_seed_sd=('AMI', 'std'), ARI_mean=('ARI', 'mean'), NMI_mean=('NMI', 'mean'), largest_cluster_pct=('largest_cluster_pct', 'mean'), silhouette_cosine=('silhouette_cosine', 'mean'), cluster_seconds=('cluster_seconds', 'mean'), runs=('run_id', 'size')).reset_index()
    csv(out / 'summary.csv', summary)
    write_json(out / 'grid_complete.json', {'utc': now(), 'runs': len(runs), 'score_rows': len(rows), 'success': True})

def task1_report():
    d = pd.read_csv(OUT / 'clip/summary.csv'); rows = []
    lines = ['# Task 1: CLIP transformation ablation', '', 'Same frozen 2,000 images; 1,549 evaluated HS3 labels. K-Means: seeds 0–4, n_init=10. Average linkage: cosine, deterministic. Fixed k=5/10/20/40. Descriptive evaluation on the existing pilot, not held-out estimation.', '', '| Representation | Method | k | HS3 AMI mean | seed SD | ARI | NMI | largest cluster % | cosine silhouette |', '|---|---|---|---|---|---|---|---|---|']
    for r in d[d.label.eq('hs3')].itertuples():
        lines.append(f'| {r.representation} | {r.method} | {r.requested_k} | {r.AMI_mean:.4f} | {r.AMI_seed_sd:.4f} | {r.ARI_mean:.4f} | {r.NMI_mean:.4f} | {r.largest_cluster_pct:.2f} | {r.silhouette_cosine:.4f} |')
    equivalence = []
    for k in KS:
        a = np.load(OUT / 'clip/predictions' / (run_id('clip','pca50','average_cosine',k,0)+'.npy'))
        b = np.load(OUT / 'clip/predictions' / (run_id('clip','pca50_l2','average_cosine',k,0)+'.npy'))
        equivalence.append({'k': k, 'partition_ARI': adjusted_rand_score(a,b), 'exact_array_equal': np.array_equal(a,b)})
    csv(OUT / 'clip/cosine_normalization_check.csv', equivalence)
    lines += ['', 'Post-PCA positive row scaling cannot change cosine distances mathematically. Check saved partition agreement in `clip/cosine_normalization_check.csv`; any disagreement would be a numerical tie effect, not a normalization benefit.', '', 'Centering is isolated by raw L2 vs centered full-dimensional L2. Dimensionality is examined by centered full-dimensional vs PCA50+L2 and PCA100+L2. Post-PCA normalization is isolated by PCA50 vs PCA50+L2. For K-Means it can change Euclidean geometry; for average cosine it should not.', '', 'Full per-seed metrics/runtimes, exact silhouettes on all images, representations, PCA parameters, cluster-size distributions and predictions are saved under clip/.']
    (OUT / 'TASK1_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(d[(d.label=='hs3') & (d.method=='average_cosine')].to_string(index=False), flush=True)

def declare():
    path = OUT / 'DECLARED_PROTOCOL.json'
    protocol = {'declared_utc': now(), 'representations': REPS, 'ks': KS, 'kmeans_seeds': SEEDS, 'n_init': 10, 'max_iter': 300, 'tol': 1e-4, 'linkage': 'average', 'agglomerative_metric': 'cosine', 'pca_solver': 'full', 'pca_whiten': False, 'threads': 4, 'silhouettes': 'exact, all 2000 rows, cosine and Euclidean; report method-metric and common cosine', 'random_baseline': '20 full-row prediction permutations preserving sizes for seed-0 run of every configuration, seeds 1000..1019', 'subset_protocol': {'rows': 1600, 'seeds': [2026,2027,2028,2029,2030], 'replacement': False, 'representation': 'centered_l2', 'k': 10, 'methods': ['kmeans','average_cosine'], 'kmeans_seeds': SEEDS, 'refit_transformation_per_subset': True}, 'comparison': 'paired identical configurations, all grid cells; report fixed centered_l2 k10 anchor and equally weighted grid mean; no selection of winner from max HS AMI; descriptive only', 'labels': 'HS labels only for evaluation; separately used to stratify requested human review, never to fit features or clusters', 'review_seed': 20261102, 'review_target': 180}
    if not path.exists(): write_json(path, protocol)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['clip','siglip_grid']); a=ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); declare(); environment(a.stage)
    df, X, manifest = inputs()
    with threadpool_limits(limits=4):
        if a.stage == 'clip': full_grid('clip', X, df); task1_report()
        else:
            m = json.loads((OUT/'siglip/embedding_manifest.json').read_text())
            assert sha(m['array_path']) == m['array_sha256']
            assert m['inputs'] == manifest['descriptor']['inputs']
            full_grid('siglip', np.load(m['array_path'],allow_pickle=False), df)

if __name__ == '__main__': main()
