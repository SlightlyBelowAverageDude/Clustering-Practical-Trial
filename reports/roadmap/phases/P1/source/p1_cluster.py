"""P1.3: controlled DINOv2 S/B/L x layer (-1, -2, -4) clustering comparison under the frozen protocol FAIR-CMP-v1.

Stages:
  fit    load every DINOv2 dev embedding (S, B, L at 224 px; layers -1/-2/-4), check its hash and that its row order equals
         the frozen manifest MIMO-HS3-DEV12K-v1, apply centered_l2, fit k-Means (seeds 0-4), Agglomerative (average, cosine)
         and Spectral (15-NN, seeds 0-4) at k = 10, save assignments + internal metrics. NO labels are read in this stage.
  score  read HS3/HS1 labels of the 12,000 dev rows, compute AMI/ARI/NMI, aggregate over seeds, write the P1 result tables.
Everything else (factory, transform, silhouette sample, threads) is imported/copied from the frozen P0 protocol.
Usage (repo root): .venv/bin/python reports/roadmap/phases/P1/source/p1_cluster.py fit|score
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')  # FAIR-CMP-v1: 4 BLAS threads
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI,
                             normalized_mutual_info_score as NMI, pairwise_distances)
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[5]
P1 = ROOT / 'reports/roadmap/phases/P1'
ART = P1 / 'artifacts'; ASG = ART / 'assignments'
EMB = ROOT / 'reports/dinov2/embeddings'
sys.path.insert(0, str(ROOT / 'reports/week6/source'))
from p2_benchmark import make, sizes, sil  # noqa: E402  frozen factory (FAIR-CMP-v1 'factory')

PROTO = json.loads((ROOT / 'reports/roadmap/phases/P0/artifacts/EXPERIMENT_PROTOCOL.json').read_text())
MAN = json.loads((ROOT / 'reports/roadmap/phases/P0/artifacts/DEV_MANIFEST.json').read_text())
ORDER_KEY = MAN['object_order']['ordered_inputs_sha256']
MODELS = ['dinov2_s', 'dinov2_b', 'dinov2_l']
LAYERS = [1, 2, 4]  # -1, -2, -4
METHODS = ['kmeans', 'agglomerative_average_cosine', 'spectral_knn']
SEEDS = PROTO['clustering']['seeds']['stochastic_methods']
K = PROTO['clustering']['primary_controlled_reference']['k']
assert K == 10 and SEEDS == [0, 1, 2, 3, 4]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load(model, layer):
    m = json.loads((EMB / f'dev_{model}_224.json').read_text())
    assert m['n'] == 12000 and m['ordered_inputs_sha256'] == ORDER_KEY, f'{model}: not the frozen dev manifest'
    a = m['arrays'][f'-{layer}']
    p = ROOT / a['path']; assert sha(p) == a['sha256'], f'{p}: hash mismatch'
    X = np.load(p); assert X.shape == (12000, a['shape'][1]) and X.dtype == np.float32 and np.isfinite(X).all()
    return X, a


def fit():
    ASG.mkdir(parents=True, exist_ok=True)
    sil_idx = np.random.default_rng(PROTO['clustering']['seeds']['silhouette_sample_seed']).choice(12000, 4000, replace=False)
    runs = []
    t_all = time.perf_counter()
    with threadpool_limits(limits=4):
        for model in MODELS:
            for layer in LAYERS:
                X, a = load(model, layer)
                t = time.perf_counter(); Z = normalize(X - X.mean(axis=0)).astype(np.float32); tsec = time.perf_counter() - t  # centered_l2
                D = pairwise_distances(Z[sil_idx], metric='cosine'); np.maximum(D, 0, out=D); np.fill_diagonal(D, 0)
                variant = f'{model}_224_L{layer}'
                for method in METHODS:
                    for seed in (SEEDS if method != 'agglomerative_average_cosine' else [0]):
                        rid = f'{variant}__centered_l2__{method}__k{K}__s{seed}'
                        est = make(method, K, seed, 20)
                        t0 = time.perf_counter(); pred = est.fit_predict(Z).astype(np.int16); sec = time.perf_counter() - t0
                        p = ASG / f'{rid}.npy'; np.save(p, pred)
                        runs.append({'run_id': rid, 'model': model, 'layer': -layer, 'variant': variant, 'dims': X.shape[1], 'representation': 'centered_l2',
                                     'method': method, 'k': K, 'seed': seed, 'fit_seconds': sec, 'transform_seconds': tsec,
                                     'silhouette_cosine': sil(D, pred[sil_idx]), 'silhouette_rows': len(sil_idx),
                                     'embedding_path': a['path'], 'embedding_sha256': a['sha256'], 'assignment_sha256': sha(p), **sizes(pred.astype(int))})
                        print(rid, f'{sec:.1f}s', flush=True)
                pd.DataFrame(runs).to_csv(ART / 'p1_runs.csv', index=False)
    pd.DataFrame(runs).to_csv(ART / 'p1_runs.csv', index=False)
    print(f'fit done: {len(runs)} fits, {time.perf_counter() - t_all:.0f}s', flush=True)


def score():
    dev = pd.read_csv(ROOT / MAN['source_file']['path'], dtype=str, keep_default_na=False)
    assert sha(ROOT / MAN['source_file']['path']) == MAN['source_file']['sha256'] and (dev.partition == 'dev').all() and len(dev) == 12000
    y3, y1 = dev.hs3.to_numpy(), dev.hs1.to_numpy()
    assert (y3 != '').all() and (y1 != '').all()  # FAIR-CMP-v1 masks are all-true for the frozen manifest
    runs = pd.read_csv(ART / 'p1_runs.csv')
    rows = []
    for r in runs.itertuples():
        p = ASG / f'{r.run_id}.npy'; assert sha(p) == r.assignment_sha256
        pred = np.load(p).astype(int); assert len(pred) == 12000
        rows.append({'run_id': r.run_id, 'HS3_AMI': AMI(y3, pred), 'HS3_ARI': ARI(y3, pred), 'HS3_NMI': NMI(y3, pred), 'HS1_AMI': AMI(y1, pred), 'HS1_ARI': ARI(y1, pred), 'n_evaluated': 12000})
    sc = runs.merge(pd.DataFrame(rows), on='run_id'); sc.to_csv(ART / 'p1_scores_per_run.csv', index=False)
    sd = lambda s: s.std(ddof=1) if len(s) > 1 else np.nan
    g = sc.groupby(['model', 'layer', 'dims', 'method'], sort=False).agg(
        n_seeds=('seed', 'nunique'), HS3_AMI_mean=('HS3_AMI', 'mean'), HS3_AMI_sd=('HS3_AMI', sd), HS3_AMI_min=('HS3_AMI', 'min'), HS3_AMI_max=('HS3_AMI', 'max'),
        HS3_ARI_mean=('HS3_ARI', 'mean'), HS3_ARI_sd=('HS3_ARI', sd), HS3_NMI_mean=('HS3_NMI', 'mean'), HS3_NMI_sd=('HS3_NMI', sd),
        HS1_AMI_mean=('HS1_AMI', 'mean'), HS1_ARI_mean=('HS1_ARI', 'mean'),
        silhouette_cosine_mean=('silhouette_cosine', 'mean'), silhouette_cosine_sd=('silhouette_cosine', sd),
        largest_cluster_pct_mean=('largest_cluster_pct', 'mean'), largest_cluster_pct_max=('largest_cluster_pct', 'max'),
        assigned_clusters_mean=('actual_k', 'mean'), smallest_cluster_n_min=('smallest_cluster_n', 'min'),
        fit_seconds_mean=('fit_seconds', 'mean'), fit_seconds_sd=('fit_seconds', sd), n_evaluated=('n_evaluated', 'first')).reset_index()
    g.insert(0, 'source', 'P1 new execution (FAIR-CMP-v1)')
    g.insert(4, 'representation', 'centered_l2'); g.insert(5, 'k', K)
    g['model_label'] = g.model.map({'dinov2_s': 'DINOv2-S/14', 'dinov2_b': 'DINOv2-B/14', 'dinov2_l': 'DINOv2-L/14'})
    g.to_csv(ART / 'DINOV2_LAYER_ABLATION.csv', index=False, float_format='%.6f')
    # main scaling table: layer -1 for S/B/L + CLIP/SigLIP context rows from the P0-verified baseline reference
    main = g[g.layer == -1].copy()
    base = pd.read_csv(ROOT / 'reports/roadmap/phases/P0/artifacts/BASELINE_REFERENCE.csv')
    ctx = base[(base.role == 'primary_fixed_k10') & base.backbone.isin(['clip', 'siglip']) & base.method.isin(METHODS)]
    ctx = pd.DataFrame({'source': 'historical cached result (Week 6 dev run; P0-verified, not re-run in P1)', 'model': ctx.backbone, 'layer': 'final (projected image embedding)',
                        'dims': ctx.dims, 'representation': 'centered_l2', 'k': K, 'method': ctx.method, 'n_seeds': ctx.n_seeds,
                        'HS3_AMI_mean': ctx.HS3_AMI_mean, 'HS3_AMI_sd': ctx.HS3_AMI_sd, 'HS3_ARI_mean': ctx.HS3_ARI_mean, 'HS3_ARI_sd': ctx.HS3_ARI_sd,
                        'HS3_NMI_mean': ctx.HS3_NMI_mean, 'HS1_AMI_mean': ctx.HS1_AMI_mean, 'HS1_ARI_mean': ctx.HS1_ARI_mean,
                        'silhouette_cosine_mean': ctx.silhouette_cosine_mean, 'largest_cluster_pct_max': ctx.largest_cluster_pct_max,
                        'assigned_clusters_mean': ctx.assigned_clusters_mean, 'fit_seconds_mean': ctx.fit_seconds_mean_saved, 'n_evaluated': 12000,
                        'model_label': ctx.backbone.map({'clip': 'CLIP ViT-B/32', 'siglip': 'SigLIP ViT-B/16'})})
    out = pd.concat([main, ctx], ignore_index=True)
    out.to_csv(ART / 'DINOV2_SBL_12K.csv', index=False, float_format='%.6f')
    print(g[['model', 'layer', 'method', 'HS3_AMI_mean', 'HS3_AMI_sd', 'HS3_ARI_mean', 'largest_cluster_pct_max', 'silhouette_cosine_mean']].round(4).to_string(index=False))


if __name__ == '__main__':
    {'fit': fit, 'score': score}[sys.argv[1]]()
