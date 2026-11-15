"""Week 6 / P2: RQ1 clustering methods x backbones x blind k on the validated 2,000-image pilot.

Stages (run in order; label access confined to `score`):
  fit     - all clusterings + internal criteria (silhouette, inertia, BIC, eigengap). Never loads HS labels.
  select  - blind k selection from runs.csv / curves only. Never loads HS labels.
  score   - AMI/NMI/ARI against HS1/HS2/HS3 on the fixed phase-2 masks.
Reuses phase-2 saved representations and helpers (results/phase2/source/benchmark.py).
Writes only under reports/week6/.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse, json, sys, time, platform, importlib.metadata, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score, pairwise_distances
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'results/phase2/source'))
from benchmark import sha, write_json, array, csv, now  # reuse phase-2 helpers (no side effects on import)

P2 = ROOT / 'results/phase2'
OUT = ROOT / 'reports/week6/experiments'
PROTO = json.loads((ROOT / 'reports/week6/P1_EXPERIMENT_PROTOCOL.json').read_text())
KS = PROTO['k_grid']
SEEDS = PROTO['seeds']['clustering']
ENCODERS = ['clip', 'siglip']
REPS = list(PROTO['representations'])
HDB_MCS = PROTO['methods']['hdbscan']['min_cluster_size']
NN = PROTO['methods']['spectral_knn']['n_neighbors']
LABELS = ['hs1', 'hs2', 'hs3']
EXPECTED_ORDER_SHA = PROTO['dataset']['ordered_images_sha256']


def load_rep(enc, rep):
    meta = json.loads((P2 / enc / 'representations' / f'{rep}.json').read_text())
    path = P2 / enc / 'representations' / f'{rep}.npy'
    assert sha(path) == meta['array_sha256'], f'hash mismatch {path}'
    Z = np.load(path)  # native float32, identical to phase-2 fits (amendment A1)
    assert Z.shape[0] == 2000 and np.isfinite(Z).all()
    return Z, meta['array_sha256']


def make(method, k, seed, mcs=None):
    if method == 'kmeans':
        return KMeans(n_clusters=k, n_init=10, max_iter=300, tol=1e-4, random_state=seed, algorithm='lloyd')
    if method == 'agglomerative_average_cosine':
        return AgglomerativeClustering(n_clusters=k, linkage='average', metric='cosine')
    if method == 'gmm_diag':
        return GaussianMixture(n_components=k, covariance_type='diag', n_init=3, max_iter=200, tol=1e-3,
                               reg_covar=1e-6, init_params='kmeans', random_state=seed)
    if method == 'spectral_knn':
        return SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=NN, assign_labels='kmeans',
                                  eigen_solver='arpack', n_init=10, random_state=seed, n_jobs=1)
    if method == 'hdbscan':
        return HDBSCAN(min_cluster_size=mcs, metric='euclidean', cluster_selection_method='eom')
    raise ValueError(method)


def sizes(pred):
    lab, cnt = np.unique(pred, return_counts=True)
    noise = int(cnt[lab == -1].sum()) if (lab == -1).any() else 0
    real = cnt[lab != -1]
    return {'actual_k': int(len(real)), 'noise_n': noise, 'noise_pct': 100 * noise / len(pred),
            'largest_cluster_pct': 100 * real.max() / len(pred) if len(real) else np.nan,
            'smallest_cluster_n': int(real.min()) if len(real) else 0,
            'singleton_clusters': int((real == 1).sum())}


def sil(D, pred):
    keep = pred != -1
    if len(np.unique(pred[keep])) < 2 or keep.sum() < 3:
        return np.nan
    return float(silhouette_score(D[np.ix_(keep, keep)], pred[keep], metric='precomputed'))


def eigengap(Z):
    C = kneighbors_graph(Z, n_neighbors=NN, include_self=True)
    A = 0.5 * (C + C.T)
    ncomp = connected_components(A, directed=False)[0]
    d = np.asarray(A.sum(1)).ravel()
    Dm = sp.diags(1 / np.sqrt(d))
    L = np.eye(A.shape[0]) - (Dm @ A @ Dm).toarray()
    ev = np.linalg.eigvalsh(L)[:max(KS) + 2]
    return ev, ncomp


def fit_stage():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'assignments').mkdir(exist_ok=True)
    write_json(OUT / 'environment.json', {'utc': now(), 'python': sys.version, 'platform': platform.platform(),
                                          'threads': 4, 'cpu_count': os.cpu_count(),
                                          'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'pandas', 'matplotlib', 'threadpoolctl']},
                                          'source_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
                                          'protocol_sha256': sha(ROOT / 'reports/week6/P1_EXPERIMENT_PROTOCOL.json')})
    order = pd.read_csv(P2 / 'inputs/image_order.csv', dtype=str, keep_default_na=False)
    runs, sizes_rows, eig_rows = [], [], []
    for enc in ENCODERS:
        for rep in REPS:
            Z, zsha = load_rep(enc, rep)
            Dc = pairwise_distances(Z, metric='cosine'); np.maximum(Dc, 0, out=Dc); np.fill_diagonal(Dc, 0)
            De = pairwise_distances(Z, metric='euclidean'); np.fill_diagonal(De, 0)
            t = time.perf_counter(); ev, ncomp = eigengap(Z); eg_sec = time.perf_counter() - t
            for k in KS:
                eig_rows.append({'encoder': enc, 'representation': rep, 'k': k, 'gap': float(ev[k] - ev[k - 1]),
                                 'lambda_k': float(ev[k - 1]), 'lambda_k1': float(ev[k]), 'graph_components': int(ncomp), 'seconds': eg_sec})
            jobs = [(m, k, s, None) for m in ['kmeans', 'agglomerative_average_cosine', 'gmm_diag', 'spectral_knn']
                    for k in KS for s in (SEEDS if m != 'agglomerative_average_cosine' else [0])]
            jobs += [('hdbscan', None, 0, m) for m in HDB_MCS]
            for method, k, seed, mcs in jobs:
                hp = f'mcs{mcs}' if method == 'hdbscan' else f'k{k}'
                rid = f'{enc}__{rep}__{method}__{hp}__s{seed}'
                est = make(method, k, seed, mcs)
                Zfit = Z.astype(np.float64) if method == 'gmm_diag' else Z  # A1: float64 only for GMM likelihoods
                status, err, extra = 'ok', '', {}
                t = time.perf_counter()
                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter('always')
                        pred = est.fit_predict(Zfit)
                        warn = '; '.join(sorted({str(x.category.__name__) + ':' + str(x.message)[:80] for x in w}))
                except Exception as e:  # record, do not hide
                    pred, status, err, warn = None, 'error', repr(e)[:300], ''
                sec = time.perf_counter() - t
                if pred is not None:
                    pred = np.asarray(pred, dtype=np.int64)
                    if method == 'kmeans': extra['inertia'] = float(est.inertia_)
                    if method == 'gmm_diag':
                        extra['bic'] = float(est.bic(Zfit)); extra['gmm_converged'] = bool(est.converged_)
                    p = OUT / 'assignments' / f'{rid}.npy'; array(p, pred)
                    t2 = time.perf_counter(); sc, se = sil(Dc, pred), sil(De, pred); ssec = time.perf_counter() - t2
                    st = sizes(pred)
                    if st['actual_k'] < 2: status = 'degenerate_single_cluster'
                    for c, n in zip(*np.unique(pred, return_counts=True)):
                        sizes_rows.append({'run_id': rid, 'cluster': int(c), 'images': int(n), 'pct': 100 * n / len(pred)})
                    extra.update({'assignment_path': str(p.relative_to(ROOT)), 'assignment_sha256': sha(p)})
                else:
                    sc = se = ssec = np.nan; st = {}
                runs.append({'run_id': rid, 'dataset': PROTO['dataset']['id'], 'ordered_images_sha256': EXPECTED_ORDER_SHA,
                             'encoder': enc, 'representation': rep, 'representation_sha256': zsha, 'dims': Z.shape[1], 'fit_dtype': str(Zfit.dtype),
                             'method': method, 'requested_k': k, 'min_cluster_size': mcs, 'seed': seed,
                             'hyperparameters': json.dumps({kk: vv for kk, vv in est.get_params().items() if not callable(vv)}, default=str),
                             'fit_seconds': sec, 'silhouette_cosine': sc, 'silhouette_euclidean': se, 'silhouette_seconds': ssec,
                             'status': status, 'error': err, 'warnings': warn, **st, **extra})
                print(f'{rid} {status} {sec:.2f}s sil={sc:.4f} k={st.get("actual_k")} noise={st.get("noise_pct", 0):.1f}', flush=True)
            csv(OUT / 'runs.csv', runs); csv(OUT / 'cluster_sizes.csv', sizes_rows); csv(OUT / 'eigengap.csv', eig_rows)
    write_json(OUT / 'fit_complete.json', {'utc': now(), 'runs': len(runs), 'errors': int(sum(r['status'] == 'error' for r in runs))})


def kneedle(ks, y):
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
    yn = (np.array(y) - min(y)) / (max(y) - min(y))
    # decreasing convex curve: distance below the chord from (0,1) to (1,0)
    d = (1 - x) - yn
    return int(ks[int(np.argmax(d))])


def select_stage():
    """Blind: reads only runs.csv and eigengap.csv (no labels)."""
    runs = pd.read_csv(OUT / 'runs.csv')
    eig = pd.read_csv(OUT / 'eigengap.csv')
    kr = runs[runs.method != 'hdbscan']
    curves = kr.groupby(['encoder', 'representation', 'method', 'requested_k']).agg(
        silhouette_cosine_mean=('silhouette_cosine', 'mean'), silhouette_cosine_sd=('silhouette_cosine', 'std'),
        silhouette_euclidean_mean=('silhouette_euclidean', 'mean'),
        largest_cluster_pct_mean=('largest_cluster_pct', 'mean'), largest_cluster_pct_max=('largest_cluster_pct', 'max'),
        actual_k_mean=('actual_k', 'mean'), fit_seconds_mean=('fit_seconds', 'mean'),
        inertia_mean=('inertia', 'mean'), bic_mean=('bic', 'mean'), n_seeds=('seed', 'nunique')).reset_index()
    curves = curves.merge(eig.rename(columns={'k': 'requested_k', 'gap': 'eigengap'})[['encoder', 'representation', 'requested_k', 'eigengap']],
                          on=['encoder', 'representation', 'requested_k'], how='left')
    csv(OUT / 'blind_k_curves.csv', curves)
    sel = []
    for (enc, rep, m), g in curves.groupby(['encoder', 'representation', 'method']):
        g = g.sort_values('requested_k')
        best = g.silhouette_cosine_mean.max()
        k_sil = int(g[g.silhouette_cosine_mean == best].requested_k.min())  # ties -> smaller k
        sel.append({'encoder': enc, 'representation': rep, 'method': m, 'criterion': 'silhouette_cosine', 'selected_k': k_sil, 'criterion_value': best})
        if m == 'kmeans':
            sel.append({'encoder': enc, 'representation': rep, 'method': m, 'criterion': 'elbow_kneedle_inertia',
                        'selected_k': kneedle(g.requested_k.tolist(), g.inertia_mean.tolist()), 'criterion_value': np.nan})
        if m == 'gmm_diag':
            r = g.loc[g.bic_mean.idxmin()]
            sel.append({'encoder': enc, 'representation': rep, 'method': m, 'criterion': 'bic_min', 'selected_k': int(r.requested_k), 'criterion_value': r.bic_mean})
        if m == 'spectral_knn':
            r = g.loc[g.eigengap.idxmax()]
            sel.append({'encoder': enc, 'representation': rep, 'method': m, 'criterion': 'eigengap', 'selected_k': int(r.requested_k), 'criterion_value': r.eigengap})
    sel = pd.DataFrame(sel); sel['selected_utc'] = now(); sel['labels_accessed'] = False
    csv(OUT / 'blind_k_selection.csv', sel)
    print(sel.to_string(index=False))


def scored(y, p):
    from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
    return AMI(y, p), NMI(y, p), ARI(y, p)


def score_stage():
    masks = pd.read_csv(P2 / 'inputs/evaluation_masks.csv', dtype=str, keep_default_na=False)
    assert sha(P2 / 'inputs/evaluation_masks.csv').startswith('d333e523')
    runs = pd.read_csv(OUT / 'runs.csv')
    rows = []
    for r in runs[runs.status != 'error'].itertuples():
        pred = np.load(ROOT / r.assignment_path)
        assert sha(ROOT / r.assignment_path) == r.assignment_sha256
        variants = ['all'] if r.method != 'hdbscan' else ['noise_as_cluster', 'assigned_only']
        for v in variants:
            for lab in LABELS:
                m = masks[lab + '_valid'].eq('True').to_numpy()
                n_mask = int(m.sum())
                if v == 'assigned_only': m = m & (pred != -1)
                y, p = masks.loc[m, lab].to_numpy(), pred[m]
                a, n, ar = scored(y, p) if m.sum() > 1 else (np.nan,) * 3
                rows.append({'run_id': r.run_id, 'encoder': r.encoder, 'representation': r.representation, 'method': r.method,
                             'requested_k': r.requested_k, 'min_cluster_size': r.min_cluster_size, 'seed': r.seed, 'scoring': v,
                             'label': lab, 'n_mask': n_mask, 'n_evaluated': int(m.sum()), 'coverage_pct': 100 * m.sum() / n_mask,
                             'label_classes': len(np.unique(y)), 'AMI': a, 'NMI': n, 'ARI': ar})
    csv(OUT / 'scores.csv', rows)
    print('scored rows', len(rows))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['fit', 'select', 'score']); a = ap.parse_args()
    with threadpool_limits(limits=4):
        {'fit': fit_stage, 'select': select_stage, 'score': score_stage}[a.stage]()
