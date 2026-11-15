"""Three-backbone clustering benchmark (CLIP, SigLIP, DINOv2 S/B/L) on the pilot and on the development partition.

Stages:
  pilot_fit    all clusterings + internal metrics on the 2,000 pilot rows (no labels read)
  pilot_score  HS1/HS2/HS3 AMI/NMI/ARI on the phase-2 masks
  probe        supervised 5-fold linear probe (separate table; never used for clustering)
  dev          development partition (12,000 rows): fit, then score (labels read after all fits)
Reuses reports/week6/source/p2_benchmark.py (make, sizes, sil) and the phase-2 transform rule. Protocol:
reports/dinov2/DINOV2_PROTOCOL.json (declared before any fit).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')  # 4 = declared protocol; 6 changes PCA rounding (2e-5)
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import (adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, normalized_mutual_info_score as NMI,
                             pairwise_distances, silhouette_score, accuracy_score, balanced_accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'reports/week6/source'))
from p2_benchmark import make, sizes, sil, sha  # noqa: E402

OUT = ROOT / 'reports/dinov2'
P2 = ROOT / 'results/phase2'
REPS = ['l2', 'centered_l2', 'pca50_l2', 'pca100_l2']
SEEDS = [0, 1, 2, 3, 4]
FULL_METHODS = ['kmeans', 'agglomerative_average_cosine', 'spectral_knn', 'gmm_diag', 'hdbscan']
CORE = ['kmeans', 'agglomerative_average_cosine', 'spectral_knn']
OLD = '/home/hayvan/Programming/LMU/CMVAL/ClusteringmeetsVisionandLanguage/Clustering-Practical-Trial-main'


def emb(set_, name):
    """Return (X float32, provenance) for an embedding variant."""
    if set_ == 'pilot' and name in ('clip', 'siglip'):
        man = json.loads((ROOT / ('results/pilot/embedding_manifest.json' if name == 'clip' else 'results/phase2/siglip/embedding_manifest.json')).read_text())
        p = Path(man['array_path'].replace(OLD, str(ROOT))); assert sha(p) == man['array_sha256']
        return np.load(p), {'source': str(p.relative_to(ROOT)), 'sha256': man['array_sha256']}
    if name in ('clip', 'siglip'):
        man = json.loads((OUT / 'embeddings' / f'{set_}_{name}.json').read_text())
    else:  # dinov2_b_518_L1
        m, size, layer = name.rsplit('_', 2)
        man = json.loads((OUT / 'embeddings' / f'{set_}_{m}_{size}.json').read_text())
        man['array'] = man['arrays']['-' + layer[1:]]
    p = ROOT / man['array']['path']; assert sha(p) == man['array']['sha256']
    return np.load(p), {'source': man['array']['path'], 'sha256': man['array']['sha256']}


def transform(X, rep):
    """Identical to results/phase2/source/benchmark.py transform()."""
    if rep == 'l2':
        return normalize(X)
    if rep == 'centered_l2':
        return normalize(X - X.mean(axis=0))
    d = 100 if rep == 'pca100_l2' else 50
    return normalize(PCA(n_components=d, svd_solver='full', whiten=False).fit_transform(X))


def fit_grid(set_, variants, out, k_list, sil_sample=None):
    (out / 'assignments').mkdir(parents=True, exist_ok=True)
    runs = []
    for name, methods, reps in variants:
        X, prov = emb(set_, name)
        for rep in reps:
            t = time.perf_counter(); Z = transform(X, rep).astype(np.float32); tsec = time.perf_counter() - t
            if set_ == 'pilot' and name in ('clip', 'siglip'):  # parity with saved phase-2 representations
                ref = np.load(P2 / name / 'representations' / f'{rep}.npy')
                assert np.allclose(Z, ref, atol=1e-5), f'parity failed {name} {rep}'
            if sil_sample:
                idx = np.random.default_rng(0).choice(len(Z), sil_sample, replace=False); D = pairwise_distances(Z[idx], metric='cosine')
            else:
                idx = None; D = pairwise_distances(Z, metric='cosine')
            np.maximum(D, 0, out=D); np.fill_diagonal(D, 0)
            for method in methods:
                for k in (k_list if method != 'hdbscan' else [None]):
                    for seed in (SEEDS if method in ('kmeans', 'spectral_knn', 'gmm_diag') else [0]):
                        rid = f'{name}__{rep}__{method}__k{k}__s{seed}'
                        est = make(method, k, seed, 20)
                        Zf = Z.astype(np.float64) if method == 'gmm_diag' else Z
                        t0 = time.perf_counter()
                        try:
                            pred = est.fit_predict(Zf).astype(np.int16); status = 'ok'
                        except Exception as e:  # recorded, not hidden
                            runs.append({'run_id': rid, 'variant': name, 'representation': rep, 'method': method, 'k': k, 'seed': seed, 'status': 'error', 'error': repr(e)[:200]}); continue
                        sec = time.perf_counter() - t0
                        p = out / 'assignments' / f'{rid}.npy'; np.save(p, pred)
                        s = sil(D, pred[idx] if idx is not None else pred)
                        runs.append({'run_id': rid, 'set': set_, 'variant': name, 'dims': X.shape[1], 'representation': rep, 'method': method, 'k': k, 'seed': seed,
                                     'fit_seconds': sec, 'transform_seconds': tsec, 'silhouette_cosine': s, 'silhouette_rows': len(idx) if idx is not None else len(Z),
                                     'status': status, 'embedding_source': prov['source'], 'embedding_sha256': prov['sha256'], 'assignment_sha256': sha(p), **sizes(pred.astype(int))})
            print(set_, name, rep, 'done', flush=True)
            pd.DataFrame(runs).to_csv(out / 'runs.csv', index=False)
    pd.DataFrame(runs).to_csv(out / 'runs.csv', index=False)


def score(out, labels_df, label_cols):
    runs = pd.read_csv(out / 'runs.csv'); rows = []
    for r in runs[runs.status == 'ok'].itertuples():
        pred = np.load(out / 'assignments' / f'{r.run_id}.npy').astype(int)
        for v in (['all_rows'] if r.method != 'hdbscan' else ['noise_as_cluster', 'assigned_only']):
            for lab in label_cols:
                m = labels_df[lab].ne('').to_numpy()
                if v == 'assigned_only': m = m & (pred != -1)
                y, p = labels_df.loc[m, lab].to_numpy(), pred[m]
                rows.append({'run_id': r.run_id, 'variant': r.variant, 'representation': r.representation, 'method': r.method, 'k': r.k, 'seed': r.seed, 'scoring': v,
                             'label': lab, 'n_evaluated': int(m.sum()), 'classes': len(set(y)), 'AMI': AMI(y, p), 'NMI': NMI(y, p), 'ARI': ARI(y, p)})
    sc = pd.DataFrame(rows); sc.to_csv(out / 'scores.csv', index=False)
    w = sc.merge(runs[['run_id', 'dims', 'silhouette_cosine', 'largest_cluster_pct', 'noise_pct', 'actual_k', 'fit_seconds']], on='run_id')
    summ = w.groupby(['variant', 'dims', 'representation', 'method', 'k', 'scoring', 'label'], dropna=False).agg(
        n_seeds=('seed', 'nunique'), AMI_mean=('AMI', 'mean'), AMI_sd=('AMI', 'std'), NMI_mean=('NMI', 'mean'), ARI_mean=('ARI', 'mean'), ARI_sd=('ARI', 'std'),
        silhouette_cosine=('silhouette_cosine', 'mean'), largest_cluster_pct_max=('largest_cluster_pct', 'max'), noise_pct=('noise_pct', 'mean'),
        actual_k=('actual_k', 'mean'), fit_seconds=('fit_seconds', 'mean'), n_evaluated=('n_evaluated', 'first'), classes=('classes', 'first')).reset_index()
    summ.to_csv(out / 'summary.csv', index=False)
    return summ


PILOT_MAIN = ['clip', 'siglip', 'dinov2_s_518_L1', 'dinov2_b_518_L1', 'dinov2_l_224_L1']
PILOT_RQ3 = [f'dinov2_{m}_{s}_L{l}' for m in 'sbl' for s in (224, 518) for l in (1, 2, 4) if not (m == 'l' and s == 518)]


def pilot_fit():
    out = OUT / 'pilot'; out.mkdir(parents=True, exist_ok=True)
    variants = [(v, FULL_METHODS, REPS) for v in PILOT_MAIN] + [(v, CORE, ['centered_l2']) for v in PILOT_RQ3 if v not in PILOT_MAIN]
    fit_grid('pilot', variants, out, [10])


def pilot_score():
    M = pd.read_csv(P2 / 'inputs/evaluation_masks.csv', dtype=str, keep_default_na=False)
    s = score(OUT / 'pilot', M, ['hs1', 'hs2', 'hs3'])
    # parity with P2/P3 (same code path): CLIP/SigLIP centered k=10
    p2 = pd.read_csv(ROOT / 'reports/week6/P3_PREPROCESSING_RESULTS.csv')
    chk = []
    for enc in ('clip', 'siglip'):
        for meth in CORE:
            for rep in REPS:
                a = s.query("variant==@enc and representation==@rep and method==@meth and label=='hs3'").AMI_mean.iloc[0]
                b = p2.query("encoder==@enc and representation==@rep and method==@meth and requested_k==10").HS3_AMI_mean.iloc[0]
                chk.append({'encoder': enc, 'representation': rep, 'method': meth, 'AMI_here': a, 'AMI_P3': b, 'absdiff': abs(a - b)})
    c = pd.DataFrame(chk); c.to_csv(OUT / 'pilot/parity_with_week6.csv', index=False)
    print('parity rows', len(c), 'max |diff|', c.absdiff.max())


def probe():
    M = pd.read_csv(P2 / 'inputs/evaluation_masks.csv', dtype=str, keep_default_na=False)
    m = M.hs3_valid.eq('True').to_numpy(); y = M.hs3.to_numpy()
    vc = pd.Series(y[m]).value_counts(); keep = m & np.isin(y, vc[vc >= 10].index)
    yk = y[keep]; rows = []
    runs = pd.read_csv(OUT / 'pilot/runs.csv')
    for name in PILOT_MAIN + [v for v in PILOT_RQ3 if v not in PILOT_MAIN]:
        X, _ = emb('pilot', name); Z = transform(X, 'centered_l2')
        t = time.perf_counter()
        pr = cross_val_predict(LogisticRegression(C=0.5, max_iter=2000), Z[keep], yk, cv=StratifiedKFold(5, shuffle=True, random_state=0))
        km = [AMI(yk, np.load(OUT / 'pilot/assignments' / f'{name}__centered_l2__kmeans__k10__s{s}.npy')[keep]) for s in SEEDS]
        rows.append({'variant': name, 'rows': int(keep.sum()), 'classes': int(len(set(yk))), 'probe_accuracy': accuracy_score(yk, pr), 'probe_balanced_accuracy': balanced_accuracy_score(yk, pr),
                     'probe_AMI': AMI(yk, pr), 'kmeans_k10_AMI_same_rows_mean': float(np.mean(km)), 'seconds': time.perf_counter() - t,
                     'note': 'SUPERVISED reference (5-fold out-of-fold predictions); not an unsupervised result'})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'pilot/linear_probe.csv', index=False)


def dev():
    out = OUT / 'dev'; out.mkdir(parents=True, exist_ok=True)
    variants = [(v, ['kmeans', 'agglomerative_average_cosine', 'spectral_knn', 'gmm_diag', 'hdbscan'], ['centered_l2', 'pca50_l2']) for v in ('clip', 'siglip', 'dinov2_b_224_L1')]
    variants += [(v, ['kmeans', 'agglomerative_average_cosine', 'spectral_knn'], ['l2', 'pca100_l2']) for v in ('clip', 'siglip', 'dinov2_b_224_L1')]
    fit_grid('dev', variants, out, [10, 13], sil_sample=4000)
    D = pd.read_csv(ROOT / 'reports/full_dataset/benchmark/dev_ids.csv', dtype=str, keep_default_na=False)
    assert (D.partition == 'dev').all()
    s = score(out, D, ['hs1', 'hs2', 'hs3', 'family', 'name'])
    print(s.query("label=='hs3' and scoring!='assigned_only'")[['variant', 'representation', 'method', 'k', 'AMI_mean', 'ARI_mean', 'largest_cluster_pct_max']].round(4).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['pilot_fit', 'pilot_score', 'probe', 'dev']); a = ap.parse_args()
    with threadpool_limits(limits=4):
        globals()[a.stage]()
