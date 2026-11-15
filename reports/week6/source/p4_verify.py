"""Week 6 / P4.1 independent verification.

Recomputes every presentation number from saved assignment .npy files, saved representations and the phase-2
evaluation masks, using only numpy/sklearn metric calls (no P2/P3 aggregation code), and compares with the
reported CSVs. Writes reports/week6/p4_validation/{checks.csv, presentation_claims.json, artifact_hashes.json}.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import hashlib, json, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, normalized_mutual_info_score as NMI, silhouette_score
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parents[3]
W6 = ROOT / 'reports/week6'; P2 = ROOT / 'results/phase2'; OUT = W6 / 'p4_validation'; OUT.mkdir(exist_ok=True)
A2 = W6 / 'experiments/assignments'; A3P = W6 / 'p3_experiments/preprocessing/assignments'; A3R = W6 / 'p3_experiments/robustness/assignments'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
M = pd.read_csv(P2 / 'inputs/evaluation_masks.csv', dtype=str, keep_default_na=False)
assert sha(P2 / 'inputs/evaluation_masks.csv') == json.loads((P2 / 'inputs/input_identity.json').read_text())['masks_sha256']
mask = {l: M[l + '_valid'].eq('True').to_numpy() for l in ['hs1', 'hs2', 'hs3']}
assert {l: int(v.sum()) for l, v in mask.items()} == {'hs1': 1565, 'hs2': 1563, 'hs3': 1549}
Y = {l: M[l].to_numpy() for l in mask}
checks, claims = [], {}
TOL = 1e-9


def chk(name, recomputed, reported, tol=TOL, note=''):
    ok = (recomputed == reported) if isinstance(recomputed, (str, list, tuple)) else (abs(float(recomputed) - float(reported)) <= tol)
    checks.append({'check': name, 'recomputed': recomputed, 'reported': reported, 'tolerance': tol, 'pass': bool(ok), 'note': note})
    return recomputed


def score(p, lab='hs3', fn=AMI):
    return fn(Y[lab][mask[lab]], p[mask[lab]])


def seeds(enc, rep, meth, k, base=A2):
    ss = [0] if meth == 'agglomerative_average_cosine' else [0, 1, 2, 3, 4]
    return [np.load(base / f'{enc}__{rep}__{meth}__k{k}__s{s}.npy') for s in ss]


mc = pd.read_csv(W6 / 'P2_METHOD_COMPARISON.csv'); mc = mc[mc.scoring == 'all_rows']
METH = ['kmeans', 'agglomerative_average_cosine', 'gmm_diag', 'spectral_knn']

# 1-2. k=10 headline table (both encoders, centered_l2), AMI mean/sd and ARI mean
for enc in ['clip', 'siglip']:
    for m in METH:
        P = seeds(enc, 'centered_l2', m, 10)
        a = [score(p) for p in P]; r = [score(p, fn=ARI) for p in P]
        row = mc[(mc.encoder == enc) & (mc.representation == 'centered_l2') & (mc.method == m) & (mc.requested_k == 10)].iloc[0]
        chk(f'P2 {enc} {m} k10 HS3 AMI mean', np.mean(a), row.HS3_AMI_mean)
        chk(f'P2 {enc} {m} k10 HS3 ARI mean', np.mean(r), row.HS3_ARI_mean)
        if len(a) > 1: chk(f'P2 {enc} {m} k10 HS3 AMI sd', np.std(a, ddof=1), row.HS3_AMI_sd)
        lp = max(100 * np.bincount(p).max() / len(p) for p in P)
        chk(f'P2 {enc} {m} k10 largest cluster pct max', lp, row.largest_cluster_pct_max)
        claims[f'{enc}_{m}_k10'] = {'AMI': round(np.mean(a), 3), 'AMI_sd': round(np.std(a, ddof=1), 3) if len(a) > 1 else None, 'ARI': round(np.mean(r), 3), 'largest_pct': round(lp, 1)}
# HDBSCAN noise-as-cluster mcs20
for enc in ['clip', 'siglip']:
    p = np.load(A2 / f'{enc}__centered_l2__hdbscan__mcs20__s0.npy')
    q = p.copy()
    claims[f'{enc}_hdbscan20'] = {'AMI_noise_as_cluster': round(score(q), 3), 'noise_pct': round(100 * (p == -1).mean(), 1)}
    row = pd.read_csv(W6 / 'P2_METHOD_COMPARISON.csv').query("encoder==@enc and representation=='centered_l2' and method=='hdbscan' and min_cluster_size==20 and scoring=='noise_as_cluster'").iloc[0]
    chk(f'P2 {enc} hdbscan20 noise-as-cluster AMI', score(q), row.HS3_AMI_mean)
    chk(f'P2 {enc} hdbscan20 noise pct', 100 * (p == -1).mean(), row.noise_pct_mean)
hn = [100 * (np.load(A2 / f'{e}__{r}__hdbscan__mcs{m}__s0.npy') == -1).mean() for e in ['clip', 'siglip'] for r in ['centered_l2', 'pca50_l2'] for m in [20, 50]]
claims['hdbscan_noise_range'] = [round(min(hn), 1), round(max(hn), 1)]

# 3. SigLIP vs CLIP: count of method x k x representation cells (k-requiring methods)
wins, cells = 0, 0
for rep in ['centered_l2', 'pca50_l2']:
    for m in METH:
        for k in [2, 3, 5, 8, 10, 15, 20]:
            s = np.mean([score(p) for p in seeds('siglip', rep, m, k)]); c = np.mean([score(p) for p in seeds('clip', rep, m, k)])
            wins += s > c; cells += 1
chk('SigLIP > CLIP cells', wins, 51); chk('SigLIP vs CLIP total cells', cells, 56)
claims['siglip_vs_clip'] = {'wins': int(wins), 'cells': cells}

# 4. Blind k (P2): recompute cosine silhouettes from saved representations and reselect k
sel = pd.read_csv(W6 / 'experiments/blind_k_selection.csv'); sel = sel[sel.criterion == 'silhouette_cosine']
bk = pd.read_csv(W6 / 'P2_BLIND_K_RESULTS.csv')
for enc in ['clip', 'siglip']:
    for rep in ['centered_l2', 'pca50_l2']:
        Z = np.load(P2 / enc / 'representations' / f'{rep}.npy'); D = pairwise_distances(Z, metric='cosine'); np.maximum(D, 0, out=D); np.fill_diagonal(D, 0)
        for m in METH:
            curve = {k: np.mean([silhouette_score(D, p, metric='precomputed') for p in seeds(enc, rep, m, k)]) for k in [2, 3, 5, 8, 10, 15, 20]}
            best = max(curve.values()); ksel = min(k for k, v in curve.items() if v == best)
            rep_k = int(sel[(sel.encoder == enc) & (sel.representation == rep) & (sel.method == m)].selected_k.iloc[0])
            chk(f'P2 blind-k silhouette {enc} {rep} {m} selected k', ksel, rep_k, tol=0)
            ami_sel = np.mean([score(p) for p in seeds(enc, rep, m, ksel)])
            rr = bk[(bk.encoder == enc) & (bk.representation == rep) & (bk.method == m) & (bk.criterion == 'silhouette_cosine')].iloc[0]
            chk(f'P2 blind-k {enc} {rep} {m} HS3 AMI at selected k', ami_sel, rr.HS3_AMI_selected)
            if rep == 'centered_l2':
                claims[f'blindk_{enc}_{m}'] = {'k': ksel, 'AMI': round(ami_sel, 3), 'AMI_k10': round(np.mean([score(p) for p in seeds(enc, rep, m, 10)]), 3)}
        print('blind-k verified', enc, rep, flush=True)

# 5. AMI vs ARI decomposition (SigLIP centered k10 seed 0)
for m in ['spectral_knn', 'agglomerative_average_cosine']:
    p = np.load(A2 / f'siglip__centered_l2__{m}__k10__s0.npy'); y, q = Y['hs3'][mask['hs3']], p[mask['hs3']]
    C = pd.crosstab(q, y).to_numpy(); n = C.sum(); b = C.sum(1); a = C.sum(0)
    tp = (C * (C - 1) / 2).sum(); sc_ = (b * (b - 1) / 2).sum(); scl = (a * (a - 1) / 2).sum()
    big = b.argmax()
    s = pd.read_csv(W6 / 'p3_experiments/ami_vs_ari/summary.csv').query("encoder=='siglip' and method==@m").iloc[0]
    chk(f'AMIvsARI {m} pair precision', tp / sc_, s.pair_precision); chk(f'AMIvsARI {m} pair recall', tp / scl, s.pair_recall)
    chk(f'AMIvsARI {m} share pairs in largest', b[big] * (b[big] - 1) / 2 / sc_, s.share_of_same_cluster_pairs_in_largest)
    chk(f'AMIvsARI {m} largest purity', C[big].max() / b[big], s.largest_cluster_purity)
    claims[f'amiari_{m}'] = {'AMI': round(AMI(y, q), 3), 'ARI': round(ARI(y, q), 3), 'pair_precision': round(tp / sc_, 2), 'pair_recall': round(tp / scl, 2),
                             'largest_labelled_pct': round(100 * b[big] / n, 1), 'largest_all_rows': int(np.bincount(p).max()),
                             'largest_all_pct': round(100 * np.bincount(p).max() / len(p), 1), 'share_pairs_largest': round(b[big] * (b[big] - 1) / 2 / sc_, 2),
                             'largest_purity': round(C[big].max() / b[big], 2), 'largest_n_classes': int((C[big] > 0).sum())}

# 6. Preprocessing results
pp = pd.read_csv(W6 / 'P3_PREPROCESSING_RESULTS.csv')
for r in pp.itertuples():
    P = seeds(r.encoder, r.representation, r.method, r.requested_k, base=A3P)
    chk(f'P3 preproc {r.encoder} {r.representation} {r.method} k{r.requested_k} AMI', np.mean([score(p) for p in P]), r.HS3_AMI_mean)
    chk(f'P3 preproc {r.encoder} {r.representation} {r.method} k{r.requested_k} largest max', max(100 * np.bincount(p).max() / len(p) for p in P), r.largest_cluster_pct_max)
    if r.requested_k == 10:
        claims[f'pre_{r.encoder}_{r.representation}_{r.method}'] = {'AMI_full': float(r.HS3_AMI_mean), 'AMI': round(r.HS3_AMI_mean, 3), 'ARI': round(r.HS3_ARI_mean, 3), 'largest_max': round(r.largest_cluster_pct_max, 1)}

# 7. Subset robustness
rob = pd.read_csv(W6 / 'P3_ROBUSTNESS_RESULTS.csv')
for r in rob.itertuples():
    rows = np.load(P2 / 'evaluation' / r.encoder / 'subsets' / str(r.subset_seed) / 'rows.npy')
    sub_m = mask['hs3'][rows]; ysub = Y['hs3'][rows][sub_m]
    ss = [0] if r.method == 'agglomerative_average_cosine' else [0, 1, 2, 3, 4]
    v = [AMI(ysub, np.load(A3R / f'{r.encoder}__ss{r.subset_seed}__{r.method}__k10__s{s}.npy')[sub_m]) for s in ss]
    chk(f'P3 subset {r.encoder} {r.subset_seed} {r.method} AMI', np.mean(v), r.HS3_AMI_mean)
for enc in ['clip', 'siglip']:
    g = rob[rob.encoder == enc].pivot_table(index='subset_seed', columns='method', values=['HS3_AMI_mean', 'HS3_ARI_mean'])
    claims[f'robust_{enc}'] = {'spec_gt_km_AMI': int((g['HS3_AMI_mean']['spectral_knn'] > g['HS3_AMI_mean']['kmeans']).sum()),
                               'spec_gt_agg_AMI': int((g['HS3_AMI_mean']['spectral_knn'] > g['HS3_AMI_mean']['agglomerative_average_cosine']).sum()),
                               'agg_gt_spec_ARI': int((g['HS3_ARI_mean']['agglomerative_average_cosine'] > g['HS3_ARI_mean']['spectral_knn']).sum()),
                               'spectral_AMI_mean': round(g['HS3_AMI_mean']['spectral_knn'].mean(), 3), 'spectral_AMI_min': round(g['HS3_AMI_mean']['spectral_knn'].min(), 3),
                               'spectral_AMI_max': round(g['HS3_AMI_mean']['spectral_knn'].max(), 3), 'kmeans_AMI_mean': round(g['HS3_AMI_mean']['kmeans'].mean(), 3)}
bkr = pd.read_csv(W6 / 'P3_BLIND_K_ROBUSTNESS.csv'); sp = bkr[bkr.method == 'spectral_knn'].pivot_table(index=['encoder', 'subset_seed'], columns='criterion', values=['selected_k', 'HS3_AMI_selected', 'gap_to_oracle'])
claims['robust_blindk_spectral'] = {'eigengap_ks': sorted(sp['selected_k']['eigengap'].astype(int).tolist()), 'silhouette_ks': sorted(sp['selected_k']['silhouette_cosine'].astype(int).tolist()),
                                    'eig_better': int((sp['HS3_AMI_selected']['eigengap'] > sp['HS3_AMI_selected']['silhouette_cosine'] + 1e-12).sum()),
                                    'equal': int((abs(sp['HS3_AMI_selected']['eigengap'] - sp['HS3_AMI_selected']['silhouette_cosine']) < 1e-12).sum()),
                                    'eig_gap_oracle': round(sp['gap_to_oracle']['eigengap'].mean(), 3), 'sil_gap_oracle': round(sp['gap_to_oracle']['silhouette_cosine'].mean(), 3)}
# recompute a subset blind-k selection from scratch (silhouettes on subset representations)
for enc in ['clip', 'siglip']:
    for ss in [2026, 2030]:
        Z = np.load(P2 / 'evaluation' / enc / 'subsets' / str(ss) / 'centered_l2.npy'); D = pairwise_distances(Z, metric='cosine'); np.maximum(D, 0, out=D); np.fill_diagonal(D, 0)
        cur = {k: np.mean([silhouette_score(D, np.load(A3R / f'{enc}__ss{ss}__spectral_knn__k{k}__s{s}.npy'), metric='precomputed') for s in range(5)]) for k in [2, 3, 5, 8, 10, 15, 20]}
        best = max(cur.values()); ksel = min(k for k, v in cur.items() if v == best)
        chk(f'P3 subset blind-k silhouette {enc} {ss} spectral', ksel, int(bkr.query("encoder==@enc and subset_seed==@ss and method=='spectral_knn' and criterion=='silhouette_cosine'").selected_k.iloc[0]), tol=0)

# 8. Precision sensitivity: do headline conclusions hold in the float64 run?
f64 = pd.read_csv(W6 / 'experiments_sensitivity_float64/P2_METHOD_COMPARISON.csv'); f64 = f64[(f64.scoring == 'all_rows') & (f64.representation == 'centered_l2') & (f64.requested_k == 10)]
for enc in ['clip', 'siglip']:
    g = f64[f64.encoder == enc].set_index('method').HS3_AMI_mean
    chk(f'float64 run: {enc} spectral > kmeans at k10', bool(g['spectral_knn'] > g['kmeans']), True, tol=0)
    gr = f64[f64.encoder == enc].set_index('method').HS3_ARI_mean
    chk(f'float64 run: {enc} agglomerative ARI > spectral ARI at k10', bool(gr['agglomerative_average_cosine'] > gr['spectral_knn']), True, tol=0)

# 9. Originals unchanged + figure provenance
snap = json.loads((P2 / 'inputs/original_snapshot.json').read_text())['files']
chk('original pilot snapshot files unchanged', sum(sha(ROOT / f) == h for f, h in snap.items()), len(snap), tol=0)
fv = json.loads((P2 / 'FINAL_VALIDATION.json').read_text())
chk('phase-2 sources unchanged', sum(sha(P2 / 'source' / n) == h for n, h in fv['source_sha256'].items()), len(fv['source_sha256']), tol=0)
chk('phase-2 reports unchanged', sum(sha(P2 / n) == h for n, h in fv['report_sha256'].items()), len(fv['report_sha256']), tol=0)
hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(list(W6.glob('*.csv')) + list(W6.glob('*.json')) + list(W6.glob('*.md')) + list((W6 / 'figures').glob('*.png')))}
(OUT / 'artifact_hashes.json').write_text(json.dumps(hashes, indent=2))
claims['dataset'] = {'images': 2000, 'museums': int(pd.read_csv(ROOT / 'data/mimo_pilot/sample_pilot.csv', usecols=['museum']).museum.nunique()), 'hs3_n': int(mask['hs3'].sum()), 'hs3_classes': int(len(np.unique(Y['hs3'][mask['hs3']])))}
chk('dataset museums', claims['dataset']['museums'], 216, tol=0); chk('dataset HS3 classes', claims['dataset']['hs3_classes'], 22, tol=0)

c = pd.DataFrame(checks); c.to_csv(OUT / 'checks.csv', index=False)
(OUT / 'presentation_claims.json').write_text(json.dumps(claims, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print('checks', len(c), 'passed', int(c['pass'].sum()), 'failed', int((~c['pass']).sum()))
print(c[~c['pass']].to_string())
print(json.dumps({k: v for k, v in claims.items() if not k.startswith('pre_')}, indent=1, default=str)[:4000])
