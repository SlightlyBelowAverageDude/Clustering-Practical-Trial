"""Week 6 / P3 experiments. Stages:
  robust_fit    - 80% subset refits (phase-2 subsets) for kmeans / agglomerative / spectral over the k grid. No labels.
  robust_select - blind k (silhouette, eigengap) per subset. No labels.
  robust_score  - HS scoring + aggregation -> P3_ROBUSTNESS_RESULTS.csv, P3_BLIND_K_ROBUSTNESS.csv
  preproc       - representation ablation (5 phase-2 representations x 3 methods x k 5/10/20) -> P3_PREPROCESSING_RESULTS.csv
  amiari        - AMI vs ARI decomposition on saved P2 assignments
Reuses P2 helpers (reports/week6/source/p2_benchmark.py) and phase-2 artifacts. Writes only under reports/week6/.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '4'); os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse, json, sys, time, itertools, platform, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI,
                             normalized_mutual_info_score as NMI, pairwise_distances)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2_benchmark import make, sizes, sil, eigengap, sha, array, csv, write_json, now, P2, ROOT  # noqa: E402

W6 = ROOT / 'reports/week6'
OUT = W6 / 'p3_experiments'
PROTO = json.loads((W6 / 'P3_EXPERIMENT_PROTOCOL.json').read_text())
KS = PROTO['task_3_1_robustness']['k_grid']
SEEDS = [0, 1, 2, 3, 4]
SUBSETS = [2026, 2027, 2028, 2029, 2030]
ENCODERS = ['clip', 'siglip']
METHODS = ['kmeans', 'agglomerative_average_cosine', 'spectral_knn']
LABELS = ['hs1', 'hs2', 'hs3']
P2METHOD = {'kmeans': 'kmeans', 'agglomerative_average_cosine': 'average_cosine'}


def masks_df():
    m = pd.read_csv(P2 / 'inputs/evaluation_masks.csv', dtype=str, keep_default_na=False)
    assert sha(P2 / 'inputs/evaluation_masks.csv').startswith('d333e523')
    return m


def raw(enc):
    if enc == 'clip':
        man = json.loads((ROOT / 'results/pilot/embedding_manifest.json').read_text())
    else:
        man = json.loads((P2 / 'siglip/embedding_manifest.json').read_text())
    assert sha(man['array_path']) == man['array_sha256']
    return np.load(man['array_path'])


def score_rows(pred, lab_df):
    out = {}
    for lab in LABELS:
        m = lab_df[lab + '_valid'].eq('True').to_numpy() if lab + '_valid' in lab_df else lab_df[lab].ne('').to_numpy()
        y, p = lab_df.loc[m, lab].to_numpy(), pred[m]
        out.update({f'{lab.upper()}_AMI': AMI(y, p), f'{lab.upper()}_NMI': NMI(y, p), f'{lab.upper()}_ARI': ARI(y, p), f'{lab.upper()}_n': int(m.sum())})
    return out


def env(stage):
    write_json(OUT / f'environment_{stage}.json', {'utc': now(), 'python': sys.version, 'platform': platform.platform(), 'threads': 4,
               'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'pandas', 'threadpoolctl']},
               'source_sha256': {p.name: sha(p) for p in HERE.glob('*.py')}, 'p3_protocol_sha256': sha(W6 / 'P3_EXPERIMENT_PROTOCOL.json')})


# ---------------------------------------------------------------- 3.1 robustness
def robust_fit():
    D = OUT / 'robustness'; (D / 'assignments').mkdir(parents=True, exist_ok=True); env('robust_fit')
    runs, eig, parity, sizes_rows = [], [], [], []
    for enc in ENCODERS:
        X = raw(enc)
        for ss in SUBSETS:
            sd = P2 / 'evaluation' / enc / 'subsets' / str(ss)
            rows = np.load(sd / 'rows.npy')
            assert np.array_equal(rows, np.sort(np.random.default_rng(ss).choice(2000, 1600, replace=False)))
            Z = np.load(sd / 'centered_l2.npy')
            Xs = X[rows]; Zr = normalize(Xs - Xs.mean(0))
            assert np.allclose(Z, Zr, atol=1e-5), 'subset representation mismatch'
            Dc = pairwise_distances(Z, metric='cosine'); np.maximum(Dc, 0, out=Dc); np.fill_diagonal(Dc, 0)
            ev, ncomp = eigengap(Z)
            for k in KS:
                eig.append({'encoder': enc, 'subset_seed': ss, 'k': k, 'gap': float(ev[k] - ev[k - 1]), 'graph_components': int(ncomp)})
            for method in METHODS:
                for k in KS:
                    for seed in (SEEDS if method != 'agglomerative_average_cosine' else [0]):
                        rid = f'{enc}__ss{ss}__{method}__k{k}__s{seed}'
                        t = time.perf_counter(); pred = make(method, k, seed).fit_predict(Z).astype(np.int64); sec = time.perf_counter() - t
                        p = D / 'assignments' / f'{rid}.npy'; array(p, pred)
                        st = sizes(pred)
                        runs.append({'run_id': rid, 'encoder': enc, 'subset_seed': ss, 'n_rows': len(rows), 'representation': 'centered_l2 (subset-refit)',
                                     'method': method, 'requested_k': k, 'seed': seed, 'fit_seconds': sec, 'silhouette_cosine': sil(Dc, pred),
                                     'assignment_path': str(p.relative_to(ROOT)), 'assignment_sha256': sha(p), **st})
                        for c, n in zip(*np.unique(pred, return_counts=True)):
                            sizes_rows.append({'run_id': rid, 'cluster': int(c), 'images': int(n)})
                        if k == 10 and method in P2METHOD:
                            old = np.load(sd / f'{P2METHOD[method]}_s{seed}.npy')
                            parity.append({'run_id': rid, 'phase2_file': str((sd / f'{P2METHOD[method]}_s{seed}.npy').relative_to(ROOT)), 'identical': bool(np.array_equal(old, pred)), 'ARI_vs_phase2': ARI(old, pred)})
                print(enc, ss, method, 'done', flush=True)
            csv(D / 'runs.csv', runs); csv(D / 'eigengap.csv', eig); csv(D / 'phase2_parity.csv', parity); csv(D / 'cluster_sizes.csv', sizes_rows)
    write_json(D / 'fit_complete.json', {'utc': now(), 'runs': len(runs), 'parity_identical': sum(p['identical'] for p in parity), 'parity_rows': len(parity)})


def robust_select():
    """Label-blind."""
    D = OUT / 'robustness'
    r = pd.read_csv(D / 'runs.csv'); e = pd.read_csv(D / 'eigengap.csv')
    cur = r.groupby(['encoder', 'subset_seed', 'method', 'requested_k']).silhouette_cosine.mean().reset_index()
    sel = []
    for (enc, ss, m), g in cur.groupby(['encoder', 'subset_seed', 'method']):
        best = g.silhouette_cosine.max()
        sel.append({'encoder': enc, 'subset_seed': ss, 'method': m, 'criterion': 'silhouette_cosine', 'selected_k': int(g[g.silhouette_cosine == best].requested_k.min()), 'criterion_value': best})
        if m == 'spectral_knn':
            ge = e[(e.encoder == enc) & (e.subset_seed == ss)]
            row = ge.loc[ge.gap.idxmax()]
            sel.append({'encoder': enc, 'subset_seed': ss, 'method': m, 'criterion': 'eigengap', 'selected_k': int(row.k), 'criterion_value': row.gap})
    sel = pd.DataFrame(sel); sel['labels_accessed'] = False; sel['selected_utc'] = now()
    csv(D / 'blind_k_selection.csv', sel); csv(D / 'silhouette_curves.csv', cur)
    print(sel.pivot_table(index=['encoder', 'method', 'criterion'], columns='subset_seed', values='selected_k').to_string())


def robust_score():
    D = OUT / 'robustness'
    M = masks_df()
    r = pd.read_csv(D / 'runs.csv')
    rows = []
    for x in r.itertuples():
        pred = np.load(ROOT / x.assignment_path); assert sha(ROOT / x.assignment_path) == x.assignment_sha256
        sub = M.iloc[np.load(P2 / 'evaluation' / x.encoder / 'subsets' / str(x.subset_seed) / 'rows.npy')].reset_index(drop=True)
        rows.append({'run_id': x.run_id, **score_rows(pred, sub)})
    sc = r.merge(pd.DataFrame(rows), on='run_id'); csv(D / 'scores.csv', sc)

    # partition stability at k=10, seed 0
    stab = []
    w6 = W6 / 'experiments/assignments'
    for enc in ENCODERS:
        for m in METHODS:
            full = np.load(w6 / f'{enc}__centered_l2__{m}__k10__s0.npy')
            preds = {}
            for ss in SUBSETS:
                rr = np.load(P2 / 'evaluation' / enc / 'subsets' / str(ss) / 'rows.npy')
                p = np.load(D / 'assignments' / f'{enc}__ss{ss}__{m}__k10__s0.npy'); preds[ss] = (rr, p)
                stab.append({'encoder': enc, 'method': m, 'comparison': 'subset_vs_full_restricted', 'subset_a': ss, 'subset_b': None, 'common_rows': len(rr), 'partition_ARI': ARI(full[rr], p)})
            for a, b in itertools.combinations(SUBSETS, 2):
                (ra, pa), (rb, pb) = preds[a], preds[b]
                common, ia, ib = np.intersect1d(ra, rb, return_indices=True)
                stab.append({'encoder': enc, 'method': m, 'comparison': 'subset_vs_subset_common_rows', 'subset_a': a, 'subset_b': b, 'common_rows': len(common), 'partition_ARI': ARI(pa[ia], pb[ib])})
    stab = pd.DataFrame(stab); csv(D / 'partition_stability.csv', stab)

    # robustness table at k=10
    k10 = sc[sc.requested_k == 10]
    agg = k10.groupby(['encoder', 'subset_seed', 'method']).agg(
        n_seeds=('seed', 'nunique'), HS3_AMI_mean=('HS3_AMI', 'mean'), HS3_AMI_sd=('HS3_AMI', 'std'), HS3_NMI_mean=('HS3_NMI', 'mean'),
        HS3_ARI_mean=('HS3_ARI', 'mean'), HS1_AMI_mean=('HS1_AMI', 'mean'), HS2_AMI_mean=('HS2_AMI', 'mean'), HS3_n=('HS3_n', 'first'),
        largest_cluster_pct_mean=('largest_cluster_pct', 'mean'), largest_cluster_pct_max=('largest_cluster_pct', 'max'),
        silhouette_cosine_mean=('silhouette_cosine', 'mean'), fit_seconds_mean=('fit_seconds', 'mean')).reset_index()
    # paired deltas vs kmeans within subset (same seed index; agglomerative vs each kmeans seed)
    d = []
    for m in ['spectral_knn', 'agglomerative_average_cosine']:
        for (enc, ss), g in k10.groupby(['encoder', 'subset_seed']):
            row = {'encoder': enc, 'subset_seed': ss, 'method': m}
            for metric in ['AMI', 'ARI']:
                km = g[g.method == 'kmeans'].set_index('seed')[f'HS3_{metric}']
                x = g[g.method == m].set_index('seed')[f'HS3_{metric}']
                dd = (x.iloc[0] - km) if m == 'agglomerative_average_cosine' else (x - km)
                row.update({f'delta_HS3_{metric}_vs_kmeans': dd.mean(), f'seeds_positive_{metric}': int((dd > 0).sum())})
            d.append(row)
    agg = agg.merge(pd.DataFrame(d), on=['encoder', 'subset_seed', 'method'], how='left')
    fs = stab[stab.comparison == 'subset_vs_full_restricted'][['encoder', 'method', 'subset_a', 'partition_ARI']].rename(columns={'subset_a': 'subset_seed', 'partition_ARI': 'ARI_vs_full_pilot_partition_s0'})
    agg = agg.merge(fs, on=['encoder', 'method', 'subset_seed'], how='left')
    agg.insert(0, 'dataset', 'mimo_pilot_2000 / 80% subsets (phase-2 seeds)'); agg['k'] = 10
    agg.to_csv(W6 / 'P3_ROBUSTNESS_RESULTS.csv', index=False)

    # blind-k robustness
    sel = pd.read_csv(D / 'blind_k_selection.csv')
    cur = sc.groupby(['encoder', 'subset_seed', 'method', 'requested_k']).agg(HS3_AMI=('HS3_AMI', 'mean'), HS3_ARI=('HS3_ARI', 'mean'), largest=('largest_cluster_pct', 'mean')).reset_index()
    out = []
    for s in sel.itertuples():
        g = cur[(cur.encoder == s.encoder) & (cur.subset_seed == s.subset_seed) & (cur.method == s.method)].set_index('requested_k')
        ok = int(g.HS3_AMI.idxmax())
        out.append({'encoder': s.encoder, 'subset_seed': s.subset_seed, 'method': s.method, 'criterion': s.criterion, 'selected_k': s.selected_k,
                    'labels_used_for_selection': False, 'HS3_AMI_selected': g.HS3_AMI[s.selected_k], 'HS3_ARI_selected': g.HS3_ARI[s.selected_k],
                    'largest_cluster_pct_selected': g.largest[s.selected_k], 'HS3_AMI_fixed_k10': g.HS3_AMI[10],
                    'delta_vs_fixed_k10': g.HS3_AMI[s.selected_k] - g.HS3_AMI[10],
                    'oracle_k_NOT_BLIND': ok, 'oracle_HS3_AMI_NOT_BLIND': g.HS3_AMI[ok], 'gap_to_oracle': g.HS3_AMI[s.selected_k] - g.HS3_AMI[ok]})
    pd.DataFrame(out).to_csv(W6 / 'P3_BLIND_K_ROBUSTNESS.csv', index=False)
    pd.set_option('display.width', 250)
    print(agg[['encoder', 'subset_seed', 'method', 'HS3_AMI_mean', 'HS3_AMI_sd', 'HS3_ARI_mean', 'largest_cluster_pct_max', 'delta_HS3_AMI_vs_kmeans', 'seeds_positive_AMI', 'delta_HS3_ARI_vs_kmeans', 'ARI_vs_full_pilot_partition_s0']].round(4).to_string(index=False))
    print(pd.DataFrame(out).round(4).to_string(index=False))
    print(stab.groupby(['encoder', 'method', 'comparison']).partition_ARI.agg(['mean', 'min']).round(3))


# ---------------------------------------------------------------- 3.3 preprocessing
def preproc():
    D = OUT / 'preprocessing'; (D / 'assignments').mkdir(parents=True, exist_ok=True); env('preproc')
    M = masks_df()
    reps = PROTO['task_3_3_preprocessing']['representations']; ks = PROTO['task_3_3_preprocessing']['ks']
    runs, parity = [], []
    for enc in ENCODERS:
        for rep in reps:
            meta = json.loads((P2 / enc / 'representations' / f'{rep}.json').read_text())
            path = P2 / enc / 'representations' / f'{rep}.npy'; assert sha(path) == meta['array_sha256']
            Z = np.load(path)
            Dc = pairwise_distances(Z, metric='cosine'); np.maximum(Dc, 0, out=Dc); np.fill_diagonal(Dc, 0)
            for method in METHODS:
                for k in ks:
                    for seed in (SEEDS if method != 'agglomerative_average_cosine' else [0]):
                        rid = f'{enc}__{rep}__{method}__k{k}__s{seed}'
                        t = time.perf_counter(); pred = make(method, k, seed).fit_predict(Z).astype(np.int64); sec = time.perf_counter() - t
                        p = D / 'assignments' / f'{rid}.npy'; array(p, pred)
                        runs.append({'run_id': rid, 'encoder': enc, 'representation': rep, 'representation_sha256': meta['array_sha256'], 'dims': Z.shape[1],
                                     'method': method, 'requested_k': k, 'seed': seed, 'fit_seconds': sec, 'silhouette_cosine': sil(Dc, pred),
                                     'assignment_path': str(p.relative_to(ROOT)), **sizes(pred), **score_rows(pred, M)})
                        ref = None
                        if method in P2METHOD:
                            ref = P2 / enc / 'predictions' / f'{enc}__{rep}__{P2METHOD[method]}__k{k}__s{seed}.npy'
                        elif rep in ('centered_l2', 'pca50_l2'):
                            ref = W6 / 'experiments/assignments' / f'{enc}__{rep}__{method}__k{k}__s{seed}.npy'
                        if ref is not None and ref.exists():
                            parity.append({'run_id': rid, 'reference': str(ref.relative_to(ROOT)), 'identical': bool(np.array_equal(np.load(ref), pred))})
                print(enc, rep, method, flush=True)
            csv(D / 'runs.csv', runs); csv(D / 'parity.csv', parity)
    r = pd.DataFrame(runs)
    agg = r.groupby(['encoder', 'representation', 'method', 'requested_k']).agg(
        n_seeds=('seed', 'nunique'), dims=('dims', 'first'), HS3_AMI_mean=('HS3_AMI', 'mean'), HS3_AMI_sd=('HS3_AMI', 'std'),
        HS3_NMI_mean=('HS3_NMI', 'mean'), HS3_ARI_mean=('HS3_ARI', 'mean'), HS3_ARI_sd=('HS3_ARI', 'std'),
        HS1_AMI_mean=('HS1_AMI', 'mean'), HS2_AMI_mean=('HS2_AMI', 'mean'), HS3_n=('HS3_n', 'first'),
        silhouette_cosine_mean=('silhouette_cosine', 'mean'), actual_k_mean=('actual_k', 'mean'),
        largest_cluster_pct_mean=('largest_cluster_pct', 'mean'), largest_cluster_pct_max=('largest_cluster_pct', 'max'),
        fit_seconds_mean=('fit_seconds', 'mean')).reset_index()
    agg.insert(0, 'dataset', 'mimo_pilot_2000'); agg['representation_source'] = 'results/phase2/<enc>/representations'
    agg.to_csv(W6 / 'P3_PREPROCESSING_RESULTS.csv', index=False)
    par = pd.DataFrame(parity); print('parity', par.identical.sum(), '/', len(par))
    pd.set_option('display.width', 250)
    print(agg[agg.requested_k == 10][['encoder', 'representation', 'method', 'HS3_AMI_mean', 'HS3_AMI_sd', 'HS3_ARI_mean', 'largest_cluster_pct_max', 'silhouette_cosine_mean']].round(4).to_string(index=False))


# ---------------------------------------------------------------- 3.2 AMI vs ARI
def amiari():
    D = OUT / 'ami_vs_ari'; D.mkdir(parents=True, exist_ok=True)
    M = masks_df()
    sample = pd.read_csv(ROOT / 'data/mimo_pilot/sample_pilot.csv', dtype=str, keep_default_na=False)
    assert sample.image_id.tolist() == M.image_id.tolist()
    vocab = pd.read_csv(ROOT / 'data/mimo_pilot/hs_vocab.csv', dtype=str, keep_default_na=False)
    m = M.hs3_valid.eq('True').to_numpy(); y = M.loc[m, 'hs3'].to_numpy()
    A = W6 / 'experiments/assignments'
    configs = [(enc, meth) for enc in ENCODERS for meth in ['spectral_knn', 'agglomerative_average_cosine', 'kmeans']]
    summ, comp, clus = [], [], []
    for enc, meth in configs:
        pred_all = np.load(A / f'{enc}__centered_l2__{meth}__k10__s0.npy'); p = pred_all[m]
        C = contingency_matrix(y, p)  # classes x clusters
        classes = np.unique(y)
        n = C.sum(); a = C.sum(1); b = C.sum(0)
        tp = (C * (C - 1) / 2).sum(); same_cluster = (b * (b - 1) / 2).sum(); same_class = (a * (a - 1) / 2).sum(); total = n * (n - 1) / 2
        # mutual information contribution per cluster
        Pij = C / n; Pi = a / n; Pj = b / n
        with np.errstate(divide='ignore', invalid='ignore'):
            mi_terms = np.where(C > 0, Pij * np.log(Pij / np.outer(Pi, Pj)), 0)
        purity = C.max(0).sum() / n
        largest = b.max()
        lg = int(np.argmax(b))
        summ.append({'encoder': enc, 'method': meth, 'seed': 0, 'k': 10, 'n_labelled': int(n), 'HS3_AMI': AMI(y, p), 'HS3_ARI': ARI(y, p), 'HS3_NMI': NMI(y, p),
                     'purity_descriptive': purity, 'inverse_purity_descriptive': C.max(1).sum() / n,
                     'pair_precision': tp / same_cluster, 'pair_recall': tp / same_class,
                     'pairs_same_cluster': same_cluster, 'pairs_same_class': same_class, 'pairs_total': total,
                     'largest_cluster_labelled_n': int(largest), 'largest_cluster_labelled_pct': 100 * largest / n,
                     'share_of_same_cluster_pairs_in_largest': (largest * (largest - 1) / 2) / same_cluster,
                     'largest_cluster_purity': C[:, lg].max() / largest, 'largest_cluster_top_class': str(classes[C[:, lg].argmax()]),
                     'largest_cluster_n_classes_ge5pct': int((C[:, lg] / largest >= 0.05).sum()),
                     'MI_nats': mi_terms.sum(), 'MI_share_from_largest_cluster': mi_terms[:, lg].sum() / mi_terms.sum(),
                     'cluster_entropy_nats': -(Pj[Pj > 0] * np.log(Pj[Pj > 0])).sum(),
                     'classes_split_ge3_clusters_with_ge10pct': int(((C / a[:, None]) >= 0.10).sum(1).__ge__(3).sum()),
                     'classes_with_ge70pct_in_one_cluster': int(((C / a[:, None]).max(1) >= 0.70).sum())})
        for j in range(C.shape[1]):
            col = C[:, j]; order = np.argsort(-col)
            clus.append({'encoder': enc, 'method': meth, 'cluster': j, 'size_all_rows': int((pred_all == np.unique(p)[j]).sum()), 'labelled': int(col.sum()),
                         'purity': col.max() / col.sum(), 'top1_class': str(classes[order[0]]), 'top1_share': col[order[0]] / col.sum(),
                         'top2_class': str(classes[order[1]]) if col[order[1]] else '', 'top2_share': col[order[1]] / col.sum(),
                         'top3_class': str(classes[order[2]]) if col[order[2]] else '', 'top3_share': col[order[2]] / col.sum(),
                         'n_classes_present': int((col > 0).sum()), 'MI_contribution_nats': mi_terms[:, j].sum()})
            for i in np.nonzero(col)[0]:
                comp.append({'encoder': enc, 'method': meth, 'cluster': j, 'hs3': str(classes[i]), 'count': int(col[i]), 'share_of_cluster': col[i] / col.sum(), 'share_of_class': col[i] / a[i]})
        np.save(D / f'contingency_{enc}_{meth}.npy', C); pd.Series(classes).to_csv(D / f'contingency_{enc}_{meth}_classes.csv', index=False)
    summ = pd.DataFrame(summ); csv(D / 'summary.csv', summ); csv(D / 'cluster_profiles.csv', clus); csv(D / 'cluster_composition.csv', comp)

    # split / merge tables for SigLIP spectral vs agglomerative
    cp = pd.DataFrame(comp)
    split = cp[cp.share_of_class >= 0.10].groupby(['encoder', 'method', 'hs3']).agg(clusters_with_ge10pct=('cluster', 'nunique'), max_share=('share_of_class', 'max')).reset_index()
    csv(D / 'class_split.csv', split)

    # mixed-cluster examples: rule fixed in the protocol
    cl = pd.DataFrame(clus)
    ex = cl[(cl.encoder == 'siglip') & (cl.method == 'spectral_knn') & (cl.top1_share < 0.60)].sort_values('labelled', ascending=False).head(2)
    pred_all = np.load(A / 'siglip__centered_l2__spectral_knn__k10__s0.npy')
    labels_sorted = np.unique(pred_all)
    examples = []
    rng = np.random.default_rng(20261116)
    for r in ex.itertuples():
        cid = labels_sorted[r.cluster]
        rows = np.nonzero((pred_all == cid) & m)[0]
        for cls in [r.top1_class, r.top2_class, r.top3_class]:
            rr = rows[M.hs3.to_numpy()[rows] == cls]
            pick = np.sort(rng.choice(rr, size=min(4, len(rr)), replace=False))
            for i in pick:
                s = sample.iloc[i]
                examples.append({'cluster': int(cid), 'cluster_labelled': r.labelled, 'cluster_size_all_rows': r.size_all_rows, 'row': int(i), 'hs3': cls,
                                 'hs3_label': s.hs_label, 'title': s.title, 'instrument_names': s.instrument_names, 'museum': s.museum,
                                 'rights': s.rights, 'record_id': s.id, 'image_file': 'data/mimo_pilot/' + s.file})
    csv(D / 'mixed_cluster_examples.csv', examples)
    hs_names = {}
    for code in set(cp.hs3):
        hit = sample[sample.hs3.eq(code)].hs_label
        hs_names[code] = hit.mode().iloc[0] if len(hit) else ''
    write_json(D / 'hs3_names_from_sample.json', hs_names)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(summ.round(4).T.to_string())
    print(cl[(cl.encoder == 'siglip') & cl.method.isin(['spectral_knn', 'agglomerative_average_cosine'])].sort_values(['method', 'labelled'], ascending=[True, False]).round(3).to_string(index=False))
    print(pd.DataFrame(examples)[['cluster', 'hs3', 'title', 'museum']].to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['robust_fit', 'robust_select', 'robust_score', 'preproc', 'amiari']); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        globals()[a.stage]()
