"""Week 6 / P2: aggregate saved runs/scores into the deliverable CSVs (mean +/- SD over seeds; no best-seed picking)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
W6 = ROOT / 'reports/week6'
EXP = W6 / (sys.argv[1] if len(sys.argv) > 1 else 'experiments')
DEST = W6 if EXP.name == 'experiments' else EXP  # sensitivity runs write their summaries inside their own folder
P2 = ROOT / 'results/phase2'

runs = pd.read_csv(EXP / 'runs.csv')
scores = pd.read_csv(EXP / 'scores.csv')
sel = pd.read_csv(EXP / 'blind_k_selection.csv')

# ---- consistency check vs phase 2 (identical rep + method + params + seeds) ----
chk = []
for enc in ['clip', 'siglip']:
    old = pd.read_csv(P2 / enc / 'scores.csv')
    old = old[old.representation.isin(['centered_l2', 'pca50_l2'])]
    for r in old.itertuples():
        m = 'agglomerative_average_cosine' if r.method == 'average_cosine' else 'kmeans'
        new = scores[(scores.encoder == enc) & (scores.representation == r.representation) & (scores.method == m)
                     & (scores.requested_k == r.requested_k) & (scores.seed == r.seed) & (scores.label == r.label)]
        if len(new):
            chk.append({'encoder': enc, 'run': r.run_id, 'label': r.label, 'AMI_phase2': r.AMI, 'AMI_week6': new.AMI.iloc[0],
                        'absdiff': abs(r.AMI - new.AMI.iloc[0])})
chk = pd.DataFrame(chk)
chk.to_csv(EXP / 'phase2_consistency_check.csv', index=False)
print('phase-2 overlap rows', len(chk), 'max |dAMI|', chk.absdiff.max())

# ---- wide per-run table ----
s = scores.copy()
s['scoring'] = s.scoring.replace({'all': 'all_rows'})
wide = s.pivot_table(index=['run_id', 'scoring'], columns='label', values=['AMI', 'NMI', 'ARI', 'coverage_pct'], aggfunc='first')
wide.columns = [f'{lab.upper()}_{m}' for m, lab in wide.columns]
wide = wide.reset_index()
per_run = runs.merge(wide, on='run_id')
per_run.to_csv(EXP / 'per_run_wide.csv', index=False)

# ---- method comparison: mean/SD across seeds ----
keys = ['encoder', 'representation', 'method', 'requested_k', 'min_cluster_size', 'scoring']
per_run['requested_k'] = per_run.requested_k.fillna(-1).astype(int)
per_run['min_cluster_size'] = per_run.min_cluster_size.fillna(-1).astype(int)
agg = {'n_seeds': ('seed', 'nunique'), 'actual_k_mean': ('actual_k', 'mean'),
       'largest_cluster_pct_mean': ('largest_cluster_pct', 'mean'), 'largest_cluster_pct_max': ('largest_cluster_pct', 'max'),
       'noise_pct_mean': ('noise_pct', 'mean'), 'silhouette_cosine_mean': ('silhouette_cosine', 'mean'),
       'silhouette_cosine_sd': ('silhouette_cosine', 'std'), 'fit_seconds_mean': ('fit_seconds', 'mean')}
for lab in ['HS1', 'HS2', 'HS3']:
    for m in ['AMI', 'NMI', 'ARI']:
        agg[f'{lab}_{m}_mean'] = (f'{lab}_{m}', 'mean')
    agg[f'{lab}_AMI_sd'] = (f'{lab}_AMI', 'std')
    agg[f'{lab}_coverage_pct'] = (f'{lab}_coverage_pct', 'mean')
mc = per_run.groupby(keys).agg(**agg).reset_index()
mc.insert(0, 'dataset', 'mimo_pilot_2000')
mc['n_hs3_evaluated'] = 1549
mc['status'] = np.where(mc.actual_k_mean < 2, 'degenerate', 'ok')
mc['fixed_k_anchor'] = mc.requested_k.eq(10)
mc['note'] = np.where(mc.method.eq('hdbscan') & mc.scoring.eq('assigned_only'),
                      'HDBSCAN scored on non-noise rows only (see coverage); NOT comparable to full-mask scores',
                      np.where(mc.method.eq('hdbscan'), 'HDBSCAN noise treated as one extra cluster; full mask', ''))
mc = mc.replace({-1: np.nan})
mc.to_csv(DEST / 'P2_METHOD_COMPARISON.csv', index=False)

# ---- blind-k results ----
full = mc[mc.scoring.eq('all_rows')]
rows = []
for r in sel.itertuples():
    g = full[(full.encoder == r.encoder) & (full.representation == r.representation) & (full.method == r.method)].set_index('requested_k')
    at, fixed = g.loc[r.selected_k], g.loc[10]
    oracle_k = int(g.HS3_AMI_mean.idxmax())
    rows.append({'dataset': 'mimo_pilot_2000', 'encoder': r.encoder, 'representation': r.representation, 'method': r.method,
                 'criterion': r.criterion, 'criterion_value': r.criterion_value, 'selected_k': r.selected_k,
                 'labels_used_for_selection': False,
                 'hs1_classes': 5, 'hs2_classes': 11, 'hs3_classes': 22, 'hs3_classes_ge10': 13, 'n_hs3_evaluated': 1549,
                 'n_seeds': int(at.n_seeds), 'actual_k_mean': at.actual_k_mean, 'largest_cluster_pct_mean': at.largest_cluster_pct_mean,
                 'silhouette_cosine_mean': at.silhouette_cosine_mean,
                 **{f'{l}_{m}_selected': at[f'{l}_{m}_mean'] for l in ['HS1', 'HS2', 'HS3'] for m in ['AMI', 'NMI', 'ARI']},
                 'HS3_AMI_sd_selected': at.HS3_AMI_sd,
                 'HS3_AMI_fixed_k10': fixed.HS3_AMI_mean, 'HS3_AMI_sd_fixed_k10': fixed.HS3_AMI_sd,
                 'delta_HS3_AMI_vs_fixed_k10': at.HS3_AMI_mean - fixed.HS3_AMI_mean,
                 'oracle_k_by_HS3_AMI_NOT_BLIND': oracle_k, 'oracle_HS3_AMI_NOT_BLIND': g.loc[oracle_k].HS3_AMI_mean,
                 'blind_minus_oracle_HS3_AMI': at.HS3_AMI_mean - g.loc[oracle_k].HS3_AMI_mean})
bk = pd.DataFrame(rows)
bk.to_csv(DEST / 'P2_BLIND_K_RESULTS.csv', index=False)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
v = mc[(mc.requested_k.eq(10) & mc.scoring.eq('all_rows')) | mc.scoring.eq('noise_as_cluster')]
print(v[['encoder', 'representation', 'method', 'requested_k', 'min_cluster_size', 'actual_k_mean', 'HS1_AMI_mean', 'HS2_AMI_mean', 'HS3_AMI_mean', 'HS3_AMI_sd', 'HS3_ARI_mean', 'HS3_NMI_mean', 'largest_cluster_pct_mean', 'noise_pct_mean', 'silhouette_cosine_mean', 'fit_seconds_mean']].round(4).to_string(index=False))
print(mc[mc.scoring.eq('assigned_only')][['encoder', 'representation', 'min_cluster_size', 'HS3_AMI_mean', 'HS3_coverage_pct', 'HS1_AMI_mean']].round(4).to_string(index=False))
print(bk[['encoder', 'representation', 'method', 'criterion', 'selected_k', 'HS3_AMI_selected', 'HS3_AMI_sd_selected', 'HS3_AMI_fixed_k10', 'delta_HS3_AMI_vs_fixed_k10', 'oracle_k_by_HS3_AMI_NOT_BLIND', 'oracle_HS3_AMI_NOT_BLIND', 'largest_cluster_pct_mean']].round(4).to_string(index=False))

# ---- paired contrasts (same rows, same masks, same seed index) ----
from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI
hs3 = per_run[per_run.scoring.eq('all_rows')]
con = []
for (enc, rep, k), g in hs3[hs3.method.ne('hdbscan')].groupby(['encoder', 'representation', 'requested_k']):
    base = g[g.method.eq('kmeans')].set_index('seed').HS3_AMI
    for m, gm in g[g.method.ne('kmeans')].groupby('method'):
        x = gm.set_index('seed').HS3_AMI
        if m == 'agglomerative_average_cosine':  # deterministic: compare with every k-means seed
            d = x.iloc[0] - base
        else:
            d = x - base
        con.append({'contrast': f'{m} - kmeans', 'encoder': enc, 'representation': rep, 'k': k,
                    'delta_HS3_AMI_mean': d.mean(), 'delta_sd': d.std(), 'seed_pairs_positive': int((d > 0).sum()), 'seed_pairs': len(d)})
for (rep, m, k), g in hs3[hs3.method.ne('hdbscan')].groupby(['representation', 'method', 'requested_k']):
    a = g[g.encoder.eq('siglip')].set_index('seed').HS3_AMI; b = g[g.encoder.eq('clip')].set_index('seed').HS3_AMI
    d = a - b
    con.append({'contrast': 'siglip - clip', 'method': m, 'representation': rep, 'k': k,
                'delta_HS3_AMI_mean': d.mean(), 'delta_sd': d.std(), 'seed_pairs_positive': int((d > 0).sum()), 'seed_pairs': len(d)})
con = pd.DataFrame(con)
con.to_csv(EXP / 'paired_contrasts.csv', index=False)

# seed-to-seed partition stability (pairwise ARI of assignments)
stab = []
for (enc, rep, m, k), g in runs[runs.method.isin(['kmeans', 'gmm_diag', 'spectral_knn'])].groupby(['encoder', 'representation', 'method', 'requested_k']):
    P = [np.load(ROOT / p) for p in g.sort_values('seed').assignment_path]
    v = [ARI(P[i], P[j]) for i in range(len(P)) for j in range(i + 1, len(P))]
    stab.append({'encoder': enc, 'representation': rep, 'method': m, 'k': int(k), 'pairwise_seed_ARI_mean': np.mean(v), 'pairwise_seed_ARI_min': np.min(v)})
pd.DataFrame(stab).to_csv(EXP / 'seed_partition_stability.csv', index=False)

# HDBSCAN fairness control: k-means k=10 scored on exactly the rows HDBSCAN assigned
masks = pd.read_csv(P2 / 'inputs/evaluation_masks.csv', dtype=str, keep_default_na=False)
ctrl = []
for r in runs[runs.method.eq('hdbscan')].itertuples():
    hp = np.load(ROOT / r.assignment_path)
    km = runs[(runs.encoder == r.encoder) & (runs.representation == r.representation) & (runs.method == 'kmeans') & (runs.requested_k == 10)]
    for lab in ['hs1', 'hs3']:
        m = masks[lab + '_valid'].eq('True').to_numpy() & (hp != -1)
        y = masks.loc[m, lab].to_numpy()
        kv = [AMI(y, np.load(ROOT / p)[m]) for p in km.assignment_path]
        ctrl.append({'encoder': r.encoder, 'representation': r.representation, 'min_cluster_size': r.min_cluster_size, 'label': lab,
                     'rows_scored': int(m.sum()), 'label_classes_on_rows': len(np.unique(y)),
                     'HDBSCAN_assigned_only_AMI': AMI(y, hp[m]), 'kmeans_k10_same_rows_AMI_mean': np.mean(kv), 'kmeans_k10_same_rows_AMI_sd': np.std(kv, ddof=1)})
ctrl = pd.DataFrame(ctrl); ctrl.to_csv(EXP / 'hdbscan_same_rows_control.csv', index=False)
print(con.round(4).to_string(index=False)); print(pd.DataFrame(stab).round(3).to_string(index=False)); print(ctrl.round(4).to_string(index=False))
