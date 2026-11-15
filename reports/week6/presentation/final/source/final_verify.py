"""Final Week 6 deck: independent recomputation of every headline number from saved assignments and source tables.
Writes ../final_claims.json (the only number source for build_final_deck.js) and ../validation_checks.csv."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

ROOT = Path(__file__).resolve().parents[5]
FIN = ROOT / 'reports/week6/presentation/final'
DEV = ROOT / 'reports/dinov2/dev'
FD = ROOT / 'reports/full_dataset'
checks, C = [], {}


def chk(name, got, want, tol=1e-9):
    ok = (got == want) if isinstance(got, (str, bool)) else abs(float(got) - float(want)) <= tol
    checks.append({'check': name, 'recomputed': got, 'source_value': want, 'pass': bool(ok)})


# ---- dataset counts (from the build outputs, re-derived from the tables)
st = json.loads((FD / 'stats.json').read_text())
b = pd.read_csv(FD / 'benchmark/benchmark_objects.csv', dtype=str, keep_default_na=False)
c = pd.read_csv(FD / 'clean_records.csv', dtype=str, keep_default_na=False)
led = pd.read_csv(ROOT / 'data/mimo_full/download_ledger.csv', dtype=str, keep_default_na=False)
counts = {'records': len(led), 'valid_images': int((led.status == 'ok').sum()), 'cleaned': len(c), 'with_hs': int((c.hs1 != '').sum()),
          'benchmark': len(b), 'dev': int((b.partition == 'dev').sum()), 'heldout': int((b.partition == 'heldout').sum()),
          'pilot_overlap_excluded': int((b.partition == 'excluded_pilot_overlap').sum()), 'hs3_classes': b.hs3.nunique(), 'name_classes': b.name.nunique(),
          'families': b.family.nunique(), 'museums_clean': c.provider.nunique(), 'museums_benchmark': b.provider.nunique()}
for k, want in [('records', 63478), ('valid_images', 63473), ('cleaned', 59658), ('with_hs', 40694), ('benchmark', 27347), ('dev', 12000), ('heldout', 14696),
                ('pilot_overlap_excluded', 651), ('hs3_classes', 13), ('name_classes', 129), ('families', 6)]:
    chk(f'count {k}', counts[k], want, tol=0)
chk('stats.json cleaned', st['cleaned_photographs'], counts['cleaned'], tol=0); chk('stats.json benchmark', st['benchmark_rows'], counts['benchmark'], tol=0)
top5 = b.provider.value_counts().head(5).sum() / len(b); counts['top5_museum_share'] = round(float(top5), 3)
counts['dev_hs3'] = b[b.partition == 'dev'].hs3.value_counts().to_dict()
C['dataset'] = counts

# ---- dev isolation and configuration
dev = pd.read_csv(FD / 'benchmark/dev_ids.csv', dtype=str, keep_default_na=False)
ho = pd.read_csv(FD / 'benchmark/heldout_ids_DO_NOT_USE_FOR_SELECTION.csv', dtype=str)
chk('dev ∩ held-out', len(set(dev.id) & set(ho.id)), 0, tol=0)
chk('dev rows == embedded rows (CLIP/SigLIP/DINOv2)', len({json.loads((ROOT / f'reports/dinov2/embeddings/dev_{m}.json').read_text())['n'] for m in ('clip', 'siglip', 'dinov2_b_224')} | {len(dev)}), 1, tol=0)
dman = json.loads((ROOT / 'reports/dinov2/embeddings/dev_dinov2_b_224.json').read_text())
chk('dev DINOv2 input size', dman['descriptor']['size'], 224, tol=0); chk('dev DINOv2 model', dman['descriptor']['repo'], 'timm/vit_base_patch14_dinov2.lvd142m')
chk('no held-out embedding manifest', len(list((ROOT / 'reports/dinov2/embeddings').glob('heldout*'))), 0, tol=0)
y = dev.hs3.to_numpy(); chk('dev HS3 labelled rows', int((y != '').sum()), 12000, tol=0)

# ---- headline scores recomputed from assignments
summ = pd.read_csv(DEV / 'summary.csv')
V = {'clip': 'clip', 'siglip': 'siglip', 'dinov2': 'dinov2_b_224_L1'}
M = {'kmeans': 'kmeans', 'agglo': 'agglomerative_average_cosine', 'spectral': 'spectral_knn', 'gmm': 'gmm_diag', 'hdbscan': 'hdbscan'}
C['scores'] = {}
for rep in ('l2', 'centered_l2', 'pca50_l2', 'pca100_l2'):
    for vk, v in V.items():
        for mk, m in M.items():
            if m in ('gmm_diag', 'hdbscan') and rep not in ('centered_l2', 'pca50_l2'):
                continue
            ks = [None] if m == 'hdbscan' else [10, 13]
            for k in ks:
                seeds = [0, 1, 2, 3, 4] if m in ('kmeans', 'spectral_knn', 'gmm_diag') else [0]
                P = [np.load(DEV / 'assignments' / f'{v}__{rep}__{m}__k{k}__s{s}.npy').astype(int) for s in seeds]
                a = [AMI(y, p) for p in P]; r = [ARI(y, p) for p in P]; n = [NMI(y, p) for p in P]
                big = max(100 * np.bincount(p[p >= 0]).max() / len(p) for p in P)
                noise = float(np.mean([100 * (p == -1).mean() for p in P]))
                row = summ[(summ.variant == v) & (summ.representation == rep) & (summ.method == m) & (summ.label == 'hs3') & (summ.scoring != 'assigned_only')]
                row = row[row.k.isna()] if k is None else row[row.k == k]
                chk(f'{vk} {rep} {mk} k{k} AMI', np.mean(a), row.AMI_mean.iloc[0]); chk(f'{vk} {rep} {mk} k{k} ARI', np.mean(r), row.ARI_mean.iloc[0])
                C['scores'][f'{vk}|{rep}|{mk}|k{k}'] = {'AMI': round(float(np.mean(a)), 3), 'AMI_sd': round(float(np.std(a, ddof=1)), 3) if len(a) > 1 else None,
                                                       'ARI': round(float(np.mean(r)), 3), 'NMI': round(float(np.mean(n)), 3), 'largest_pct_max': round(float(big), 1), 'noise_pct': round(noise, 1)}
# brief consistency with the user-supplied table
for (vk, mk, want) in [('clip', 'kmeans', .451), ('siglip', 'kmeans', .523), ('dinov2', 'kmeans', .538), ('clip', 'agglo', .512), ('siglip', 'agglo', .542), ('dinov2', 'agglo', .561),
                       ('clip', 'spectral', .493), ('siglip', 'spectral', .559), ('dinov2', 'spectral', .476), ('clip', 'gmm', .466), ('siglip', 'gmm', .512), ('dinov2', 'gmm', .499)]:
    chk(f'table {vk} {mk} centered k10 (3 dp)', C['scores'][f'{vk}|centered_l2|{mk}|k10']['AMI'], want, tol=5e-4)
for vk, want in [('clip', .401), ('siglip', .394), ('dinov2', .448)]:
    chk(f'table {vk} hdbscan centered', C['scores'][f'{vk}|centered_l2|hdbscan|kNone']['AMI'], want, tol=5e-4)
chk('dinov2 agglo ARI', C['scores']['dinov2|centered_l2|agglo|k10']['ARI'], .417, tol=5e-4); chk('siglip spectral ARI', C['scores']['siglip|centered_l2|spectral|k10']['ARI'], .311, tol=5e-4)
for vk, want in [('clip', .002), ('siglip', .001), ('dinov2', .361)]:
    chk(f'{vk} raw L2 agglo', C['scores'][f'{vk}|l2|agglo|k10']['AMI'], want, tol=5e-4)
for vk, want in [('clip', .509), ('siglip', .515), ('dinov2', .568)]:
    chk(f'{vk} pca100 agglo', C['scores'][f'{vk}|pca100_l2|agglo|k10']['AMI'], want, tol=5e-4)
chk('dinov2 agglo k13 (label-informed)', C['scores']['dinov2|centered_l2|agglo|k13']['AMI'], .584, tol=5e-4)

# ---- photo rights: final deck has no museum photos; check the dataset rights statement
chk('benchmark rights all CC BY-NC-SA 4.0', int(b.rights.eq('http://creativecommons.org/licenses/by-nc-sa/4.0/').all()), 1, tol=0)
pd.DataFrame(checks).to_csv(FIN / 'validation_checks.csv', index=False)
(FIN / 'final_claims.json').write_text(json.dumps(C, indent=2, default=str))
cc = pd.DataFrame(checks); print('checks', len(cc), 'passed', int(cc['pass'].sum())); print(cc[~cc['pass']].to_string())
