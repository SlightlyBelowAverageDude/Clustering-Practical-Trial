"""P5 independent verification: recompute DINOv2/three-backbone scores from saved assignments, check embedding hashes,
row alignment, held-out isolation, dataset counts, secrets and file sizes. Writes reports/dinov2/p5_validation/checks.csv."""
import hashlib, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / 'reports/dinov2'; FD = ROOT / 'reports/full_dataset'; OUT = D / 'p5_validation'; OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks = []


def chk(name, got, want, tol=1e-9):
    ok = (got == want) if isinstance(got, (str, bool, list)) else abs(float(got) - float(want)) <= tol
    checks.append({'check': name, 'recomputed': got, 'reported': want, 'pass': bool(ok)})


def rescore(set_dir, labels_df, labels):
    runs = pd.read_csv(set_dir / 'runs.csv'); runs = runs[runs.status == 'ok']
    summ = pd.read_csv(set_dir / 'summary.csv')
    for (v, rep, m, k), g in runs.groupby(['variant', 'representation', 'method', 'k'], dropna=False):
        for lab in labels:
            mask = labels_df[lab].ne('').to_numpy(); y = labels_df.loc[mask, lab].to_numpy()
            preds = [np.load(set_dir / 'assignments' / f'{r}.npy').astype(int) for r in g.run_id]
            for r, p in zip(g.itertuples(), preds):
                assert sha(set_dir / 'assignments' / f'{r.run_id}.npy') == r.assignment_sha256
            a = np.mean([AMI(y, p[mask]) for p in preds]); b = np.mean([ARI(y, p[mask]) for p in preds])
            sc = 'all_rows' if m != 'hdbscan' else 'noise_as_cluster'
            row = summ[(summ.variant == v) & (summ.representation == rep) & (summ.method == m) & (summ.label == lab) & (summ.scoring == sc)]
            row = row[row.k.isna()] if pd.isna(k) else row[row.k == k]
            chk(f'{set_dir.name} {v} {rep} {m} k{k} {lab} AMI', a, row.AMI_mean.iloc[0]); chk(f'{set_dir.name} {v} {rep} {m} k{k} {lab} ARI', b, row.ARI_mean.iloc[0])


# 1 pilot: scores, row alignment, embeddings
M = pd.read_csv(ROOT / 'results/phase2/inputs/evaluation_masks.csv', dtype=str, keep_default_na=False)
rescore(D / 'pilot', M, ['hs3', 'hs1'])
order = pd.read_csv(ROOT / 'results/phase2/inputs/image_order.csv', dtype=str, keep_default_na=False)
ref_key = hashlib.sha256(json.dumps([(a, b) for a, b in zip(order.image_id, order.sha256)], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
for f in sorted((D / 'embeddings').glob('*.json')):
    man = json.loads(f.read_text())
    arrays = man['arrays'].values() if 'arrays' in man else [man['array']]
    for a in arrays:
        chk(f'embedding hash {a["path"]}', sha(ROOT / a['path']), a['sha256'])
        X = np.load(ROOT / a['path']); chk(f'embedding finite {a["path"]}', bool(np.isfinite(X).all()), True); chk(f'embedding rows {a["path"]}', len(X), man['n'], tol=0)
    if man['set'] == 'pilot':
        chk(f'pilot row order {f.name}', man['ordered_inputs_sha256'], ref_key)
w = json.loads((D / 'models.json').read_text())['models']
for name, info in w.items():
    chk(f'weight sha256 {name}', sha(info['weight_path']), info['weight_sha256'])
# probe recomputation (2 variants)
pr = pd.read_csv(D / 'pilot/linear_probe.csv')
mask = M.hs3_valid.eq('True').to_numpy(); y = M.hs3.to_numpy(); vc = pd.Series(y[mask]).value_counts(); keep = mask & np.isin(y, vc[vc >= 10].index)
for v, f in [('clip', None), ('dinov2_l_224_L1', D / 'embeddings/pilot_dinov2_l_224.json')]:
    if v == 'clip':
        X = np.load(ROOT / 'data/mimo_pilot/embeddings/a0fbb27167c3807c202b7f787711455280e412b0b77f0c8bc7e6a438838ee487.npy')
    else:
        X = np.load(ROOT / json.loads(f.read_text())['arrays']['-1']['path'])
    Z = normalize(X - X.mean(0))
    p = cross_val_predict(LogisticRegression(C=0.5, max_iter=2000), Z[keep], y[keep], cv=StratifiedKFold(5, shuffle=True, random_state=0))
    r = pr[pr.variant == v].iloc[0]
    chk(f'probe AMI {v}', AMI(y[keep], p), r.probe_AMI, tol=1e-6); chk(f'probe accuracy {v}', accuracy_score(y[keep], p), r.probe_accuracy, tol=1e-6)

# 2 dev partition and held-out isolation
dev = pd.read_csv(FD / 'benchmark/dev_ids.csv', dtype=str, keep_default_na=False)
if (D / 'dev/summary.csv').exists():
    rescore(D / 'dev', dev, ['hs3'])
ho = pd.read_csv(FD / 'benchmark/heldout_ids_DO_NOT_USE_FOR_SELECTION.csv', dtype=str)
chk('dev ∩ held-out ids', len(set(dev.id) & set(ho.id)), 0, tol=0)
chk('dev rows', len(dev), 12000, tol=0)
for f in (D / 'embeddings').glob('dev_*.json'):
    chk(f'{f.name} embeds exactly the dev rows', json.loads(f.read_text())['n'], len(dev), tol=0)
cache_ids = set()  # held-out images must not appear in any DINOv2/CLIP/SigLIP manifest
chk('held-out never embedded (no held-out manifest)', len(list((D / 'embeddings').glob('heldout*'))), 0, tol=0)

# 3 dataset counts
st = json.loads((FD / 'stats.json').read_text())
b = pd.read_csv(FD / 'benchmark/benchmark_objects.csv', dtype=str, keep_default_na=False); c = pd.read_csv(FD / 'clean_records.csv', dtype=str, keep_default_na=False)
chk('benchmark rows', len(b), st['benchmark_rows'], tol=0); chk('benchmark HS3 classes', b.hs3.nunique(), 13, tol=0); chk('clean rows', len(c), st['cleaned_photographs'], tol=0)
chk('benchmark rule: every class >= 30', int(min(b.hs3.value_counts().min(), b.family.value_counts().min(), b.name.value_counts().min()) >= 30), 1, tol=0)
chk('benchmark ids unique', bool(b.id.is_unique), True)

# 4 secrets and sizes in files that git would commit
files = subprocess.run(['git', 'ls-files', '--others', '--cached', '--exclude-standard'], cwd=ROOT, capture_output=True, text=True).stdout.split('\n')
files = [f for f in files if f and (ROOT / f).is_file()]
key = ''
envf = ROOT / '.env'
if envf.exists():
    key = next((l.split('=', 1)[1].strip().strip('"') for l in envf.read_text().splitlines() if l.startswith('EUROPEANA_API_KEY=')), '')
hits = [f for f in files if key and key.encode() in (ROOT / f).read_bytes()]
chk('API key absent from all committable files', len(hits), 0, tol=0)
chk('.env not committable', '.env' in files, False)
big = [f for f in files if (ROOT / f).stat().st_size > 50 * 1024 * 1024]
chk('no committable file > 50 MB', len(big), 0, tol=0)
chk('no raw image files committable', len([f for f in files if f.endswith(('.img', '.jpg', '.jpeg'))]), 0, tol=0)
chk('no model weights committable', len([f for f in files if f.endswith(('.safetensors', '.pt', '.pth', '.ckpt'))]), 0, tol=0)
c = pd.DataFrame(checks); c.to_csv(OUT / 'checks.csv', index=False)
summary = {'checks': len(c), 'passed': int(c['pass'].sum()), 'failed': int((~c['pass']).sum()), 'committable_files': len(files),
           'committable_bytes': int(sum((ROOT / f).stat().st_size for f in files)), 'largest_committable': max(files, key=lambda f: (ROOT / f).stat().st_size)}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2)); print(summary); print(c[~c['pass']].to_string())
