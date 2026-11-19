"""P1.5: independent verification of the DINOv2 S/B/L extraction and clustering results.

Independent of the runners: own preprocessing (torchvision, not timm.create_transform), own forward pass (manual block loop,
not forward_intermediates), own metric computation from the saved assignment files, own figure check from the CSVs.
Writes artifacts/p1_checks.csv, artifacts/P1_VALIDATION.md, artifacts/DINOV2_EXTRACTION_VALIDATION.md.
Usage (repo root): .venv/bin/python reports/roadmap/phases/P1/source/p1_verify.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import hashlib, json, subprocess, sys, time, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / 'reports/roadmap/phases/P1/artifacts'
P0A = ROOT / 'reports/roadmap/phases/P0/artifacts'
EMB = ROOT / 'reports/dinov2/embeddings'
MODELS = ['dinov2_s', 'dinov2_b', 'dinov2_l']
CHECKS = []


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'))


def chk(group, name, got, want, ok=None, note=''):
    ok = (got == want) if ok is None else ok
    CHECKS.append({'group': group, 'check': name, 'observed': got if not isinstance(got, float) else float(f'{got:.6g}'),
                   'expected': want if not isinstance(want, float) else float(f'{want:.6g}'), 'pass': bool(ok), 'note': note})
    print(('PASS ' if ok else 'FAIL ') + f'[{group}] {name}: {got}', flush=True)


def git_unchanged(path):
    return subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', path], cwd=ROOT).returncode == 0


def main():
    dm = json.loads((P0A / 'DEV_MANIFEST.json').read_text())
    key = dm['object_order']['ordered_inputs_sha256']
    dev = pd.read_csv(ROOT / dm['source_file']['path'], dtype=str, keep_default_na=False)
    models = json.loads((ROOT / 'reports/dinov2/models.json').read_text())['models']

    # A. environment and frozen inputs
    G = 'A environment / frozen inputs'
    lock = sorted(l.strip() for l in (ROOT / 'requirements-pilot.lock.txt').read_text().splitlines() if l.strip())
    inst = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata['Name'] != 'pip')
    chk(G, 'running interpreter is the repository .venv', str(Path(sys.prefix).resolve()), str((ROOT / '.venv').resolve()))
    chk(G, '.venv packages == requirements-pilot.lock.txt (pip excluded)', sorted(map(str.lower, inst)) == sorted(map(str.lower, lock)), True)
    chk(G, 'dev_ids.csv sha256 == frozen manifest', sha(ROOT / dm['source_file']['path']), dm['source_file']['sha256'])
    for f in ('reports/roadmap/phases/P0/artifacts/EXPERIMENT_PROTOCOL.json', 'reports/roadmap/phases/P0/artifacts/DEV_MANIFEST.json',
              'reports/roadmap/phases/P0/artifacts/BASELINE_REFERENCE.csv', 'reports/dinov2/embeddings/dev_dinov2_b_224.json', 'reports/dinov2/models.json',
              'reports/dinov2/source/dinov2_embed.py', 'reports/dinov2/dev/summary.csv'):
        chk(G, f'unchanged vs git HEAD: {f}', git_unchanged(f), True)
    order = hashlib.sha256(canonical([(a, b) for a, b in zip(dev.image_id, dev.sha256)]).encode()).hexdigest()
    chk(G, 'recomputed dev order key == DEV_MANIFEST ordered_inputs_sha256', order, key)

    # B. checkpoints
    G = 'B checkpoints'
    for m in MODELS:
        chk(G, f'{m} weight sha256 == models.json', sha(models[m]['weight_path']), models[m]['weight_sha256'])

    # C/D. embedding manifests and arrays
    G = 'C embeddings'
    descs, arrays = {}, {}
    for m in MODELS:
        e = json.loads((EMB / f'dev_{m}_224.json').read_text()); d = e['descriptor']; descs[m] = d
        chk(G, f'{m}: n == 12000 and set == dev', (e['n'], e['set']), (12000, 'dev'))
        chk(G, f'{m}: ordered_inputs_sha256 == frozen manifest', e['ordered_inputs_sha256'], key)
        chk(G, f'{m}: repo@revision/weights == models.json', (d['repo'], d['revision'], d['weight_sha256']), (models[m]['repo'], models[m]['revision'], models[m]['weight_sha256']))
        chk(G, f'{m}: cache fingerprint == sha256(descriptor)', e['fingerprint'], hashlib.sha256(canonical(d).encode()).hexdigest())
        chk(G, f'{m}: size 224, layers [-1,-2,-4], CLS + final LN', (d['size'], d['layers'], d['pooling']), (224, [-1, -2, -4], 'CLS token, final LayerNorm applied (norm=True)'))
        for lay, a in e['arrays'].items():
            p = ROOT / a['path']; X = np.load(p); arrays[(m, lay)] = X
            chk(G, f'{m} L{lay}: sha256', sha(p), a['sha256'])
            chk(G, f'{m} L{lay}: shape/dtype', (list(X.shape), str(X.dtype)), ([12000, models[m]['embed_dim']], 'float32'))
            chk(G, f'{m} L{lay}: all finite (no NaN/inf)', bool(np.isfinite(X).all()), True)
            chk(G, f'{m} L{lay}: no duplicate rows from a stuck batch', int(len(np.unique(X.round(5), axis=0))), 12000, ok=len(np.unique(X.round(5), axis=0)) >= 11990,
                note='identical images would legitimately repeat; dev has 0 duplicate pixel hashes')
        chk(G, f'{m}: layers -1/-2/-4 are three different arrays', len({sha(ROOT / a['path']) for a in e['arrays'].values()}), 3)
    for k in ('adaptation', 'transform', 'pooling', 'size', 'layers'):
        chk(G, f'S/B/L share identical {k}', len({canonical(descs[m][k]) for m in MODELS}), 1)
    chk(G, 'DINOv2-B L1 sha == P0 DEV_MANIFEST (reused, not re-extracted)', json.loads((EMB / 'dev_dinov2_b_224.json').read_text())['arrays']['-1']['sha256'],
        dm['embeddings_aligned']['dinov2_b_224_L1']['sha256'])

    # E. caches contain dev chunks only
    G = 'D reserved isolation'
    ch = {s: hashlib.sha256(canonical([(a, b) for a, b in zip(dev.image_id[s:s + 16], dev.sha256[s:s + 16])]).encode()).hexdigest() for s in range(0, 12000, 16)}
    for m in MODELS:
        fp = json.loads((EMB / f'dev_{m}_224.json').read_text())['fingerprint']; files = sorted((ROOT / 'data/dinov2/cache' / fp).glob('*.npz'))
        bad = sum(str(np.load(f)['key']) != ch.get(int(f.stem)) for f in files)
        chk(G, f'{m}: cache chunks == 750 dev chunks, keys all dev', (len(files), bad), (750, 0))
    chk(G, 'no held-out / reserved embedding manifest', sorted(p.name for p in EMB.glob('*held*')), [])
    res = pd.read_csv(ROOT / 'reports/full_dataset/benchmark/heldout_ids_DO_NOT_USE_FOR_SELECTION.csv', dtype=str, usecols=['image_id'])
    chk(G, 'dev image ids ∩ reserved image ids', len(set(dev.image_id) & set(res.image_id)), 0)

    # F. independent preprocessing + forward pass on a random dev sample; layer mapping
    G = 'E layer mapping (independent forward)'
    import torch, torchvision.transforms as T
    from PIL import Image, ImageOps
    sys.path.insert(0, str(ROOT / 'reports/dinov2/source'))
    from dinov2_weights import load
    torch.set_num_threads(4)
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    idx = np.sort(np.random.default_rng(2026).choice(12000, 24, replace=False))
    imgs = []
    for i in idx:
        r = dev.iloc[i]; p = ROOT / 'data/mimo_pilot' / r.file; assert sha(p) == r.sha256
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert('RGB'); s = max(im.size)
            imgs.append(tf(ImageOps.pad(im, (s, s), color=(255, 255, 255))))
    x = torch.stack(imgs)
    layer_rows = []
    for m in MODELS:
        model = load(m)
        with torch.inference_mode():
            h = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
            outs = []
            for blk in model.blocks:
                h = blk(h); outs.append(model.norm(h)[:, 0].float().numpy())
        depth = len(model.blocks)
        for lay in ('-1', '-2', '-4'):
            saved = arrays[(m, lay)][idx]
            diffs = {L: float(np.abs(outs[depth + int(L)] - saved).max()) for L in ('-1', '-2', '-4')}
            cs = np.sum(outs[depth + int(lay)] * saved, 1) / (np.linalg.norm(outs[depth + int(lay)], axis=1) * np.linalg.norm(saved, axis=1))
            own = diffs[lay]; other = min(v for L, v in diffs.items() if L != lay)
            chk(G, f'{m} saved L{lay} == independent CLS of block {depth + int(lay)} (max |Δ|, min cos)', f'{own:.2e}, {cs.min():.7f}', '< 1e-3, > 0.99999',
                ok=own < 1e-3 and cs.min() > 0.99999)
            chk(G, f'{m} saved L{lay} differs from the other two layers (min max|Δ|)', f'{other:.2f}', '> 1', ok=other > 1)
            layer_rows.append({'model': m, 'layer': int(lay), 'block_index_0based': depth + int(lay), 'blocks': depth, 'rows_checked': len(idx),
                               'max_abs_diff_own_layer': own, 'min_cosine_own_layer': float(cs.min()), 'min_max_abs_diff_other_layers': other})
        if m == 'dinov2_b':
            with torch.inference_mode():
                pooled = model(x).float().numpy()
            chk(G, 'B: independent block-loop layer -1 == timm pooled output', float(np.abs(pooled - outs[-1]).max()), 0.0, ok=np.abs(pooled - outs[-1]).max() < 1e-4)
        del model
    pd.DataFrame(layer_rows).to_csv(ART / 'layer_mapping_check.csv', index=False)

    # G. independent rescoring of every saved assignment
    G = 'F metrics (independent rescoring)'
    runs = pd.read_csv(ART / 'p1_runs.csv'); per = pd.read_csv(ART / 'p1_scores_per_run.csv').set_index('run_id')
    abl = pd.read_csv(ART / 'DINOV2_LAYER_ABLATION.csv'); main_t = pd.read_csv(ART / 'DINOV2_SBL_12K.csv')
    y3, y1 = dev.hs3.to_numpy(), dev.hs1.to_numpy()
    rec = []
    for r in runs.itertuples():
        p = ART / 'assignments' / f'{r.run_id}.npy'; pred = np.load(p).astype(int)
        rec.append({'run_id': r.run_id, 'model': r.model, 'layer': r.layer, 'method': r.method, 'sha_ok': sha(p) == r.assignment_sha256, 'n': len(pred),
                    'AMI': AMI(y3, pred), 'ARI': ARI(y3, pred), 'NMI': NMI(y3, pred), 'HS1_AMI': AMI(y1, pred)})
    rec = pd.DataFrame(rec)
    chk(G, 'fits saved (3 models × 3 layers × 11)', len(rec), 99)
    chk(G, 'assignment sha256 == runs.csv, length 12000', bool(rec.sha_ok.all() and (rec.n == 12000).all()), True)
    d = max(float((rec.set_index('run_id')[c] - per[c2].reindex(rec.run_id).values).abs().max()) for c, c2 in [('AMI', 'HS3_AMI'), ('ARI', 'HS3_ARI'), ('NMI', 'HS3_NMI'), ('HS1_AMI', 'HS1_AMI')])
    chk(G, 'per-run HS3 AMI/ARI/NMI, HS1 AMI == runner (max |Δ|)', d, 0.0, ok=d < 1e-12)
    agg = rec.groupby(['model', 'layer', 'method']).agg(AMI=('AMI', 'mean'), ARI=('ARI', 'mean'), NMI=('NMI', 'mean'), sd=('AMI', lambda s: s.std(ddof=1) if len(s) > 1 else np.nan)).reset_index()
    mm = agg.merge(abl, on=['model', 'layer', 'method'])
    d = float(max((mm.AMI - mm.HS3_AMI_mean).abs().max(), (mm.ARI - mm.HS3_ARI_mean).abs().max(), (mm.NMI - mm.HS3_NMI_mean).abs().max(), (mm.sd - mm.HS3_AMI_sd).abs().max()))
    chk(G, 'DINOV2_LAYER_ABLATION.csv means/SD == independent aggregation (27 cells, max |Δ|)', d, 0.0, ok=len(mm) == 27 and d < 1e-6, note='CSV rounded to 6 decimals')
    m1 = agg[agg.layer == -1].merge(main_t[main_t.model.isin(MODELS)], on=['model', 'method'])
    d = float(max((m1.AMI - m1.HS3_AMI_mean).abs().max(), (m1.ARI - m1.HS3_ARI_mean).abs().max()))
    chk(G, 'DINOV2_SBL_12K.csv DINOv2 rows == independent aggregation (9 cells)', d, 0.0, ok=len(m1) == 9 and d < 1e-6)
    base = pd.read_csv(P0A / 'BASELINE_REFERENCE.csv')
    ctx = main_t[main_t.model.isin(['clip', 'siglip'])].merge(base[base.role == 'primary_fixed_k10'], left_on=['model', 'method'], right_on=['backbone', 'method'])
    chk(G, 'CLIP/SigLIP context rows == P0 BASELINE_REFERENCE (6 cells)', float((ctx.HS3_AMI_mean_x - ctx.HS3_AMI_mean_y).abs().max()), 0.0,
        ok=len(ctx) == 6 and (ctx.HS3_AMI_mean_x - ctx.HS3_AMI_mean_y).abs().max() < 1e-6)
    n_sd = abl[(abl.method == 'agglomerative_average_cosine')].HS3_AMI_sd.isna().all() and abl[abl.method == 'agglomerative_average_cosine'].n_seeds.eq(1).all()
    chk(G, 'deterministic agglomerative reported with n_seeds 1 and no SD', bool(n_sd), True)

    # H. DINOv2-B baseline reproduces
    G = 'G DINOv2-B baseline reproduction'
    b = runs[(runs.model == 'dinov2_b') & (runs.layer == -1)]
    same = [np.array_equal(np.load(ART / 'assignments' / f'{r}.npy'), np.load(ROOT / 'reports/dinov2/dev/assignments' / f'{r}.npy')) for r in b.run_id]
    chk(G, 'P1 DINOv2-B L1 assignments bit-identical to Week 6 dev assignments', f'{sum(same)}/{len(same)}', f'{len(same)}/{len(same)}', ok=all(same) and len(same) == 11)
    for meth, (a_, r_) in {'kmeans': (0.538, 0.314), 'agglomerative_average_cosine': (0.561, 0.417), 'spectral_knn': (0.476, 0.165)}.items():
        x = agg[(agg.model == 'dinov2_b') & (agg.layer == -1) & (agg.method == meth)].iloc[0]
        ref = base[(base.backbone == 'dinov2_b_224_L1') & (base.method == meth) & (base.role == 'primary_fixed_k10')].iloc[0]
        chk(G, f'B {meth}: AMI/ARI == P0 reference ({a_}/{r_})', f'{x.AMI:.3f}/{x.ARI:.3f}', f'{a_:.3f}/{r_:.3f}', ok=abs(x.AMI - ref.HS3_AMI_mean) < 1e-6 and abs(x.ARI - ref.HS3_ARI_mean) < 1e-6, note='tolerance 1e-6 = 6-decimal rounding of BASELINE_REFERENCE.csv')

    # I. re-fit spot check (determinism of the new fits)
    G = 'H re-fit determinism'
    sys.path.insert(0, str(ROOT / 'reports/week6/source'))
    from p2_benchmark import make
    from sklearn.preprocessing import normalize
    with threadpool_limits(limits=4):
        for rid in ('dinov2_s_224_L1__centered_l2__kmeans__k10__s3', 'dinov2_l_224_L1__centered_l2__agglomerative_average_cosine__k10__s0',
                    'dinov2_l_224_L4__centered_l2__spectral_knn__k10__s1'):
            r = runs[runs.run_id == rid].iloc[0]; X = arrays[(r.model, str(r.layer))]
            pred = make(r.method, 10, int(r.seed), 20).fit_predict(normalize(X - X.mean(0)).astype(np.float32)).astype(np.int16)
            chk(G, f'refit {rid} bit-identical', bool(np.array_equal(pred, np.load(ART / 'assignments' / f'{rid}.npy'))), True)

    # J. figures vs CSV
    G = 'I figures'
    fv = json.loads((ART / 'figures/figure_values.json').read_text())
    for name, png in fv['_png_sha256'].items():
        chk(G, f'{name} exists and sha256 == figure_values.json', sha(ART / 'figures' / name), png)
    allt = pd.concat([abl, main_t[main_t.model.isin(['clip', 'siglip'])].assign(layer=-1)])
    worst = 0.0; n = 0
    for fig_, items in fv.items():
        if fig_.startswith('_'): continue
        for it in items:
            q = allt[(allt.model == it['model']) & (allt.method == it['method'])]
            q = q[q.layer == it.get('layer', -1)].iloc[0]
            for k in ('HS3_AMI_mean', 'HS3_ARI_mean', 'largest_cluster_pct_max'):
                if k in it: worst = max(worst, abs(it[k] - q[k])); n += 1
    chk(G, f'every plotted value ({n}) == source CSV', worst, 0.0, ok=worst < 1e-6)
    rt = pd.read_csv(ART / 'DINOV2_RUNTIME_MEMORY.csv'); ex = rt[rt.stage == 'extraction'].set_index('model')
    w = max(abs(it['extraction_minutes'] - ex.loc[it['model'], 'active_seconds'] / 60) for it in fv['P1_RUNTIME_TRADEOFF'])
    chk(G, 'runtime figure x-values == DINOV2_RUNTIME_MEMORY.csv', w, 0.0, ok=w < 1e-9)
    for m in MODELS:
        chk(G, f'{m} extraction seconds == embedding manifest', ex.loc[m, 'seconds'], json.loads((EMB / f'dev_{m}_224.json').read_text())['seconds'])

    c = pd.DataFrame(CHECKS); c.to_csv(ART / 'p1_checks.csv', index=False)
    summ = {'checks': len(c), 'passed': int(c['pass'].sum()), 'failed': int((~c['pass']).sum()), 'by_group': {g: f"{int(d['pass'].sum())}/{len(d)}" for g, d in c.groupby('group')},
            'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'python': sys.executable}
    (ART / 'p1_summary.json').write_text(json.dumps(summ, indent=2))

    def table(groups):
        L = []
        for g, d in c[c.group.isin(groups)].rename(columns={'pass': 'ok'}).groupby('group', sort=False):
            L += [f'### {g}', '', '| # | Check | Observed | Expected | Result |', '|---|---|---|---|---|']
            for i, r in enumerate(d.itertuples(), 1):
                o, e = str(r.observed).replace('|', '\\|'), str(r.expected).replace('|', '\\|')
                o = o if len(o) < 75 else o[:72] + '…'; e = e if len(e) < 75 else e[:72] + '…'
                L.append(f'| {i} | {r.check} | {o} | {e} | {"PASS" if r.ok else "FAIL"}{" — " + r.note if isinstance(r.note, str) and r.note else ""} |')
            L.append('')
        return L
    head = [f'Generated by `reports/roadmap/phases/P1/source/p1_verify.py` at {summ["utc"]} (repository `.venv`, OMP_NUM_THREADS=4).', '']
    ext_groups = ['A environment / frozen inputs', 'B checkpoints', 'C embeddings', 'D reserved isolation', 'E layer mapping (independent forward)']
    ce = c[c.group.isin(ext_groups)]
    (ART / 'DINOV2_EXTRACTION_VALIDATION.md').write_text('\n'.join(
        ['# DINOv2 S/B/L extraction validation (P1)', ''] + head +
        [f'**{int(ce["pass"].sum())} / {len(ce)} extraction checks passed.**', '',
         'Layer mapping was checked by an independent re-implementation: torchvision preprocessing (white square pad → Resize 224 bicubic → CenterCrop 224 → ImageNet normalisation) '
         'and a manual loop over `model.blocks` with the final LayerNorm applied to the CLS token of every block, on 24 random dev images '
         '(`np.random.default_rng(2026)`). Each saved layer must match its own block and must differ from the other two. Per-model numbers: `layer_mapping_check.csv`.', ''] + table(ext_groups)))
    (ART / 'P1_VALIDATION.md').write_text('\n'.join(
        ['# P1 validation evidence (generated)', ''] + head + [f'**{summ["passed"]} / {summ["checks"]} checks passed, {summ["failed"]} failed.**', '',
                                                               '| Group | Passed |', '|---|---|'] + [f'| {g} | {v} |' for g, v in summ['by_group'].items()] + [''] +
        ['Extraction groups A–E are listed in full in `DINOV2_EXTRACTION_VALIDATION.md`.', ''] + table([g for g in c.group.unique() if g not in ext_groups])))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
