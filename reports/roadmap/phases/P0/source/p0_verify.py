"""P0: freeze a fair comparison protocol (execution roadmap W8 -> Final).

Audit-only. Nothing is embedded; no reserved (held-out / pilot-overlap) object is clustered or scored.

Stages (run all with no argument, or one by name):
  dataset   revalidate the 12,000-object development partition and write artifacts/DEV_MANIFEST.json
  baseline  rescore every saved development baseline assignment, recompute silhouettes, re-fit the k=10 grid from the
            cached development embeddings, write artifacts/BASELINE_REFERENCE.csv and artifacts/baseline_refit.csv
  protocol  write artifacts/EXPERIMENT_PROTOCOL.json (frozen rules + hashes of every input it references)
  report    write artifacts/p0_checks.csv, artifacts/p0_summary.json and artifacts/P0_VALIDATION.md

Reserved-data discipline:
  * reports/full_dataset/benchmark/heldout_ids_DO_NOT_USE_FOR_SELECTION.csv holds only id/partition/image_id/sha256.
  * From benchmark_objects.csv only identifier/provenance columns are loaded for non-development rows
    (id, partition, image_id, sha256, pixel_sha256, in_explored_pilot); label columns are read for development rows only.

Usage (repository root): python reports/roadmap/phases/P0/source/p0_verify.py [dataset|baseline|protocol|report]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')  # frozen protocol: 4 BLAS threads (6 changes PCA rounding, see DINOV2_BENCHMARK.md)
import csv, hashlib, json, platform, sys, time, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_mutual_info_score as AMI, adjusted_rand_score as ARI,
                             normalized_mutual_info_score as NMI, pairwise_distances, silhouette_score)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[5]
P0 = ROOT / 'reports/roadmap/phases/P0'
ART = P0 / 'artifacts'
FD = ROOT / 'reports/full_dataset'
BM = FD / 'benchmark'
DV = ROOT / 'reports/dinov2'
sys.path.insert(0, str(ROOT / 'reports/week6/source'))
from p2_benchmark import make, sizes  # noqa: E402  (the exact clustering factory used for every Week 6 / dev result)

DEV_CSV = BM / 'dev_ids.csv'
HELDOUT_CSV = BM / 'heldout_ids_DO_NOT_USE_FOR_SELECTION.csv'
BENCH_CSV = BM / 'benchmark_objects.csv'
ID_COLS = ['id', 'partition', 'image_id', 'sha256', 'pixel_sha256', 'in_explored_pilot']
LABEL_LEVELS = ['hs1', 'hs2', 'hs3', 'family', 'name']
BACKBONES = {'clip': 'dev_clip.json', 'siglip': 'dev_siglip.json', 'dinov2_b_224_L1': 'dev_dinov2_b_224.json'}
SEEDS = [0, 1, 2, 3, 4]
CHECKS = []


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'))


def rel(p):
    return str(Path(p).relative_to(ROOT))


def chk(group, name, got, want, ok=None, note='', kind='gate'):
    """kind: gate = must pass for P0; info = recorded value; finding = pre-existing issue documented and handed on."""
    if ok is None:
        ok = got == want
    CHECKS.append({'group': group, 'kind': kind, 'check': name, 'observed': got if not isinstance(got, float) else round(got, 6),
                   'expected': want if not isinstance(want, float) else round(want, 6), 'pass': bool(ok), 'note': note})
    print(('PASS ' if ok else ('FINDING ' if kind == 'finding' else 'FAIL ')) + f'[{group}] {name}: {got} (expected {want})', flush=True)


def save_checks(stage):
    f = ART / f'checks_{stage}.csv'
    pd.DataFrame(CHECKS).to_csv(f, index=False)


# ----------------------------------------------------------------------------------------------------------- dataset
def stage_dataset():
    G = 'dataset'
    stats = json.loads((FD / 'stats.json').read_text())
    man = json.loads((FD / 'manifest.json').read_text())

    # 0. split files are the ones the Week 6 build recorded
    for f in (BENCH_CSV, DEV_CSV, HELDOUT_CSV):
        chk(G, f'sha256 {rel(f)} matches full_dataset/manifest.json', sha(f), man['files'][rel(f)]['sha256'])

    # 1. identifiers only, for every benchmark row (no label columns for reserved rows)
    ids = pd.read_csv(BENCH_CSV, dtype=str, keep_default_na=False, usecols=ID_COLS)
    part = ids.partition.value_counts().to_dict()
    chk(G, 'eligible benchmark objects', len(ids), 27347)
    chk(G, 'development objects', part.get('dev', 0), 12000)
    chk(G, 'reserved objects (heldout)', part.get('heldout', 0), 14696)
    chk(G, 'pilot-overlap exclusions', part.get('excluded_pilot_overlap', 0), 651)
    chk(G, 'partition labels are exactly {dev, heldout, excluded_pilot_overlap}', sorted(part), sorted(['dev', 'heldout', 'excluded_pilot_overlap']))
    chk(G, 'benchmark object ids unique', bool(ids.id.is_unique), True)
    chk(G, 'cleaned photographs (stats.json, recorded at build)', stats['cleaned_photographs'], 59658)
    clean = pd.read_csv(FD / 'clean_records.csv', dtype=str, keep_default_na=False, usecols=['id'])
    chk(G, 'cleaned photographs (rows in clean_records.csv)', len(clean), 59658)
    chk(G, 'cleaned photograph ids unique', bool(clean.id.is_unique), True)
    chk(G, 'every benchmark object is a cleaned photograph', bool(ids.id.isin(set(clean.id)).all()), True)

    # 2. the development subset rule reproduces exactly (rng(0).choice over benchmark row order, sorted)
    pick = np.sort(np.random.default_rng(0).choice(np.arange(len(ids)), 12000, replace=False))
    rule_ids = ids.id.to_numpy()[pick].tolist()
    dev = pd.read_csv(DEV_CSV, dtype=str, keep_default_na=False)
    chk(G, 'dev_ids.csv rows', len(dev), 12000)
    chk(G, 'dev_ids.csv: every row partition == dev', bool((dev.partition == 'dev').all()), True)
    chk(G, 'dev ids == default_rng(0).choice(27347, 12000) rule, same order', dev.id.tolist() == rule_ids, True)
    chk(G, 'dev ids unique', bool(dev.id.is_unique), True)
    chk(G, 'dev order == benchmark order restricted to dev', dev.id.tolist() == ids.loc[ids.partition == 'dev', 'id'].tolist(), True)

    # 3. dev label rows in dev_ids.csv are byte-identical to the dev rows of benchmark_objects.csv (dev rows only parsed)
    with open(BENCH_CSV, encoding='utf8', newline='') as fh:
        rd = csv.reader(fh); header = next(rd)
        pi = header.index('partition')
        bench_dev = [r for r in rd if r[pi] == 'dev']
    chk(G, 'dev_ids.csv columns == benchmark_objects.csv columns', list(dev.columns) == header, True)
    chk(G, 'dev rows identical in dev_ids.csv and benchmark_objects.csv', dev.astype(str).values.tolist() == bench_dev, True)

    # 4. separation: ids, images, bytes and pixels never cross the dev / reserved boundary
    res = ids[ids.partition != 'dev']
    ho = pd.read_csv(HELDOUT_CSV, dtype=str, keep_default_na=False)
    chk(G, 'heldout file columns are identifiers only (no labels)', list(ho.columns), ['id', 'partition', 'image_id', 'sha256'])
    chk(G, 'heldout file == all non-dev benchmark rows (id, partition, image_id, sha256)',
        ho.values.tolist() == res[['id', 'partition', 'image_id', 'sha256']].values.tolist(), True)
    for col in ('id', 'image_id', 'sha256', 'pixel_sha256'):
        n = len(set(dev[col]) & set(res[col]))
        chk(G, f'dev ∩ reserved on {col}', n, 0)
    pilot = pd.read_csv(ROOT / 'data/mimo_pilot/sample_pilot.csv', dtype=str, keep_default_na=False, usecols=['id', 'image_id'])
    ip = ids.id.isin(set(pilot.id))
    chk(G, 'in_explored_pilot flag == membership in data/mimo_pilot/sample_pilot.csv', bool((ip == ids.in_explored_pilot.eq('True')).all()), True)
    chk(G, 'pilot objects inside dev (explored, allowed by split rule)', int(ip[ids.partition == 'dev'].sum()), 465)
    chk(G, 'pilot objects inside reserved heldout', int(ip[ids.partition == 'heldout'].sum()), 0)
    chk(G, 'every excluded_pilot_overlap row is a pilot object', bool(ip[ids.partition == 'excluded_pilot_overlap'].all()), True)
    chk(G, 'heldout image_ids ∩ pilot image_ids', len(set(ids.loc[ids.partition == 'heldout', 'image_id']) & set(pilot.image_id)), 0)

    # 5. image-to-object mapping: one record = one object = one image, file present and hash-identical
    chk(G, 'dev image_id unique (one image per object)', bool(dev.image_id.is_unique), True)
    chk(G, 'dev file column == images/<image_id>.img', bool((dev.file == 'images/' + dev.image_id + '.img').all()), True)
    bad, missing = 0, 0
    for r in dev.itertuples():
        p = ROOT / 'data/mimo_pilot' / r.file
        if not p.exists():
            missing += 1
        elif sha(p) != r.sha256:
            bad += 1
    chk(G, 'dev image files present locally', missing, 0)
    chk(G, 'dev image file sha256 == manifest sha256', bad, 0)
    dup_px = int(dev.pixel_sha256.duplicated().sum())
    chk(G, 'dev rows sharing a pixel hash with another dev row (byte/pixel duplicates)', dup_px, 0, note='exact pixel duplicates only; near-duplicates are a P5 task')

    # 6. labels: complete, hierarchical, consistent with the HS notation, 13 HS3 classes
    for lev in LABEL_LEVELS:
        chk(G, f'dev rows with empty {lev}', int((dev[lev] == '').sum()), 0)
    chk(G, 'hs1 values ⊂ {1..5}', sorted(dev.hs1.unique()), sorted(set(dev.hs1) & {'1', '2', '3', '4', '5'}))
    chk(G, 'hs2 startswith hs1 (len 2)', int((~(dev.hs2.str.len().eq(2) & (dev.hs2.str[:1] == dev.hs1))).sum()), 0)
    chk(G, 'hs3 startswith hs2 (len 3)', int((~(dev.hs3.str.len().eq(3) & (dev.hs3.str[:2] == dev.hs2))).sum()), 0)
    digits = dev.notation.str.replace(r'[^0-9]', '', regex=True)
    chk(G, 'hs3 == first three digits of HS notation', int((digits.str[:3] != dev.hs3).sum()), 0)
    hs3_classes = sorted(dev.hs3.unique())
    chk(G, 'dev HS3 classes', len(hs3_classes), 13)
    chk(G, 'dev HS3 class set == benchmark HS3 class set (stats.json keys)', hs3_classes, sorted(stats['hs3_benchmark_counts']))
    dist = pd.read_csv(FD / 'class_distributions.csv', dtype=str, keep_default_na=False)
    for lev in ('hs1', 'hs3', 'family'):
        ref = dist[(dist.scope == 'dev') & (dist.level == lev)].set_index('class').n.astype(int).to_dict()
        chk(G, f'dev {lev} counts == class_distributions.csv (scope dev)', dev[lev].value_counts().to_dict() == ref, True)

    # 7. museums
    mus = pd.read_csv(FD / 'museum_distribution.csv')
    ref_m = mus[mus.dev > 0].set_index('museum').dev.to_dict()
    chk(G, 'dev rows with empty museum (provider)', int((dev.provider == '').sum()), 0)
    chk(G, 'dev museum counts == museum_distribution.csv (dev column)', dev.provider.value_counts().to_dict() == ref_m, True)
    n_mus = dev.provider.nunique()
    top5 = dev.provider.value_counts().head(5)
    chk(G, 'dev museums', n_mus, n_mus, kind='info')

    # 8. embedding alignment: every cached dev embedding was computed on exactly these rows in this order
    order_key = hashlib.sha256(canonical([(a, b) for a, b in zip(dev.image_id, dev.sha256)]).encode()).hexdigest()
    emb = {}
    for name, f in BACKBONES.items():
        m = json.loads((DV / 'embeddings' / f).read_text())
        arr = m['arrays']['-1'] if 'arrays' in m else m['array']
        chk(G, f'{f}: ordered_inputs_sha256 == dev (image_id, sha256) order', m['ordered_inputs_sha256'], order_key)
        chk(G, f'{f}: n', m['n'], 12000)
        p = ROOT / arr['path']; h = sha(p)
        chk(G, f'{arr["path"]}: sha256', h, arr['sha256'])
        X = np.load(p)
        chk(G, f'{arr["path"]}: shape', list(X.shape), arr['shape'])
        chk(G, f'{arr["path"]}: dtype float32, all finite', f'{X.dtype},{bool(np.isfinite(X).all())}', 'float32,True')
        nrm = np.linalg.norm(X, axis=1)
        emb[name] = {'manifest': rel(DV / 'embeddings' / f), 'array': arr['path'], 'sha256': h, 'shape': list(X.shape), 'dtype': str(X.dtype),
                     'row_norm_min': float(nrm.min()), 'row_norm_max': float(nrm.max())}
        if name in ('clip', 'siglip'):
            chk(G, f'{arr["path"]}: rows L2-normalised at extraction', bool(np.allclose(nrm, 1, atol=1e-4)), True)
    for layer in ('-2', '-4'):
        m = json.loads((DV / 'embeddings/dev_dinov2_b_224.json').read_text())['arrays'][layer]
        chk(G, f'{m["path"]}: sha256', sha(ROOT / m['path']), m['sha256'])

    # 9. no reserved image was ever embedded by any cached pipeline
    chk(G, 'no held-out embedding manifest in reports/dinov2/embeddings', len(list((DV / 'embeddings').glob('heldout*'))), 0)
    chk(G, 'data/dinov2/emb contains only dev/ and pilot/', sorted(p.name for p in (ROOT / 'data/dinov2/emb').iterdir()), ['dev', 'pilot'])
    pil_all = pd.read_csv(ROOT / 'data/mimo_pilot/sample_pilot.csv', dtype=str, keep_default_na=False)
    expect = {}
    for rows in (list(zip(dev.image_id, dev.sha256)), list(zip(pil_all.image_id, pil_all.sha256))):
        for s in range(0, len(rows), 16):
            expect.setdefault(s, set()).add(hashlib.sha256(canonical(rows[s:s + 16]).encode()).hexdigest())
    n_npz, unmatched = 0, 0
    for p in sorted((ROOT / 'data/dinov2/cache').glob('*/*.npz')):
        n_npz += 1
        key = str(np.load(p)['key'])
        if key not in expect.get(int(p.stem), set()):
            unmatched += 1
    chk(G, f'DINOv2 cache chunks ({n_npz}) whose input key is not a dev or pilot chunk', unmatched, 0)
    # pre-split exposures of reserved objects (identifiers only; no reserved labels are read)
    ho_ids = ids[ids.partition == 'heldout'].set_index('image_id')
    exposure = []
    for j in sorted((ROOT / 'data/mimo_pilot/embeddings').glob('*.json')):
        d = json.loads(j.read_text())
        for r in d.get('descriptor', {}).get('inputs', []):
            if r.get('image_id') in ho_ids.index:
                exposure.append({'id': ho_ids.loc[r['image_id'], 'id'], 'image_id': r['image_id'], 'partition': 'heldout',
                                 'exposure': 'CLIP ViT-B/32 32-image smoke test, 29 Oct 2026 (results/smoke/: embedded, clustered, HS1-3 AMI scored)',
                                 'cache': rel(j)})
    pd.DataFrame(exposure, columns=['id', 'image_id', 'partition', 'exposure', 'cache']).to_csv(ART / 'RESERVED_PRIOR_EXPOSURE.csv', index=False)
    chk(G, 'reserved heldout objects found in any local embedding cache (pre-split smoke test)', len(exposure), 0, ok=False if exposure else True,
        kind='finding' if exposure else 'gate',
        note='pre-split (29 Oct) smoke test; not caused by P0; ids in artifacts/RESERVED_PRIOR_EXPOSURE.csv; P5A must exclude them when it locks its split' if exposure else '')
    chk(G, 'teammate pre-split analyses on all 27,348 benchmark rows (IDEAS.md §Results; spikes/clean_analysis.py)', 'documented, artefacts not local', 'none', ok=False, kind='finding',
        note='Weeks 2-4 descriptive k-means/probe scores over the whole benchmark incl. today\'s reserved pool; the reserved pool is untouched since the 15 Nov split, not never-seen')

    # 10. frozen development manifest
    vocab = pd.read_csv(ROOT / 'data/mimo/hs_vocab.csv', dtype=str, keep_default_na=False)
    vl = dict(zip(vocab.notation, vocab.label))
    fallback = {'112': 'Indirectly struck idiophones (standard H-S name; not in local MIMO vocabulary)',
                '412': 'Interruptive free aerophones (standard H-S name; not in local MIMO vocabulary)',
                '531': 'Analogue electronic instruments (MIMO electrophone revision; label not in local vocabulary)'}
    hs1_names = {'1': 'Idiophones', '2': 'Membranophones', '3': 'Chordophones', '4': 'Aerophones', '5': 'Electrophones'}
    ordered_ids_sha = hashlib.sha256('\n'.join(dev.id).encode()).hexdigest()
    manifest = {
        'manifest_id': 'MIMO-HS3-DEV12K-v1',
        'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'frozen_by': 'roadmap phase P0 (reports/roadmap/phases/P0/source/p0_verify.py)',
        'status': 'FROZEN: every later phase must use exactly these rows in exactly this order; any change requires a new manifest_id',
        'source_file': {'path': rel(DEV_CSV), 'sha256': sha(DEV_CSV), 'bytes': DEV_CSV.stat().st_size, 'rows': len(dev),
                        'note': 'not copied: the validated Week 6 file is referenced by path + hash'},
        'built_by': 'reports/full_dataset/source/full_build.py (15 Nov 2026); split rule: np.sort(np.random.default_rng(0).choice(27347, 12000, replace=False)) over benchmark_objects.csv row order',
        'object_order': {'definition': 'row order of dev_ids.csv (= benchmark order restricted to dev); row_index 0..11999',
                         'ordered_ids_sha256': ordered_ids_sha, 'ordered_ids_sha256_rule': "sha256('\\n'.join(ids))",
                         'ordered_inputs_sha256': order_key,
                         'ordered_inputs_sha256_rule': "sha256(json.dumps([[image_id, sha256], ...], sort_keys=True, separators=(',',':'))) — the key every embedding manifest must carry",
                         'first_id': dev.id.iloc[0], 'last_id': dev.id.iloc[-1]},
        'image_to_object': {'rule': 'one Europeana record = one museum object = one image (first preview, else source image)',
                            'image_path': 'data/mimo_pilot/images/<image_id>.img (git-ignored, local only)', 'image_bytes_sha256_column': 'sha256',
                            'pixel_sha256_column': 'pixel_sha256', 'unique_image_ids': int(dev.image_id.nunique()), 'files_verified': 12000 - missing - bad},
        'labels': {
            'hs1': {'definition': 'first digit of the Hornbostel-Sachs notation of the first HS concept (common.hs_level rule)', 'classes': len(dev.hs1.unique()),
                    'counts': {k: {'name': hs1_names.get(k, ''), 'n': int(v)} for k, v in dev.hs1.value_counts().sort_index().items()}},
            'hs3': {'definition': 'first three digits of the same notation; benchmark keeps HS3/family/name classes with >= 30 objects',
                    'classes': len(hs3_classes),
                    'counts': {k: {'name': vl.get(k, fallback.get(k, '')), 'n': int(v)} for k, v in dev.hs3.value_counts().sort_index().items()}},
            'secondary': {lev: int(dev[lev].nunique()) for lev in ('hs2', 'family', 'name')}},
        'evaluation_masks': {lev: {'rule': f'row is scored for {lev} iff {lev} != ""', 'valid_rows': int((dev[lev] != '').sum()), 'invalid_rows': int((dev[lev] == '').sum())}
                             for lev in LABEL_LEVELS},
        'evaluation_mask_note': 'all 12,000 development rows carry valid HS1, HS2, HS3, family and name labels, so every mask is all-true; '
                                'the masks are defined by rule (not a file) so that any future subset applies the same rule',
        'museums': {'column': 'provider', 'n_museums': int(n_mus), 'top5': {k: int(v) for k, v in top5.items()},
                    'top5_share': round(float(top5.sum() / len(dev)), 4), 'per_museum_counts': 'reports/full_dataset/museum_distribution.csv (dev column, verified equal)'},
        'pilot_overlap': {'dev_rows_in_explored_pilot': int(ip[ids.partition == 'dev'].sum()),
                          'note': 'kept in dev by the original split rule; these rows were seen in the 2,000-image pilot (descriptive only)'},
        'separation': {'reserved_heldout': part.get('heldout', 0), 'excluded_pilot_overlap': part.get('excluded_pilot_overlap', 0),
                       'reserved_file': {'path': rel(HELDOUT_CSV), 'sha256': sha(HELDOUT_CSV)},
                       'overlap_with_reserved': {c: 0 for c in ('id', 'image_id', 'sha256', 'pixel_sha256')},
                       'reserved_rule': 'pool for optional P5A validation (Week 8 or 10) plus a separate protected Week 14 final test; P0 created no P5A split'},
        'embeddings_aligned': emb,
    }
    manifest['manifest_content_sha256'] = hashlib.sha256(canonical({k: v for k, v in manifest.items() if k != 'frozen_utc'}).encode()).hexdigest()
    (ART / 'DEV_MANIFEST.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    save_checks('dataset')


# ---------------------------------------------------------------------------------------------------------- baseline
def transform(X, rep):
    from sklearn.preprocessing import normalize
    if rep == 'centered_l2':
        return normalize(X - X.mean(axis=0))
    if rep == 'l2':
        return normalize(X)
    raise ValueError(rep)


def stage_baseline():
    G = 'baseline'
    dev = pd.read_csv(DEV_CSV, dtype=str, keep_default_na=False)
    runs = pd.read_csv(DV / 'dev/runs.csv'); summ = pd.read_csv(DV / 'dev/summary.csv')
    chk(G, 'dev runs.csv: fits / errors', f'{len(runs)}/{int((runs.status != "ok").sum())}', '330/0')
    chk(G, 'reports/dinov2/dev_benchmark.log exists', (ROOT / 'reports/dinov2/dev_benchmark.log').exists(), True)
    ckpt = {'clip': 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K@1a25a446', 'siglip': 'timm/ViT-B-16-SigLIP@41f57576',
            'dinov2_b_224_L1': 'timm/vit_base_patch14_dinov2.lvd142m@4685c99d (224 px, layer -1 CLS)'}
    hyper = {'kmeans': 'KMeans(n_init=10, max_iter=300, tol=1e-4, algorithm=lloyd, random_state=seed)',
             'agglomerative_average_cosine': 'AgglomerativeClustering(linkage=average, metric=cosine) — deterministic',
             'spectral_knn': 'SpectralClustering(affinity=nearest_neighbors, n_neighbors=15, assign_labels=kmeans, eigen_solver=arpack, n_init=10, random_state=seed)',
             'gmm_diag': 'GaussianMixture(covariance_type=diag, n_init=3, max_iter=200, tol=1e-3, reg_covar=1e-6, init_params=kmeans, random_state=seed), float64 input',
             'hdbscan': 'sklearn HDBSCAN(min_cluster_size=20, metric=euclidean, cluster_selection_method=eom) — deterministic, k not requested'}
    configs = []
    for v in BACKBONES:
        for m in ['kmeans', 'agglomerative_average_cosine', 'spectral_knn', 'gmm_diag', 'hdbscan']:
            configs.append(('primary_fixed_k10', v, 'centered_l2', m, None if m == 'hdbscan' else 10.0))
        for m in ['kmeans', 'agglomerative_average_cosine', 'spectral_knn', 'gmm_diag']:
            configs.append(('label_informed_k13_reference', v, 'centered_l2', m, 13.0))
    y = {lev: dev[lev].to_numpy() for lev in ('hs3', 'hs1')}
    sil_idx = np.random.default_rng(0).choice(12000, 4000, replace=False)
    rows, refits = [], []
    with threadpool_limits(limits=4):
        for v in BACKBONES:
            man = json.loads((DV / 'embeddings' / BACKBONES[v]).read_text())
            arr = man['arrays']['-1'] if 'arrays' in man else man['array']
            X = np.load(ROOT / arr['path']); assert sha(ROOT / arr['path']) == arr['sha256']
            Z = transform(X, 'centered_l2').astype(np.float32)
            D = pairwise_distances(Z[sil_idx], metric='cosine'); np.maximum(D, 0, out=D); np.fill_diagonal(D, 0)
            for role, vv, rep, m, k in [c for c in configs if c[1] == v]:
                g = runs[(runs.variant == v) & (runs.representation == rep) & (runs.method == m)]
                g = g[g.k.isna()] if k is None else g[g.k == k]
                sc = 'noise_as_cluster' if m == 'hdbscan' else 'all_rows'
                s = summ[(summ.variant == v) & (summ.representation == rep) & (summ.method == m) & (summ.scoring == sc)]
                s = s[s.k.isna()] if k is None else s[s.k == k]
                rep_hs3 = s[s.label == 'hs3'].iloc[0]; rep_hs1 = s[s.label == 'hs1'].iloc[0]
                met = {f'{lev}_{n}': [] for lev in ('hs3', 'hs1') for n in ('AMI', 'ARI', 'NMI')}
                sils, hash_ok, len_ok = [], True, True
                ao_ami, ao_cov = [], []
                for r in g.itertuples():
                    p = DV / 'dev/assignments' / f'{r.run_id}.npy'
                    hash_ok &= sha(p) == r.assignment_sha256
                    pred = np.load(p).astype(int); len_ok &= len(pred) == 12000
                    for lev in ('hs3', 'hs1'):
                        met[f'{lev}_AMI'].append(AMI(y[lev], pred)); met[f'{lev}_ARI'].append(ARI(y[lev], pred)); met[f'{lev}_NMI'].append(NMI(y[lev], pred))
                    ps = pred[sil_idx]; keep = ps != -1
                    sils.append(float(silhouette_score(D[np.ix_(keep, keep)], ps[keep], metric='precomputed')) if len(set(ps[keep])) > 1 else np.nan)
                    if m == 'hdbscan':
                        a = pred != -1; ao_ami.append(AMI(y['hs3'][a], pred[a])); ao_cov.append(float(a.mean()))
                mean = {kk: float(np.mean(vals)) for kk, vals in met.items()}
                sd = {kk: (float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan) for kk, vals in met.items()}
                d_ami = abs(mean['hs3_AMI'] - rep_hs3.AMI_mean); d_ari = abs(mean['hs3_ARI'] - rep_hs3.ARI_mean)
                d_nmi = abs(mean['hs3_NMI'] - rep_hs3.NMI_mean); d_sil = abs(float(np.nanmean(sils)) - rep_hs3.silhouette_cosine)
                d_h1 = abs(mean['hs1_AMI'] - rep_hs1.AMI_mean)
                tag = f'{v} {m} k{k}'
                chk(G, f'{tag}: assignment files ({len(g)}) sha256 == runs.csv, length 12000', bool(hash_ok and len_ok and len(g) == len(SEEDS if m in ("kmeans", "spectral_knn", "gmm_diag") else [0])), True)
                chk(G, f'{tag}: recomputed HS3 AMI/ARI/NMI == summary.csv (max |Δ|)', max(d_ami, d_ari, d_nmi), 0.0, ok=max(d_ami, d_ari, d_nmi) < 1e-9)
                chk(G, f'{tag}: recomputed HS1 AMI == summary.csv', d_h1, 0.0, ok=d_h1 < 1e-9)
                chk(G, f'{tag}: recomputed silhouette == summary.csv', d_sil, 0.0, ok=d_sil < 1e-6)
                rows.append({'baseline_id': f'B-{v}-{m}-k{"auto" if k is None else int(k)}', 'role': role, 'backbone': v, 'checkpoint': ckpt[v],
                             'embedding_file': arr['path'], 'embedding_sha256': arr['sha256'], 'dims': X.shape[1], 'representation': rep, 'method': m,
                             'hyperparameters': hyper[m], 'k_requested': 'none (density)' if k is None else int(k), 'seeds': ' '.join(str(int(s_)) for s_ in g.seed),
                             'n_seeds': len(g), 'n_objects': 12000, 'n_evaluated_hs3': int(rep_hs3.n_evaluated), 'hs3_classes': int(rep_hs3.classes),
                             'HS3_AMI_mean': mean['hs3_AMI'], 'HS3_AMI_sd': sd['hs3_AMI'], 'HS3_ARI_mean': mean['hs3_ARI'], 'HS3_ARI_sd': sd['hs3_ARI'],
                             'HS3_NMI_mean': mean['hs3_NMI'], 'HS1_AMI_mean': mean['hs1_AMI'], 'HS1_ARI_mean': mean['hs1_ARI'],
                             'silhouette_cosine_mean': float(np.nanmean(sils)), 'largest_cluster_pct_max': float(g.largest_cluster_pct.max()),
                             'assigned_clusters_mean': float(g.actual_k.mean()), 'noise_pct_mean': float(g.noise_pct.mean()),
                             'hdbscan_assigned_only_HS3_AMI': float(np.mean(ao_ami)) if ao_ami else np.nan, 'hdbscan_coverage': float(np.mean(ao_cov)) if ao_cov else np.nan,
                             'fit_seconds_mean_saved': float(g.fit_seconds.mean()), 'source_result_file': 'reports/dinov2/dev/summary.csv; runs.csv; scores.csv',
                             'assignment_files': 'reports/dinov2/dev/assignments/' + g.run_id.iloc[0].rsplit('__s', 1)[0] + '__s{seed}.npy',
                             'rescore_max_abs_diff': max(d_ami, d_ari, d_nmi, d_h1), 'silhouette_abs_diff': d_sil})
            # independent re-fit of the primary k=10 grid (and HDBSCAN) from the cached embedding
            for role, vv, rep, m, k in [c for c in configs if c[1] == v and c[0] == 'primary_fixed_k10']:
                g = runs[(runs.variant == v) & (runs.representation == rep) & (runs.method == m)]
                g = g[g.k.isna()] if k is None else g[g.k == k]
                for r in g.itertuples():
                    est = make(m, None if k is None else int(k), int(r.seed), 20)
                    Zf = Z.astype(np.float64) if m == 'gmm_diag' else Z
                    t0 = time.perf_counter(); pred = est.fit_predict(Zf).astype(np.int16); sec = time.perf_counter() - t0
                    saved = np.load(DV / 'dev/assignments' / f'{r.run_id}.npy')
                    same = bool(np.array_equal(pred, saved)); agree = float(ARI(saved, pred))
                    refits.append({'run_id': r.run_id, 'identical_labels': same, 'ARI_saved_vs_refit': agree,
                                   'HS3_AMI_refit': AMI(y['hs3'], pred), 'HS3_AMI_saved': AMI(y['hs3'], saved),
                                   'fit_seconds_refit': sec, 'fit_seconds_saved': float(r.fit_seconds), **{f'refit_{a}': b for a, b in sizes(pred.astype(int)).items()}})
                    print('refit', r.run_id, same, round(agree, 6), round(sec, 1), flush=True)
                    pd.DataFrame(refits).to_csv(ART / 'baseline_refit.csv', index=False)
    rf = pd.DataFrame(refits); rf.to_csv(ART / 'baseline_refit.csv', index=False)
    chk(G, 'k=10 baseline re-fits with labels identical to saved assignments', f'{int(rf.identical_labels.sum())}/{len(rf)}', f'{len(rf)}/{len(rf)}', ok=bool(rf.identical_labels.all()))
    out = pd.DataFrame(rows)
    rf['cfg'] = rf.run_id.str.rsplit('__s', n=1).str[0]
    stat = rf.groupby('cfg').agg(n=('identical_labels', 'size'), ident=('identical_labels', 'sum'), min_ari=('ARI_saved_vs_refit', 'min'),
                                 max_dami=('HS3_AMI_refit', lambda s: 0.0)).reset_index()
    status, disc, rcol = [], [], []
    for r in out.itertuples():
        cfg = r.assignment_files.rsplit('/', 1)[1].replace('__s{seed}.npy', '')
        sr = stat[stat.cfg == cfg]
        if len(sr):
            s0 = sr.iloc[0]
            rcol.append(f'{int(s0.ident)}/{int(s0.n)} identical (min ARI saved-vs-refit {s0.min_ari:.6f})')
            ok = r.rescore_max_abs_diff < 1e-9 and s0.ident == s0.n
            sub = rf[rf.cfg == cfg]; dmax = float((sub.HS3_AMI_refit - sub.HS3_AMI_saved).abs().max())
            disc.append('none' if ok else f'refit differs: max |ΔAMI| {dmax:.2e}')
            status.append('VERIFIED (rescored + re-fitted)' if ok else ('VERIFIED (rescored); refit NOT bit-identical' if r.rescore_max_abs_diff < 1e-9 else 'UNVERIFIED'))
        else:
            rcol.append('not re-fitted (label-informed reference; rescored only)')
            ok = r.rescore_max_abs_diff < 1e-9
            status.append('VERIFIED (rescored from saved assignments)' if ok else 'UNVERIFIED'); disc.append('none' if ok else 'rescore mismatch')
    out['refit_check'] = rcol; out['verification_status'] = status; out['discrepancy'] = disc
    # Week 6 headline numbers as presented
    for v, m, ami, ari in [('dinov2_b_224_L1', 'agglomerative_average_cosine', 0.561, 0.417), ('siglip', 'spectral_knn', 0.559, 0.311)]:
        r = out[(out.backbone == v) & (out.method == m) & (out.role == 'primary_fixed_k10')].iloc[0]
        chk(G, f'Week 6 headline {v} {m}: HS3 AMI/ARI rounds to {ami}/{ari}', f'{r.HS3_AMI_mean:.3f}/{r.HS3_ARI_mean:.3f}', f'{ami:.3f}/{ari:.3f}')
    out.to_csv(ART / 'BASELINE_REFERENCE.csv', index=False, float_format='%.6f')
    save_checks('baseline')


# ---------------------------------------------------------------------------------------------------------- protocol
def stage_protocol():
    G = 'protocol'
    dm = json.loads((ART / 'DEV_MANIFEST.json').read_text())
    models = json.loads((DV / 'models.json').read_text())['models']
    src = {rel(p): sha(p) for p in [ROOT / 'reports/week6/source/p2_benchmark.py', DV / 'source/dinov2_benchmark.py', DV / 'source/dinov2_embed.py',
                                     DV / 'source/openclip_embed.py', DV / 'source/dinov2_weights.py', Path(__file__)]}
    dino_in = lambda m: {'checkpoint': f"{models[m]['repo']}@{models[m]['revision']}", 'weight_sha256': models[m]['weight_sha256'],
                         'params_million': models[m]['params_million'], 'embedding_dim': models[m]['embed_dim'], 'depth': models[m]['depth'],
                         'native_resolution_timm_default': 518, 'protocol_resolution': 224,
                         'license_per_model_card': 'Apache-2.0 (Hugging Face API cardData.license, checked 2026-11-19; repo head revision == pinned revision)',
                         'weights_local': True}
    proto = {
        'protocol_id': 'FAIR-CMP-v1',
        'declared_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'declared_in': 'roadmap phase P0',
        'supersedes': None,
        'consistent_with': ['reports/dinov2/DINOV2_PROTOCOL.json (15 Nov 2026)', 'reports/week6/P1_EXPERIMENT_PROTOCOL.json', 'reports/week6/P3_EXPERIMENT_PROTOCOL.json'],
        'change_rule': 'Any change to a frozen field creates FAIR-CMP-v2 with a written reason; results under different protocol ids are not pooled.',
        'dataset': {
            'manifest_id': dm['manifest_id'], 'manifest_file': 'reports/roadmap/phases/P0/artifacts/DEV_MANIFEST.json',
            'manifest_content_sha256': dm['manifest_content_sha256'], 'dev_ids_csv': dm['source_file'],
            'object_order': dm['object_order'], 'n_objects': 12000,
            'image_to_object': dm['image_to_object']['rule'],
            'labels': {'primary': 'hs3 (13 classes)', 'secondary': ['hs1 (5 classes)', 'hs2', 'family (Western, 6)', 'name'],
                       'definitions': {'hs1': dm['labels']['hs1']['definition'], 'hs3': dm['labels']['hs3']['definition']}},
            'evaluation_masks': {'rule': 'score a row for label L iff L != "" in dev_ids.csv; clustering always uses all 12,000 rows',
                                 'current_valid_rows': {k: v['valid_rows'] for k, v in dm['evaluation_masks'].items()}},
            'museum_column': 'provider', 'n_museums': dm['museums']['n_museums'],
            'reserved_data': 'reserved (heldout + excluded_pilot_overlap) objects are never embedded, clustered, scored or used for selection under this protocol; '
                             'P5A must lock its own object-level validation split before touching them',
        },
        'embeddings': {
            'shared_input_adaptation': 'PIL: ImageOps.exif_transpose -> convert(RGB) -> ImageOps.pad to a white (255,255,255) square of side max(w,h); '
                                       'then the checkpoint\'s own evaluation transform. Because the image is already square, resize+center-crop (timm, OpenCLIP) and '
                                       'squash (SigLIP) all reduce to an aspect-preserving full-image resize: no part of the instrument is cropped.',
            'storage': 'float32 .npy, one row per manifest row in manifest order, under data/<family>/emb/dev/ (git-ignored); a tracked JSON manifest records '
                       'checkpoint, revision, weight sha256, transform repr, array sha256, shape and ordered_inputs_sha256',
            'alignment_gate': f"ordered_inputs_sha256 must equal {dm['object_order']['ordered_inputs_sha256']}; arrays must be finite with shape (12000, d)",
            'device_precision': 'CPU, torch float32 inference (no autocast/fp16); torch.manual_seed(42); 4-6 torch threads (inference is deterministic per input; threads affect speed only)',
            'models': {
                'clip': {'checkpoint': 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K@1a25a446712ba5ee05982a381eed697ef9b435cf', 'library': 'open_clip_torch 3.3.0 (arch ViT-B-32)',
                         'weight_sha256': 'ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6', 'embedding_dim': 512, 'native_resolution': 224, 'protocol_resolution': 224,
                         'transform': 'OpenCLIP eval: resize shortest side 224 (bicubic) + center crop 224; mean (0.48145466, 0.4578275, 0.40821073), std (0.26862954, 0.26130258, 0.27577711)',
                         'output': 'encode_image() projected image embedding, L2-normalised at extraction', 'array': dm['embeddings_aligned']['clip'],
                         'license_per_model_card': 'MIT (Hugging Face API cardData.license, checked 2026-11-19; repo head revision == pinned revision)'},
                'siglip': {'checkpoint': 'timm/ViT-B-16-SigLIP@41f575766f40e752fdd1383e9565b7f02388c1c4', 'library': 'open_clip_torch 3.3.0 (arch SiglipPhase2 = vit_base_patch16_siglip_224, MAP pool)',
                           'weight_sha256': '81942ea8c09b9e41963357cca5d2682118bf7eb7491c2ae51a28bc4f4f95b194', 'embedding_dim': 768, 'native_resolution': 224, 'protocol_resolution': 224,
                           'transform': 'resize_mode squash to 224x224 (bicubic); mean (0.5,0.5,0.5), std (0.5,0.5,0.5)',
                           'output': 'encode_image() pooled image embedding, L2-normalised at extraction', 'array': dm['embeddings_aligned']['siglip'],
                           'license_per_model_card': 'Apache-2.0 (Hugging Face API cardData.license, checked 2026-11-19; repo head revision == pinned revision)'},
                'dinov2_b': {**dino_in('dinov2_b'), 'library': 'timm 1.0.30', 'transform': 'timm eval transform with input_size 224, crop_pct 1.0: Resize(224, bicubic) + CenterCrop(224); ImageNet mean/std',
                             'output': 'CLS token after final LayerNorm (forward_intermediates norm=True); layer -1 == pooled output (asserted at run time); not L2-normalised at extraction',
                             'array': dm['embeddings_aligned']['dinov2_b_224_L1'], 'other_layers_cached': ['-2', '-4']},
                'dinov2_s': {**dino_in('dinov2_s'), 'library': 'timm 1.0.30', 'status': 'P1 to extract on dev; pilot-only so far'},
                'dinov2_l': {**dino_in('dinov2_l'), 'library': 'timm 1.0.30', 'status': 'P1 to extract on dev; pilot-only so far'},
            },
            'rule_for_new_encoders': ['same 12,000 rows, same order, same alignment gate',
                                      'same shared input adaptation, then the checkpoint\'s own evaluation transform at its NATIVE resolution; record native and used resolution separately',
                                      'image-only forward pass; record exactly which output is used (pooled/projected embedding for VLM encoders, final-LN CLS for SSL ViTs, documented pooling for embedding LLMs)',
                                      'record checkpoint id + revision, weight sha256, licence, embedding dim, dtype, seconds, threads, peak RSS',
                                      'DINOv2 S/B/L (P1) are compared at 224 px so that only model size changes; 518 px is a separate, labelled resolution arm',
                                      'model families are not assumed to share native preprocessing; a model whose native transform crops (crop_pct<1) still receives the white-square-padded image first'],
        },
        'clustering': {
            'primary_controlled_reference': {'representation': 'centered_l2', 'k': 10, 'methods': ['kmeans', 'agglomerative_average_cosine', 'spectral_knn', 'gmm_diag', 'hdbscan'],
                                             'primary_metric': 'HS3 AMI (mean over seeds); ARI reported with equal prominence'},
            'k_policy': {'fixed_k': 10, 'fixed_k_note': 'predeclared anchor, not tuned on labels',
                         'label_informed_reference': 'k = 13 (= number of HS3 benchmark classes); reported in a separate table, never called blind',
                         'blind_k': 'P3/P12: select k from label-free criteria only (silhouette, elbow, eigengap, BIC) on the same rows; HS labels are read only after k is fixed'},
            'representations': {'centered_l2': 'X - mean(X over the 12,000 clustered rows), then row L2 (primary)',
                                'l2': 'row L2 only (control; collapses VLM agglomerative)', 'pca50_l2 / pca100_l2': "PCA(n, svd_solver='full', whiten=False) fitted on the clustered rows, then row L2",
                                'fit_scope': 'every transform is fitted on the clustered rows only, label-free; features cast to float32 (GMM to float64)'},
            'methods': {'kmeans': {'class': 'sklearn.cluster.KMeans', 'n_init': 10, 'max_iter': 300, 'tol': 1e-4, 'algorithm': 'lloyd', 'init': 'k-means++', 'distance': 'euclidean (= cosine ranking on L2 rows)'},
                        'agglomerative_average_cosine': {'class': 'sklearn.cluster.AgglomerativeClustering', 'linkage': 'average', 'metric': 'cosine', 'deterministic': True},
                        'spectral_knn': {'class': 'sklearn.cluster.SpectralClustering', 'affinity': 'nearest_neighbors', 'n_neighbors': 15, 'assign_labels': 'kmeans', 'eigen_solver': 'arpack', 'n_init': 10, 'n_jobs': 1},
                        'gmm_diag': {'class': 'sklearn.mixture.GaussianMixture', 'covariance_type': 'diag', 'n_init': 3, 'max_iter': 200, 'tol': 1e-3, 'reg_covar': 1e-6, 'init_params': 'kmeans', 'input_dtype': 'float64'},
                        'hdbscan': {'class': 'sklearn.cluster.HDBSCAN', 'min_cluster_size': 20, 'metric': 'euclidean', 'cluster_selection_method': 'eom', 'deterministic': True}},
            'factory': 'reports/week6/source/p2_benchmark.py make(method, k, seed, mcs=20)',
            'seeds': {'stochastic_methods': SEEDS, 'deterministic_methods': [0], 'embedding_extraction_torch_seed': 42, 'silhouette_sample_seed': 0},
            'numerics': {'OMP_NUM_THREADS': 4, 'threadpoolctl_limit': 4, 'note': '4 BLAS threads are part of the protocol: 6 threads changed PCA results by up to 2.3e-5 in P5',
                         'environment': 'requirements-pilot.lock.txt (Python 3.12.3, numpy 2.5.4, scikit-learn 1.9.1, torch 2.15.0+cpu, timm 1.0.30, open_clip_torch 3.3.0)'},
        },
        'evaluation': {
            'external_metrics': {'AMI': 'sklearn adjusted_mutual_info_score(labels, pred), average_method=arithmetic (default)',
                                 'ARI': 'sklearn adjusted_rand_score(labels, pred)', 'NMI': 'sklearn normalized_mutual_info_score(labels, pred), arithmetic (default)',
                                 'levels': 'HS3 primary; HS1, HS2, family, name secondary'},
            'internal_metrics': {'silhouette': 'cosine silhouette on the clustered representation, on a fixed 4,000-row sample: np.random.default_rng(0).choice(12000, 4000, replace=False); HDBSCAN noise rows excluded',
                                 'largest_cluster_pct': '100 * size of largest non-noise cluster / 12,000 (report max over seeds)',
                                 'assigned_clusters': 'number of distinct non-noise labels (actual_k)', 'noise_fraction': 'share of rows labelled -1 (HDBSCAN only; 0 otherwise)',
                                 'runtime': 'wall-clock seconds of fit_predict only (time.perf_counter), 4 threads; transform time recorded separately; embedding time per model in its manifest'},
            'hdbscan_noise': {'primary': 'noise_as_cluster: all 12,000 rows scored, label -1 treated as one extra cluster',
                              'secondary': 'assigned_only: score only rows with label != -1 and report coverage; never quote assigned_only without coverage'},
            'invalid_labels': 'rows with an empty label at level L are clustered but excluded from that level\'s external metrics; n_evaluated is always reported (currently 12,000 at every level)',
            'seed_aggregation': 'mean over seeds 0-4 and sample SD (ddof=1) for AMI and ARI; min/max kept in per-run files; largest_cluster_pct reported as the worst (max) seed; '
                                'deterministic methods have n_seeds = 1 and no SD. Seed-to-seed partition agreement (pairwise ARI) is reported in P3.',
            'statistics': 'no significance claims without paired seeds or bootstrap over objects; any difference < 2 x seed SD is reported as no clear difference',
            'leakage': 'labels are read only after every fit of a grid has been saved (fit stage then score stage)',
        },
        'baselines': {'file': 'reports/roadmap/phases/P0/artifacts/BASELINE_REFERENCE.csv', 'status': 'immutable references for every later comparison'},
        'source_sha256': src,
    }
    proto['protocol_content_sha256'] = hashlib.sha256(canonical({k: v for k, v in proto.items() if k != 'declared_utc'}).encode()).hexdigest()
    (ART / 'EXPERIMENT_PROTOCOL.json').write_text(json.dumps(proto, indent=2, ensure_ascii=False))
    # internal consistency
    chk(G, 'protocol k == DINOV2_PROTOCOL.json k', proto['clustering']['primary_controlled_reference']['k'], json.loads((DV / 'DINOV2_PROTOCOL.json').read_text())['k'])
    chk(G, 'protocol methods == methods in BASELINE_REFERENCE.csv', sorted(proto['clustering']['methods']), sorted(pd.read_csv(ART / 'BASELINE_REFERENCE.csv').method.unique()))
    chk(G, 'protocol order key == every aligned embedding manifest key', all(json.loads((DV / 'embeddings' / f).read_text())['ordered_inputs_sha256'] == dm['object_order']['ordered_inputs_sha256'] for f in BACKBONES.values()), True)
    for m in ('dinov2_s', 'dinov2_b', 'dinov2_l'):
        w = Path(models[m]['weight_path'])
        chk(G, f'{m} weights present locally and sha256 pinned', w.exists() and sha(w) == models[m]['weight_sha256'], True)
    for name, f in (('clip', 'data/mimo_pilot/model/model.json'), ('siglip', 'data/phase2/siglip_model/model.json')):
        info = json.loads((ROOT / f).read_text())
        stored = Path(info['weight_path']); local = next((ROOT / 'data').glob(f"**/snapshots/{info['revision']}/open_clip_model.safetensors"), None)
        chk(G, f'{name} weights present locally and sha256 pinned', bool(local) and sha(local) == info['weight_sha256'], True)
        chk(G, f'{name} model.json weight_path resolves (needed only to re-embed)', stored.exists(), True, ok=stored.exists(), kind='info',
            note='informational: path points to the pre-move checkout; openclip_embed.py would need the path remapped before any CLIP/SigLIP re-embedding')
    save_checks('protocol')


# ------------------------------------------------------------------------------------------------------------ report
def stage_report():
    frames = [pd.read_csv(f) for f in sorted(ART.glob('checks_*.csv'))]
    c = pd.concat(frames, ignore_index=True)
    c.to_csv(ART / 'p0_checks.csv', index=False)
    env = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'python': sys.version.split()[0], 'executable': sys.executable,
           'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
           'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'pandas', 'torch', 'timm', 'open_clip_torch', 'threadpoolctl', 'pillow']}}
    (ART / 'environment.json').write_text(json.dumps(env, indent=2))
    gate = c[c.kind == 'gate']
    summary = {'checks': len(c), 'gates': len(gate), 'gates_passed': int(gate['pass'].sum()), 'gates_failed': int((~gate['pass']).sum()),
               'info': int((c.kind == 'info').sum()), 'findings': int((c.kind == 'finding').sum()),
               'p0_gate_status': 'PASS' if bool(gate['pass'].all()) else 'FAIL',
               'gates_by_group': {g: f"{int(d['pass'].sum())}/{len(d)}" for g, d in gate.groupby('group')},
               'artifact_sha256': {f.name: sha(f) for f in sorted(ART.iterdir()) if f.is_file() and f.suffix in ('.json', '.csv') and f.name not in ('p0_summary.json',)}}
    (ART / 'p0_summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# P0 validation evidence (generated)', '',
             f"Generated by `reports/roadmap/phases/P0/source/p0_verify.py` at {env['utc']} with Python {env['python']} "
             f"(scikit-learn {env['packages']['scikit-learn']}, numpy {env['packages']['numpy']}), OMP_NUM_THREADS={env['OMP_NUM_THREADS']}.", '',
             f"**P0 gates: {summary['gates_passed']} / {summary['gates']} passed, {summary['gates_failed']} failed.** "
             f"Also recorded: {summary['info']} informational values and {summary['findings']} findings (pre-existing issues, documented and handed on; they are not P0 gates).", '',
             'Kinds: **gate** = must pass for P0 to be complete; **info** = recorded value; **finding** = pre-existing issue that P0 cannot fix without changing the split, documented for P5A/P16.', '',
             '| Group | Gates passed |', '|---|---|'] + [f'| {g} | {v} |' for g, v in summary['gates_by_group'].items()] + ['']
    for g, d in c.rename(columns={'pass': 'ok'}).groupby('group', sort=False):
        lines += [f'## {g}', '', '| # | Kind | Check | Observed | Expected | Result |', '|---|---|---|---|---|---|']
        for i, r in enumerate(d.itertuples(), 1):
            o = str(r.observed).replace('|', '\\|'); e = str(r.expected).replace('|', '\\|')
            if len(o) > 70: o = o[:67] + '…'
            if len(e) > 70: e = e[:67] + '…'
            res = ('PASS' if r.ok else 'FAIL') if r.kind == 'gate' else ('recorded' if r.kind == 'info' else 'FINDING')
            if isinstance(r.note, str) and r.note: res += f' — {r.note}'
            lines.append(f'| {i} | {r.kind} | {r.check} | {o} | {e} | {res} |')
        lines.append('')
    (ART / 'P0_VALIDATION.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k != 'artifact_sha256'}, indent=1))


if __name__ == '__main__':
    ART.mkdir(parents=True, exist_ok=True)
    stages = sys.argv[1:] or ['dataset', 'baseline', 'protocol', 'report']
    for s in stages:
        CHECKS.clear()
        t = time.time(); globals()[f'stage_{s}'](); print(f'== stage {s} done in {time.time() - t:.0f}s', flush=True)
