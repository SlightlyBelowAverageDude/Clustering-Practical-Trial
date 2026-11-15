"""Reconstruct the cleaned MIMO dataset and the HS3/family/name benchmark with the teammate's own scripts.

1. runs/mimo_full/sample_full.csv: every harvested record (teammate CSV format, data/mimo/records.csv + hs_vocab.csv)
   whose image was validated by full_acquire.py; file points to the content-addressed image (path relative to data/mimo).
   Uncoded records are kept, as documented in IDEAS.md/mimo_full.py docstring (the committed mimo_full.py filter keeps
   only coded rows; the documented 59,661/40,697 counts require keeping them).
2. spikes/mimo_names.py (unchanged)   -> runs/mimo_names_v2/labels.csv  (needs dl/ tables from wikidata_names.py)
3. spikes/mimo_clean.py (unchanged)   -> runs/clean/records.csv, exclusions.csv
4. Benchmark ("same rows", rule copied from spikes/clean_analysis.py step 2): hs3, family and name all present, then
   classes with >= 30 objects for hs3, family, name in that order.
5. Split: development = the 12,000-row subset drawn exactly as clean_analysis.py step 6 (np.random.default_rng(0),
   sorted); held-out = the remaining benchmark rows MINUS records already explored in the 2,000-image pilot.
   Held-out ids are written but never embedded or scored in Week 6/P5.
Outputs: reports/full_dataset/benchmark/*.csv (ids, labels, split; no image bytes), manifest.json, stats.json.
"""
import hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PY = sys.executable
OUT = ROOT / 'reports/full_dataset'
BM = OUT / 'benchmark'
MIN = 30


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(script, *args):
    t = time.time()
    r = subprocess.run([PY, str(ROOT / 'spikes' / script), *args], cwd=ROOT, capture_output=True, text=True)
    log = OUT / 'logs' / (script.replace('.py', '') + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(r.stdout + '\n' + r.stderr)
    assert r.returncode == 0, f'{script} failed, see {log}'
    print(f'{script}: ok ({time.time() - t:.0f}s)\n' + r.stdout.strip()[-1500:], flush=True)


def main():
    BM.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(ROOT / 'data/mimo/records.csv', dtype=str, keep_default_na=False)
    voc = pd.read_csv(ROOT / 'data/mimo/hs_vocab.csv', dtype=str, keep_default_na=False)[['hs_uri', 'notation']]
    led = pd.read_csv(ROOT / 'data/mimo_full/download_ledger.csv', dtype=str, keep_default_na=False)
    meta = pd.read_csv(ROOT / 'data/mimo_pilot/metadata.csv', dtype=str, keep_default_na=False, usecols=['id', 'hs1', 'hs2', 'hs3', 'hs_multiple'])
    assert rec.id.is_unique and len(rec) == len(meta) and (rec.id.values == meta.id.values).all()
    df = rec.merge(voc, on='hs_uri', how='left').fillna({'notation': ''})
    df = df.merge(led[['id', 'status', 'image_id', 'file', 'sha256', 'pixel_sha256', 'width', 'height']], on='id', how='left').fillna('')
    stats = {'harvested_records': len(rec), 'records_with_preview_url': int((rec.preview != '').sum()), 'records_with_hs_uri': int((rec.hs_uri != '').sum()),
             'download_status': df.status.replace('', 'not_attempted').value_counts().to_dict()}
    ok = df[df.status == 'ok'].copy()
    ok['file'] = '../mimo_pilot/' + ok.file  # mimo_clean.py hashes ROOT/data/mimo/<file>
    stats['valid_images'] = len(ok); stats['distinct_pixel_hashes'] = int(ok.pixel_sha256.nunique())
    (ROOT / 'runs/mimo_full').mkdir(parents=True, exist_ok=True)
    ok.to_csv(ROOT / 'runs/mimo_full/sample_full.csv', index=False)
    run('mimo_names.py')
    run('mimo_clean.py')
    clean = pd.read_csv(ROOT / 'runs/clean/records.csv', dtype=str, keep_default_na=False)
    excl = pd.read_csv(ROOT / 'runs/clean/exclusions.csv', dtype=str, keep_default_na=False)
    # audit the placeholder rule: mimo_clean labels the single most frequent image hash as the Europeana placeholder
    ok['md5'] = [hashlib.md5((ROOT / 'data/mimo' / f).read_bytes()).hexdigest() for f in ok.file]
    top = ok.md5.value_counts()
    ph = ok[ok.md5 == top.index[0]]
    stats['placeholder_audit'] = {'top_hash_count': int(top.iloc[0]), 'second_hash_count': int(top.iloc[1]), 'top_hash_titles_sample': ph.title.head(5).tolist(),
                                  'top_hash_image_size': f"{ph.width.iloc[0]}x{ph.height.iloc[0]}"}
    stats['exclusions'] = excl.exclude.value_counts().to_dict()
    stats['cleaned_photographs'] = len(clean)
    stats['clean_coded_hs1'] = int((clean.hs1 != '').sum()); stats['clean_hs3'] = int((clean.hs3 != '').sum())
    stats['clean_named'] = int((clean.name != '').sum()); stats['clean_with_family'] = int((clean.family != '').sum())
    stats['clean_museums'] = int(clean.provider.nunique())
    # benchmark rows (clean_analysis.py step 2, verbatim rule)
    m = (clean.hs3 != '') & (clean.family != '') & (clean.name != '')
    sub = clean[m]
    for col in ('hs3', 'family', 'name'):
        vc = sub[col].value_counts(); sub = sub[sub[col].map(vc) >= MIN]
    sub = sub.copy()
    stats['benchmark_rows'] = len(sub)
    stats['benchmark_classes'] = {c: int(sub[c].nunique()) for c in ('hs1', 'hs2', 'hs3', 'hs4', 'hs5', 'hs6', 'family', 'name')}
    stats['benchmark_rows_with_level'] = {c: int((sub[c] != '').sum()) for c in ('hs4', 'hs5', 'hs6')}
    stats['benchmark_museums'] = int(sub.provider.nunique())
    # split (clean_analysis.py step 6 subset = development)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(np.arange(len(sub)), min(12000, len(sub)), replace=False))
    sub['partition'] = 'heldout'
    sub.iloc[pick, sub.columns.get_loc('partition')] = 'dev'
    pilot = set(pd.read_csv(ROOT / 'data/mimo_pilot/sample_pilot.csv', dtype=str, usecols=['id']).id)
    sub['in_explored_pilot'] = sub.id.isin(pilot)
    sub.loc[(sub.partition == 'heldout') & sub.in_explored_pilot, 'partition'] = 'excluded_pilot_overlap'
    img = ok.set_index('id')[['image_id', 'sha256', 'pixel_sha256', 'width', 'height']]
    sub = sub.join(img, on='id')
    sub['file'] = 'images/' + sub.image_id + '.img'
    # leakage checks: identical pixels across partitions
    px = sub.groupby('pixel_sha256').partition.nunique()
    stats['pixel_duplicates_across_partitions'] = int((px > 1).sum())
    stats['partition_counts'] = sub.partition.value_counts().to_dict()
    stats['pilot_overlap_in_benchmark'] = int(sub.in_explored_pilot.sum())
    cols = ['id', 'partition', 'provider', 'title', 'notation', 'hs1', 'hs2', 'hs3', 'hs4', 'hs5', 'hs6', 'kw', 'name', 'family', 'region',
            'image_id', 'sha256', 'pixel_sha256', 'width', 'height', 'file', 'rights', 'in_explored_pilot']
    sub[cols].to_csv(BM / 'benchmark_objects.csv', index=False)
    sub[sub.partition == 'dev'][cols].to_csv(BM / 'dev_ids.csv', index=False)
    sub[sub.partition != 'dev'][['id', 'partition', 'image_id', 'sha256']].to_csv(BM / 'heldout_ids_DO_NOT_USE_FOR_SELECTION.csv', index=False)
    clean.to_csv(OUT / 'clean_records.csv', index=False); excl.to_csv(OUT / 'exclusions.csv', index=False)
    # distributions
    dist = []
    for scope, d in [('clean', clean), ('benchmark', sub), ('dev', sub[sub.partition == 'dev'])]:
        for col in ('hs1', 'hs2', 'hs3', 'family'):
            for k, v in d[col].replace('', '(missing)').value_counts().items():
                dist.append({'scope': scope, 'level': col, 'class': k, 'n': int(v)})
    pd.DataFrame(dist).to_csv(OUT / 'class_distributions.csv', index=False)
    mus = pd.DataFrame({'clean': clean.provider.value_counts(), 'benchmark': sub.provider.value_counts(), 'dev': sub[sub.partition == 'dev'].provider.value_counts()}).fillna(0).astype(int)
    mus.index.name = 'museum'; mus.sort_values('clean', ascending=False).to_csv(OUT / 'museum_distribution.csv')
    stats['hs3_benchmark_counts'] = sub.hs3.value_counts().to_dict()
    stats['reference_counts_from_plan'] = {'original_records': 63478, 'cleaned_photographs': 59661, 'objects_with_hs_code': 40697, 'benchmark_objects': 27348, 'benchmark_hs3_classes': 13}
    stats['built_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    manifest = {'files': {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(list(BM.glob('*.csv')) + [OUT / 'clean_records.csv', OUT / 'exclusions.csv'])},
                'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'data/mimo/records.csv', ROOT / 'data/mimo/hs_vocab.csv', ROOT / 'data/mimo_full/download_ledger.csv', ROOT / 'dl/wd_mimo_instruments.csv', ROOT / 'dl/wd_mimo_ancestors.csv']},
                'image_store': {'location': 'data/mimo_pilot/images/<image_id>.img (local only, git-ignored)', 'valid_images': len(ok),
                                'total_bytes': int(sum((ROOT / 'data/mimo_pilot/images' / (i + '.img')).stat().st_size for i in ok.image_id)),
                                'per_image_sha256': 'data/mimo_full/download_ledger.csv (sha256 column); benchmark rows also in benchmark_objects.csv'}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (OUT / 'stats.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k not in ('hs3_benchmark_counts',)}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
