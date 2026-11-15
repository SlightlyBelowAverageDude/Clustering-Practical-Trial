"""Audit the saved benchmark against the actual images, cache and predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

from pilot_acquire import DATA, ROOT, api_key, atomic_json, digest, inspect_image, now
from pilot_benchmark import canonical, sha_file


def validate(tag):
    out = ROOT / 'results' / tag
    sample = pd.read_csv(DATA / f'sample_{tag}.csv', dtype=str, keep_default_na=False)
    order = pd.read_csv(out / 'image_order.csv', dtype=str, keep_default_na=False)
    scores = pd.read_csv(out / 'scores.csv')
    assignments = pd.read_csv(out / 'assignments.csv', dtype={'image_id': str})
    manifest = json.loads((out / 'embedding_manifest.json').read_text())
    descriptor = manifest['descriptor']
    assert digest(canonical(descriptor).encode()) == manifest['input_key']
    assert list(order.image_id) == list(sample.image_id)
    assert list(order.id) == list(sample.id)
    assert list(order.sha256) == list(sample.sha256)
    assert list(order.row.astype(int)) == list(range(len(sample)))
    assert sample.id.is_unique and sample.image_id.is_unique and sample.pixel_sha256.is_unique
    for i, row in enumerate(sample.to_dict('records')):
        raw = (DATA / row['file']).read_bytes()
        assert digest(raw) == row['sha256']
        w, h, ph = inspect_image(raw)
        assert ph == row['pixel_sha256'] and w == int(row['width']) and h == int(row['height'])
        assert descriptor['inputs'][i] == {k: row[k] for k in ['id', 'image_id', 'sha256']}
    array_path = Path(manifest['array_path'])
    assert sha_file(array_path) == manifest['array_sha256']
    X = np.load(array_path, allow_pickle=False)
    assert X.shape == (len(sample), 512) and np.isfinite(X).all()
    assert np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-4)
    assert sha_file(manifest['model']['weight_path']) == manifest['model']['weight_sha256']
    for run_id, sub in scores.groupby('run_id'):
        pred = np.load(out / (run_id + '.npy'), allow_pickle=False)
        rows = assignments[assignments.run_id == run_id].sort_values('row')
        assert len(pred) == len(sample) == len(rows)
        assert list(rows.image_id) == list(sample.image_id)
        assert np.array_equal(rows.cluster.to_numpy(), pred)
        assert len(set(pred)) == sub.actual_k.iloc[0] == sub.requested_k.iloc[0]
        for metric_row in sub.to_dict('records'):
            mask = sample[metric_row['label']].ne('').to_numpy()
            y, p = sample.loc[mask, metric_row['label']].to_numpy(), pred[mask]
            assert mask.sum() == metric_row['n_evaluated']
            assert len(set(y)) == metric_row['label_classes']
            if len(set(y)) < 2:
                continue
            for name, function in [('AMI', adjusted_mutual_info_score), ('ARI', adjusted_rand_score), ('NMI', normalized_mutual_info_score)]:
                assert np.isclose(function(y, p), metric_row[name], atol=1e-12)
    # Check every saved metadata/result text file for the configured secret.
    secret = api_key().encode()
    scanned = 0
    paths = (list((ROOT / 'data').rglob('*')) + list(out.rglob('*'))
             + list((ROOT / 'spikes').glob('*.py')) + list((ROOT / 'tests').glob('*.py'))
             + list(ROOT.glob('*.md')) + list(ROOT.glob('*.txt')))
    for path in paths:
        if not path.is_file() or path.suffix not in {'.csv', '.json', '.md', '.html', '.py', '.log'} or 'model' in path.relative_to(ROOT).parts:
            continue
        assert secret not in path.read_bytes(), 'Credential detected in ' + str(path.relative_to(ROOT))
        scanned += 1
    # Gallery photos must stay in data/, not embedded in committed HTML.
    from html.parser import HTMLParser
    class Images(HTMLParser):
        def handle_starttag(self, tag_name, attrs):
            if tag_name == 'img':
                src = dict(attrs)['src']
                p = (out / src).resolve()
                assert p.is_relative_to(DATA) and p.is_file()
    Images().feed((out / 'cluster_gallery.html').read_text())
    checks = {'validated_utc': now(), 'success': True, 'images_redecoded_and_hashed': len(sample), 'unique_records': len(sample), 'identical_order_verified_runs': scores.run_id.nunique(), 'metric_rows_independently_recomputed': len(scores), 'source_and_model_hashes_verified': True, 'credential_scan_text_files': scanned, 'credential_found': False, 'gallery_images_inside_ignored_data': True, 'input_key': manifest['input_key']}
    atomic_json(out / 'validation.json', checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='pilot')
    validate(ap.parse_args().tag)
