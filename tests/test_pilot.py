"""Tests for research-validity and credential-safety failure cases."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'spikes'))
import pilot_acquire as acq
import pilot_benchmark as bench


def test_no_credentials_in_cached_response():
    secret = 'synthetic-secret-for-test'
    data = {'apikey': secret, 'items': [{'id': '/test', 'url': 'https://example.org/a?wskey=' + secret + '&type=IMAGE', 'text': secret}]}
    result = acq.clean(data, secret)
    assert secret not in json.dumps(result)
    assert 'wskey' not in json.dumps(result)
    assert result['items'][0]['url'] == 'https://example.org/a?type=IMAGE'


def test_taxonomy_modifiers_and_conflicting_codes(tmp_path, monkeypatch):
    assert acq.hs_prefix('321.322-71', 3) == '321'
    assert acq.hs_prefix('422.112+422.22-62', 3) == '422'
    assert acq.hs_prefix('not-a-code', 1) == ''
    assert acq.hs_prefix('4', 3) == ''
    monkeypatch.setattr(acq, 'DATA', tmp_path)
    records = [{'hs_uris': '["a", "b"]'}, {'hs_uris': '["a", "c"]'}, {'hs_uris': '["unknown"]'}]
    rows = acq.annotate(records, {'a': {'notation': '321.322'}, 'b': {'notation': '322.1'}, 'c': {'notation': '421.1'}})
    assert rows[0]['hs1'] == '3' and rows[0]['hs2'] == '32' and rows[0]['hs3'] == ''
    assert rows[1]['hs1'] == '' and rows[2]['hs1'] == ''


def test_sampling_invariant_to_metadata_row_order():
    rows = [{'id': str(i), 'museum': 'AB'[i % 2]} for i in range(30)]
    order = acq.candidate_order(rows, 42)
    assert order == acq.candidate_order(list(reversed(rows)), 42)
    assert len({r['id'] for r in order}) == 30
    assert [r['museum'] for r in order[:4]] == ['A', 'B', 'A', 'B']


def test_pixel_dedup_ignores_file_encoding():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8))
    a, b = io.BytesIO(), io.BytesIO()
    image.save(a, format='PNG', compress_level=0)
    image.save(b, format='PNG', compress_level=9)
    assert acq.digest(a.getvalue()) != acq.digest(b.getvalue())
    assert acq.inspect_image(a.getvalue()) == acq.inspect_image(b.getvalue())
    with pytest.raises(Exception):
        acq.inspect_image(b'<html>not an image</html>')


def test_embedding_cache_rejects_missing_wrong_or_corrupt_provenance(tmp_path):
    path, meta = tmp_path / 'vector.npy', tmp_path / 'vector.json'
    desc = {'image_id': 'a', 'sha256': 'source-content', 'model_fingerprint': 'frozen-model'}
    x = np.zeros(512, dtype=np.float32)
    x[0] = 1
    bench.save_array(path, x)
    assert bench.valid_vector(path, meta, desc) is None
    acq.atomic_json(meta, {'descriptor': desc, 'array_sha256': bench.sha_file(path)})
    assert np.array_equal(bench.valid_vector(path, meta, desc), x)
    assert bench.valid_vector(path, meta, {**desc, 'sha256': 'changed-content'}) is None
    assert bench.valid_vector(path, meta, {**desc, 'model_fingerprint': 'other-model'}) is None
    path.write_bytes(b'corrupt')
    assert bench.valid_vector(path, meta, desc) is None


def test_ordered_input_hash_changes_when_rows_reordered():
    a = [{'id': 'a', 'sha256': '1'}, {'id': 'b', 'sha256': '2'}]
    assert acq.digest(bench.canonical(a).encode()) != acq.digest(bench.canonical(a[::-1]).encode())


def test_rate_limit_retried_and_exception_url_not_exposed(monkeypatch):
    waits = []
    class Response:
        def __init__(self, status):
            self.status_code = status
            self.headers = {'Retry-After': '3'}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def json(self):
            return {'success': True}
    responses = iter([Response(429), Response(200)])
    monkeypatch.setattr(acq.requests, 'get', lambda *a, **kw: next(responses))
    monkeypatch.setattr(acq.time, 'sleep', waits.append)
    assert acq.get('https://example.org')['success']
    assert 3 in waits
    def fail(*args, **kwargs):
        raise acq.requests.ConnectionError('https://example.org?wskey=private-test-value')
    monkeypatch.setattr(acq.requests, 'get', fail)
    with pytest.raises(RuntimeError) as error:
        acq.get('https://example.org', attempts=2)
    assert str(error.value) == 'ConnectionError'


def test_cursor_checkpoint_resume_and_duplicate_records(tmp_path, monkeypatch):
    monkeypatch.setattr(acq, 'DATA', tmp_path)
    monkeypatch.setattr(acq, 'verify', lambda: None)
    monkeypatch.setattr(acq, 'api_key', lambda: 'synthetic-pagination-key')
    monkeypatch.setattr(acq.time, 'sleep', lambda _: None)
    responses = iter([
        {'success': True, 'items': [{'id': '/a'}, {'id': '/b'}], 'nextCursor': 'page2', 'totalResults': 3},
        {'success': True, 'items': [{'id': '/b'}, {'id': '/c'}], 'nextCursor': 'page2', 'totalResults': 3},
    ])
    monkeypatch.setattr(acq, 'get', lambda *a, **kw: next(responses))
    assert len(acq.harvest()) == 3
    def should_not_request(*args, **kwargs):
        raise AssertionError('Cached page should have been reused')
    monkeypatch.setattr(acq, 'get', should_not_request)
    assert len(acq.harvest()) == 3
    assert json.loads((tmp_path / 'harvest.json').read_text())['complete']
    assert 'synthetic-pagination-key' not in (tmp_path / 'pages/00000.json').read_text()
