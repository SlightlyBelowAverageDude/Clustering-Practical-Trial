"""Resumable Europeana/MIMO acquisition. No credentials are included in saved requests."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests
from dotenv import load_dotenv
from PIL import Image, ImageStat, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/mimo_pilot'
API = 'https://api.europeana.eu/record/v2/search.json'
SKOSMOS = 'https://vocabulary.mimo-international.com/rest/v1/HornbostelAndSachs/data'
FILTERS = ['PROVIDER:"MIMO - Musical Instrument Museums Online"', 'TYPE:IMAGE']
UA = {'User-Agent': 'CMVAL-LMU-research-pilot/1.0'}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.part')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def safe_url(url):
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, p.path, urlencode([(k, v) for k, v in parse_qsl(p.query) if k.lower() not in {'wskey', 'apikey', 'api_key', 'key'}]), p.fragment))


def clean(value, secret=''):
    """Strip any key echo and credentials from every cached metadata string."""
    if isinstance(value, dict):
        return {k: clean(v, secret) for k, v in value.items() if k.lower() not in {'apikey', 'wskey', 'api_key'}}
    if isinstance(value, list):
        return [clean(v, secret) for v in value]
    if isinstance(value, str):
        if secret:
            value = value.replace(secret, '[REDACTED]')
        if value.startswith(('http://', 'https://')):
            return safe_url(value)
    return value


def get(url, params=None, attempts=4, image=False):
    """Never propagate Requests exceptions: their URLs may contain the API key."""
    error = 'unknown'
    for attempt in range(attempts):
        try:
            with requests.get(url, params=params, headers=UA, timeout=(10, 35), stream=image) as r:
                if r.status_code == 200:
                    if not image:
                        try:
                            return r.json()
                        except ValueError:
                            error = 'invalid_json'
                    else:
                        chunks, size = [], 0
                        for chunk in r.iter_content(65536):
                            size += len(chunk)
                            if size > 20 * 1024 * 1024:
                                raise RuntimeError('image_too_large')
                            chunks.append(chunk)
                        return b''.join(chunks)
                else:
                    error = f'HTTP_{r.status_code}'
                    if r.status_code not in {408, 429, 500, 502, 503, 504}:
                        break
                    try:
                        delay = min(60, float(r.headers.get('Retry-After', 0)))
                    except ValueError:
                        delay = 0
                    time.sleep(delay)
        except requests.RequestException as e:
            error = type(e).__name__
        if attempt + 1 < attempts:
            time.sleep(min(16, 2 ** attempt))
    raise RuntimeError(error)


def api_key():
    load_dotenv(ROOT / '.env', override=False)
    key = os.environ.get('EUROPEANA_API_KEY', '').strip()
    if not key:
        raise RuntimeError('EUROPEANA_API_KEY is missing; configure environment or local ignored .env')
    return key


def verify():
    key = api_key()
    d = get(API, {'wskey': key, 'query': '*', 'qf': FILTERS, 'rows': 1, 'profile': 'rich'})
    if not d.get('success') or not d.get('items'):
        raise RuntimeError('Europeana API returned no successful MIMO image result')
    evidence = {'verified_utc': now(), 'success': True, 'total_results': d['totalResults'], 'example_record_id': d['items'][0]['id'], 'filters': FILTERS}
    atomic_json(DATA / 'api_verification.json', evidence)
    print('Europeana verified:', evidence['total_results'], 'MIMO image records', flush=True)


def values(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [x for v in value for x in values(v)]
    if isinstance(value, dict):
        if 'def' in value:
            return values(value['def'])
        if 'value' in value:
            return values(value['value'])
        return [x for v in value.values() for x in values(v)]
    return [str(value)]


def norm_record(it):
    concepts = values(it.get('edmConcept'))
    def joined(k):
        return ' | '.join(dict.fromkeys(values(it.get(k))))
    def packed(v):
        return json.dumps(v, ensure_ascii=False)
    return {'id': it['id'], 'title': joined('title'), 'instrument_names': joined('title'),
            'hs_uris': packed([u for u in concepts if 'HornbostelAndSachs/' in u]),
            'keyword_uris': packed([u for u in concepts if 'InstrumentsKeywords/' in u]),
            'museum': joined('dataProvider'), 'provider': joined('provider'),
            'origin': joined('edmPlaceLabel'), 'origin_lang_aware': packed(it.get('edmPlaceLabelLangAware', {})),
            'provider_country': joined('country'), 'year': joined('year'), 'creator': joined('dcCreator'),
            'rights': joined('rights'), 'image_urls': packed(values(it.get('edmIsShownBy'))),
            'preview_urls': packed(values(it.get('edmPreview'))), 'landing_pages': packed(values(it.get('edmIsShownAt')))}


def harvest(max_records=0):
    verify()
    key = api_key()
    pages = DATA / 'pages'
    pages.mkdir(parents=True, exist_ok=True)
    cursor, page, records, seen = '*', 0, [], set()
    while cursor:
        path = pages / f'{page:05d}.json'
        if path.exists():
            d = json.loads(path.read_text())
            if d['request_cursor'] != cursor:
                raise RuntimeError('Cached cursor chain does not match; refusing mixed harvest')
        else:
            response = get(API, {'wskey': key, 'query': '*', 'qf': FILTERS, 'rows': 100, 'profile': 'rich', 'cursor': cursor})
            if not response.get('success') or 'items' not in response:
                raise RuntimeError('Unsuccessful Europeana search response')
            d = {'request_cursor': cursor, 'nextCursor': response.get('nextCursor'), 'totalResults': response.get('totalResults'), 'retrieved_utc': now(), 'items': clean(response['items'], key)}
            atomic_json(path, d)
            time.sleep(0.2)
        for it in d['items']:
            if it['id'] not in seen:
                records.append(norm_record(it))
                seen.add(it['id'])
        nxt = d.get('nextCursor')
        cursor = nxt if nxt and nxt != cursor and d['items'] else None
        page += 1
        if page % 25 == 0:
            print('Harvest:', len(records), 'records,', page, 'pages', flush=True)
        if max_records and len(records) >= max_records:
            break
    atomic_csv(DATA / 'records.csv', records)
    atomic_json(DATA / 'harvest.json', {'records': len(records), 'pages': page, 'total_results': d['totalResults'], 'complete': cursor is None, 'finished_utc': now(), 'filters': FILTERS, 'page_size': 100})
    print('Metadata saved:', len(records), 'records; complete:', cursor is None, flush=True)
    return records


def resolve_vocab(records):
    path = DATA / 'hs_vocab.json'
    vocab = json.loads(path.read_text()) if path.exists() else {}
    uris = sorted({u for r in records for u in json.loads(r['hs_uris'])})
    pending = [u for u in uris if not vocab.get(u, {}).get('notation')]
    def resolve(uri):
        try:
            d = get(SKOSMOS, {'uri': uri, 'format': 'application/json'})
            node = next((n for n in d.get('graph', []) if n.get('uri') == uri), {})
            labels = node.get('prefLabel', [])
            labels = labels if isinstance(labels, list) else [labels]
            label = next((v.get('value', '') for v in labels if v.get('lang') == 'en'), '')
            notation = values(node.get('skos:notation') or node.get('notation'))
            return uri, {'notation': notation[0] if notation else '', 'label': label, 'status': 'ok' if notation else 'no_notation'}
        except RuntimeError as e:
            return uri, {'notation': '', 'label': '', 'status': str(e)}
    with ThreadPoolExecutor(3) as ex:
        for i, (uri, entry) in enumerate(ex.map(resolve, pending)):
            vocab[uri] = entry
            if i % 25 == 0:
                atomic_json(path, vocab)
    atomic_json(path, vocab)
    atomic_csv(DATA / 'hs_vocab.csv', [{'hs_uri': k, **v} for k, v in sorted(vocab.items())])
    print('Vocabulary:', sum(bool(v.get('notation')) for v in vocab.values()), '/', len(vocab), 'resolved', flush=True)
    return vocab


def hs_prefix(notation, depth):
    base = re.split(r'[-+]', notation)[0].replace('.', '').strip()
    return base[:depth] if re.fullmatch(r'[1-5][0-9]*', base) and len(base) >= depth else ''


def annotate(records, vocab):
    out = []
    for r in records:
        r = dict(r)
        uris = json.loads(r['hs_uris'])
        codes = [vocab.get(u, {}).get('notation', '') for u in uris]
        r['notations'] = json.dumps(codes)
        r['hs_label'] = ' | '.join(vocab.get(u, {}).get('label', '') for u in uris)
        for depth in (1, 2, 3):
            prefixes = [hs_prefix(c, depth) for c in codes]
            r[f'hs{depth}'] = prefixes[0] if prefixes and all(prefixes) and len(set(prefixes)) == 1 else ''
        r['hs_multiple'] = len(uris) > 1
        out.append(r)
    atomic_csv(DATA / 'metadata.csv', out)
    return out


def candidate_order(records, seed):
    """Deterministic museum round-robin; no taxonomy balancing or inference."""
    groups = defaultdict(list)
    for r in records:
        groups[r['museum']].append(r)
    queues = []
    for museum in sorted(groups):
        rows = sorted(groups[museum], key=lambda r: digest(f'{seed}:{r["id"]}'.encode()))
        queues.append(deque(rows))
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.popleft())
    return out


def inspect_image(raw):
    with Image.open(io.BytesIO(raw)) as im:
        im.verify()
    with Image.open(io.BytesIO(raw)) as im:
        rgb = ImageOps.exif_transpose(im).convert('RGB')
        w, h = rgb.size
        if min(w, h) < 32 or max(w, h) < 96:
            raise ValueError('too_small')
        if max(ImageStat.Stat(rgb.resize((64, 64))).stddev) < 2:
            raise ValueError('near_uniform')
        pixel_hash = digest(f'{w}x{h}:'.encode() + rgb.tobytes())
    return w, h, pixel_hash


def download_one(row):
    row = dict(row)
    urls = list(dict.fromkeys(json.loads(row['preview_urls']) + json.loads(row['image_urls'])))
    failures = []
    for url in urls:
        if not url.startswith(('https://', 'http://')):
            continue
        image_id = digest((row['id'] + '\n' + url).encode())
        path = DATA / 'images' / (image_id + '.img')
        try:
            raw = path.read_bytes() if path.exists() else get(url, image=True, attempts=3)
            w, h, ph = inspect_image(raw)
            if not path.exists():
                tmp = path.with_suffix('.part')
                tmp.write_bytes(raw)
                tmp.replace(path)
            return {**row, 'image_id': image_id, 'file': str(path.relative_to(DATA)), 'image_url': url, 'sha256': digest(raw), 'pixel_sha256': ph, 'width': w, 'height': h, 'status': 'ok', 'url_failures': json.dumps(failures)}
        except (RuntimeError, ValueError, OSError, Image.DecompressionBombError) as e:
            path.unlink(missing_ok=True)
            failures.append({'url': safe_url(url), 'error': str(e) if isinstance(e, RuntimeError) else type(e).__name__})
    return {**row, 'status': 'download_failed' if urls else 'no_image_url', 'url_failures': json.dumps(failures)}


def download(records, n, seed, tag):
    (DATA / 'images').mkdir(parents=True, exist_ok=True)
    ordered = candidate_order(records, seed)
    atomic_csv(DATA / f'candidates_{tag}.csv', [{'rank': i, 'id': r['id'], 'museum': r['museum']} for i, r in enumerate(ordered)])
    selected, ledger, seen_pixels, seen_urls = [], [], set(), set()
    # Process responses in candidate order even when network tasks finish out of order.
    with ThreadPoolExecutor(4) as ex:
        for start in range(0, len(ordered), 16):
            if len(selected) >= n:
                break
            for row in ex.map(download_one, ordered[start:start + 16]):
                if row['status'] == 'ok':
                    if row['image_url'] in seen_urls or row['pixel_sha256'] in seen_pixels:
                        row['status'] = 'duplicate'
                    elif len(selected) >= n:
                        row['status'] = 'prefetched_unused'
                    else:
                        seen_pixels.add(row['pixel_sha256'])
                        seen_urls.add(row['image_url'])
                        selected.append(row)
                ledger.append(row)
            atomic_csv(DATA / f'sample_{tag}.csv', selected)
            atomic_csv(DATA / f'download_ledger_{tag}.csv', ledger)
            if start % 160 == 0:
                print('Downloads:', len(selected), '/', n, '; considered', len(ledger), flush=True)
    if not selected:
        raise RuntimeError('No valid images downloaded')
    report = {'created_utc': now(), 'target': n, 'selected': len(selected), 'seed': seed, 'sampling': 'museum round-robin, SHA256(seed:record_id) within museum; missing HS labels retained', 'candidate_records': len(records), 'attempted_records': len(ledger), 'statuses': dict(Counter(r['status'] for r in ledger)), 'unique_pixel_hashes': len(seen_pixels), 'manifest_sha256': digest((DATA / f'sample_{tag}.csv').read_bytes()), 'missing_selected': {col: sum(not r[col] for r in selected) for col in ['hs1', 'hs2', 'hs3', 'origin', 'rights', 'instrument_names']}, 'missing_metadata': {col: sum(not r[col] for r in records) for col in ['hs1', 'hs2', 'hs3', 'origin', 'rights', 'instrument_names']}}
    for scope, rows in [('metadata', records), ('selected', selected)]:
        for col in ['hs1', 'hs2', 'hs3', 'museum', 'rights']:
            counts = Counter(r[col] or '(missing)' for r in rows)
            atomic_csv(DATA / f'{scope}_{col}_{tag}.csv', [{'value': k, 'count': v} for k, v in counts.most_common()])
    atomic_json(DATA / f'dataset_report_{tag}.json', report)
    out = ROOT / 'results' / tag
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'dataset_report.json', report)
    for col in ['hs1', 'hs2', 'hs3', 'museum', 'rights']:
        (out / f'{col}_distribution.csv').write_bytes((DATA / f'selected_{col}_{tag}.csv').read_bytes())
    text = [f'# Dataset report: {tag}', '', f'Actual selected images: {len(selected)} / target {n}.', f'Metadata records: {len(records)}. Museums represented: {len({r["museum"] for r in selected})}.', '', '## Download outcomes', '', *[f'- {k}: {v}' for k, v in report['statuses'].items()], '', '## Missing fields in selected images', '', *[f'- {k}: {v}' for k, v in report['missing_selected'].items()], '', 'Distribution CSVs accompany this report. Detailed source metadata, raw search pages, sampling order, and download failure ledger are under ignored data/mimo_pilot/.', '', 'Images are validated by decoding and dimension/variance checks, then deduplicated by exact decoded RGB pixels and URL. Near duplicates, photographs of parts, and textured placeholders require further curation. Rights are recorded per item; origin is source place metadata and may not mean manufacture location.']
    (out / 'DATASET_REPORT.md').write_text('\n'.join(text) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-only', action='store_true')
    ap.add_argument('--max-records', type=int, default=0)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag', default='pilot')
    ap.add_argument('--reuse-metadata', action='store_true')
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if a.verify_only:
        verify()
        return
    if a.reuse_metadata:
        records = pd.read_csv(DATA / 'records.csv', dtype=str, keep_default_na=False).to_dict('records')
    else:
        records = harvest(a.max_records)
    vocab = resolve_vocab(records)
    records = annotate(records, vocab)
    download(records, a.n, a.seed, a.tag)


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print('Acquisition stopped:', e, flush=True)
        raise SystemExit(1) from None
