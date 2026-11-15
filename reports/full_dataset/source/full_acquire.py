"""Full MIMO image acquisition (all 63,478 Europeana MIMO image records), reusing the validated pilot downloader.

Reuses spikes/pilot_acquire.py: the same cached, key-redacted metadata harvest (data/mimo_pilot/pages, metadata.csv),
download_one() (Europeana preview first, source-image fallback, decode/size/near-uniform validation, byte and pixel
sha256, content-addressed file names -> already-downloaded pilot images are reused), and get() (bounded retries with
exponential backoff, Retry-After on 408/429/5xx, credential-free error messages).

Idempotent and resumable: results are appended to data/mimo_full/ledger_parts/*.csv in chunks; on rerun, rows with
status 'ok' are skipped and failures are retried. Image bytes stay in the ignored data/ folder.
Usage: python reports/full_dataset/source/full_acquire.py [--workers 6] [--chunk 1000] [--harvest]
"""
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'spikes'))
import pilot_acquire as acq  # noqa: E402

OUT = ROOT / 'data/mimo_full'
PARTS = OUT / 'ledger_parts'
KEEP = ['id', 'museum', 'status', 'image_id', 'file', 'image_url', 'sha256', 'pixel_sha256', 'width', 'height', 'url_failures']


def done_ids():
    if not PARTS.exists():
        return set()
    parts = [pd.read_csv(p, dtype=str, keep_default_na=False, usecols=['id', 'status']) for p in sorted(PARTS.glob('*.csv'))]
    if not parts:
        return set()
    d = pd.concat(parts).drop_duplicates('id', keep='last')
    return set(d[d.status == 'ok'].id)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--chunk', type=int, default=1000)
    ap.add_argument('--harvest', action='store_true', help='re-run the resumable harvest (uses cached pages) before downloading')
    a = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True); (acq.DATA / 'images').mkdir(parents=True, exist_ok=True)
    acq.verify()  # live count check with the configured key (key never printed)
    if a.harvest:
        records = acq.harvest()
    else:
        records = pd.read_csv(acq.DATA / 'metadata.csv', dtype=str, keep_default_na=False).to_dict('records')
    print('metadata records:', len(records), flush=True)
    skip = done_ids()
    todo = [r for r in records if r['id'] not in skip]
    print('already ok:', len(skip), '| to process:', len(todo), flush=True)
    t0, n = time.time(), 0
    with ThreadPoolExecutor(a.workers) as ex:
        for start in range(0, len(todo), a.chunk):
            chunk = todo[start:start + a.chunk]
            rows = [{k: r.get(k, '') for k in KEEP} for r in ex.map(acq.download_one, chunk)]
            part = PARTS / f'{int(time.time() * 1000)}_{start:06d}.csv'
            acq.atomic_csv(part, rows)
            n += len(chunk)
            ok = sum(r['status'] == 'ok' for r in rows)
            print(f'{n}/{len(todo)} processed, chunk ok {ok}/{len(chunk)}, {time.time() - t0:.0f}s', flush=True)
    # consolidated ledger (last status per record)
    led = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in sorted(PARTS.glob('*.csv'))]).drop_duplicates('id', keep='last')
    acq.atomic_csv(OUT / 'download_ledger.csv', led.to_dict('records'))
    summary = {'utc': acq.now(), 'records': len(records), 'ledger_rows': len(led), 'status': led.status.value_counts().to_dict()}
    (OUT / 'download_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
