"""MIMO harvesting entry point, retaining the original spike CSV interface.

Uses the data starter's EUROPEANA_API_KEY correction with resumable acquisition
and safe errors from pilot_acquire. Complete metadata is in data/mimo_pilot/;
compatibility CSVs remain in data/mimo/ for mimo_fetch.py and older spikes.
"""
import json
from pathlib import Path

from pilot_acquire import annotate, atomic_csv, harvest, resolve_vocab

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/mimo'


def first(d, *keys):
    for key in keys:
        value = d.get(key)
        if value:
            return value[0] if isinstance(value, list) else value
    return ''


def main():
    records = harvest()
    vocab = resolve_vocab(records)
    annotate(records, vocab)
    rows = []
    for r in records:
        hs, kw = json.loads(r['hs_uris']), json.loads(r['keyword_uris'])
        images, previews = json.loads(r['image_urls']), json.loads(r['preview_urls'])
        places = json.loads(r['origin_lang_aware'])
        en = places.get('en', []) if isinstance(places, dict) else []
        rows.append({'id': r['id'], 'title': r['title'], 'hs_uri': hs[0] if hs else '', 'n_hs': len(hs),
                     'keyword_uri': kw[0] if kw else '', 'place': r['origin'], 'place_en': en[0] if en else '',
                     'places_all': r['origin'], 'year': r['year'], 'creator': r['creator'],
                     'provider': r['museum'], 'country': r['provider_country'], 'rights': r['rights'],
                     'image': images[0] if images else '', 'preview': previews[0] if previews else ''})
    atomic_csv(OUT / 'records.csv', rows)
    atomic_csv(OUT / 'hs_vocab.csv', [{'hs_uri': uri, **entry} for uri, entry in sorted(vocab.items())])
    print('Original spike compatibility CSVs saved to data/mimo/.')


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print('Harvest stopped:', e)
        raise SystemExit(1) from None
