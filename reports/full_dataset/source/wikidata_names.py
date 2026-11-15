"""Recreate the two Wikidata tables that spikes/mimo_names.py reads (they were never committed; dl/ is ignored).

dl/wd_mimo_instruments.csv : mimo (P3763 MIMO instrument ID), item (entity URI), itemLabel (English label), en (English Wikipedia URL or '')
dl/wd_mimo_ancestors.csv   : item (Q-id), anc (Q-id), ancLabel (English label) for every P279* ancestor (incl. the item)
Query time is recorded: Wikidata changes over time, so names/families can differ slightly from the October 2026 run.
"""
import json, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
DL = ROOT / 'dl'
EP = 'https://query.wikidata.org/sparql'
UA = {'User-Agent': 'CMVAL-LMU-research/1.0 (academic project)', 'Accept': 'application/sparql-results+json'}
Q_ITEMS = """SELECT ?item ?itemLabel ?mimo ?en WHERE {
  ?item wdt:P3763 ?mimo .
  OPTIONAL { ?en schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}"""
Q_ANC = """SELECT ?item ?anc ?ancLabel WHERE {
  ?i wdt:P3763 ?m .
  ?i wdt:P279* ?anc .
  BIND(STRAFTER(STR(?i), "entity/") AS ?item)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}"""


def run(q):
    for attempt in range(5):
        r = requests.get(EP, params={'query': q, 'format': 'json'}, headers=UA, timeout=300)
        if r.status_code == 200:
            return [{k: v['value'] for k, v in b.items()} for b in r.json()['results']['bindings']]
        time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f'SPARQL failed: HTTP {r.status_code}')


def main():
    DL.mkdir(exist_ok=True)
    items = pd.DataFrame(run(Q_ITEMS)).reindex(columns=['item', 'itemLabel', 'mimo', 'en']).fillna('')
    anc = pd.DataFrame(run(Q_ANC)).reindex(columns=['item', 'anc', 'ancLabel']).fillna('')
    anc['anc'] = anc.anc.str.replace('http://www.wikidata.org/entity/', '', regex=False)
    items.to_csv(DL / 'wd_mimo_instruments.csv', index=False); anc.to_csv(DL / 'wd_mimo_ancestors.csv', index=False)
    meta = {'queried_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'endpoint': EP, 'items_rows': len(items),
            'distinct_mimo_ids': int(items.mimo.nunique()), 'ancestor_rows': len(anc), 'query_items': Q_ITEMS, 'query_ancestors': Q_ANC}
    out = ROOT / 'reports/full_dataset/wikidata_query.json'; out.write_text(json.dumps(meta, indent=2))
    print({k: v for k, v in meta.items() if not k.startswith('query_')})


if __name__ == '__main__':
    main()
