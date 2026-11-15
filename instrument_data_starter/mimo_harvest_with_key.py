"""MIMO step 1: harvest all MIMO records with images from the Europeana API (cursor pagination) and
resolve their Hornbostel-Sachs concept URIs to notation and label through the MIMO Skosmos service.

Output: data/mimo/records.csv (one row per record) and data/mimo/hs_vocab.csv (URI, notation, label).
"""
import json
import os
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mimo"
UA = {"User-Agent": "ClusteringPraktikum/0.1 (LMU Munich student project; contact via course Moodle)"}
BASE = ("https://api.europeana.eu/record/v2/search.json?query=*&qf=PROVIDER:%22MIMO%20-%20Musical%20Instrument%20Museums%20Online%22"
        "&qf=TYPE:IMAGE&wskey=" + urllib.parse.quote(os.environ.get("EUROPEANA_API_KEY", "")) + "&rows=100&profile=rich")
SKOSMOS = "https://vocabulary.mimo-international.com/rest/v1/HornbostelAndSachs/data"


def first(d, *keys):
    for k in keys:
        v = d.get(k)
        if v:
            return v[0] if isinstance(v, list) else v
    return ""


def main():
    if not os.environ.get("EUROPEANA_API_KEY"):
        raise SystemExit("Set EUROPEANA_API_KEY in your environment first. See https://pro.europeana.eu/page/get-api")
    OUT.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update(UA)
    rows, cursor, page = [], "*", 0
    while cursor:
        for attempt in range(5):
            r = s.get(BASE + "&cursor=" + urllib.parse.quote(cursor), timeout=60)
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        d = r.json()
        for it in d.get("items", []):
            hs = [u for u in it.get("edmConcept", []) if "HornbostelAndSachs" in u]
            kw = [u for u in it.get("edmConcept", []) if "InstrumentsKeywords" in u]
            lang = it.get("edmPlaceLabelLangAware", {}) or {}
            place_en = (lang.get("en") or [""])[0] if isinstance(lang, dict) else ""
            rows.append({"id": it.get("id"), "title": first(it, "title"), "hs_uri": hs[0] if hs else "", "n_hs": len(hs),
                         "keyword_uri": kw[0] if kw else "", "place": first(it, "edmPlaceLabel"), "place_en": place_en,
                         "places_all": " | ".join(p for p in it.get("edmPlaceLabel", []) if isinstance(p, str)) or
                                       " | ".join(p.get("def", "") for p in it.get("edmPlaceLabel", []) if isinstance(p, dict)),
                         "year": first(it, "year"), "creator": first(it, "dcCreator"), "provider": first(it, "dataProvider"),
                         "country": first(it, "country"), "rights": first(it, "rights"), "image": first(it, "edmIsShownBy"),
                         "preview": first(it, "edmPreview"), "completeness": it.get("completeness")})
        cursor = d.get("nextCursor")
        page += 1
        if page % 50 == 0:
            print(page, "pages,", len(rows), "records", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "records.csv", index=False)
        time.sleep(0.25)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "records.csv", index=False)
    print(len(df), "records;", (df.hs_uri != "").sum(), "with H-S;", (df.place != "").sum(), "with place;", (df.year != "").sum(), "with year")
    print(df.provider.value_counts().head(20).to_string())

    uris = sorted(set(df.hs_uri[df.hs_uri != ""]))
    print(len(uris), "distinct H-S URIs; resolving through Skosmos", flush=True)
    vocab = []
    for i, u in enumerate(uris):
        try:
            r = s.get(SKOSMOS, params={"uri": u, "format": "application/json"}, timeout=60)
            g = r.json().get("graph", [])
            node = next((n for n in g if n.get("uri") == u), {})
            lab = node.get("prefLabel", [])
            lab = next((x.get("value") for x in lab if x.get("lang") == "en"), lab[0].get("value") if lab else "") if isinstance(lab, list) else lab.get("value", "")
            vocab.append({"hs_uri": u, "notation": node.get("skos:notation") or node.get("notation", ""), "label": lab})
        except Exception as e:  # noqa: BLE001
            vocab.append({"hs_uri": u, "notation": "", "label": f"error {type(e).__name__}"})
        if i % 100 == 0:
            print(" ", i, flush=True)
        time.sleep(0.2)
    v = pd.DataFrame(vocab)
    v.to_csv(OUT / "hs_vocab.csv", index=False)
    print("resolved:", (v.notation != "").sum(), "of", len(v)); print(v.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
