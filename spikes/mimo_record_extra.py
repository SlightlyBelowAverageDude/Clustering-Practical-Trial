"""Extra metadata from the Europeana record API for the objects of the 4 largest instrument names (violin, clarinet,
Western concert flute, guitar): every dcCreator value (maker names and MIMO maker ids), the free-text dcDate, and the
first 300 characters of dcDescription. One request per record, about 6 per second, resumable.
Output: data/mimo/record_extra.csv (id, creators, date_text, description).
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mimo/record_extra.csv"
NAMES = ("violin", "clarinet", "Western concert flute", "guitar")
API = "https://api.europeana.eu/record/v2{}.json"


def values(d):
    if not isinstance(d, dict):
        return []
    out = []
    for v in d.values():
        out += [x for x in v if isinstance(x, str)]
    return out


def main():
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    ids = rec[rec.name.isin(NAMES)].id.tolist()
    done = pd.read_csv(OUT, dtype=str).fillna("") if OUT.exists() else pd.DataFrame(columns=["id"])
    have = set(done.id)
    rows = done.to_dict("records")
    s = requests.Session()
    t0 = time.time()
    n = 0
    for rid in ids:
        if rid in have:
            continue
        creators, date_text, desc = "", "", ""
        for attempt in range(4):
            try:
                r = s.get(API.format(rid), params={"wskey": "api2demo"}, timeout=60)
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                if r.status_code != 200:
                    break
                for p in r.json().get("object", {}).get("proxies", []):
                    if p.get("europeanaProxy"):
                        continue
                    creators = " | ".join(values(p.get("dcCreator")))
                    date_text = " | ".join(values(p.get("dcDate")))
                    desc = " | ".join(values(p.get("dcDescription")))[:300]
                break
            except Exception as e:  # noqa: BLE001
                print("  retry", rid, type(e).__name__, flush=True)
                time.sleep(3 * (attempt + 1))
        rows.append({"id": rid, "creators": creators, "date_text": date_text, "description": desc})
        n += 1
        time.sleep(0.15)
        if n % 200 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"{n} fetched of {len(ids) - len(have)}, {(time.time() - t0) / 60:.1f} min", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
