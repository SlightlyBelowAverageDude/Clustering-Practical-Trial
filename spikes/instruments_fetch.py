"""Musical instruments spike, step 1: fetch public-domain instrument images from the Met (department 18).

The Met's classification field for instruments follows Hornbostel-Sachs, e.g.
"Aerophone-Blow Hole-side-blown flute (transverse)" or "Chordophone-Lute-plucked-fretted". That gives a
formal hierarchy (top class, second level) as ground truth, plus culture and date.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

API = "https://collectionapi.metmuseum.org/public/collection/v1.1"
API_OBJ = "https://collectionapi.metmuseum.org/public/collection/v1/objects"
ROOT = Path(__file__).resolve().parents[1]
QUERIES = ["flute", "drum", "guitar", "violin", "harp", "lute", "trumpet", "horn", "zither", "bell", "rattle", "whistle",
           "fiddle", "oboe", "clarinet", "bagpipe", "xylophone", "cymbal", "gong", "lyre", "piano", "organ", "banjo", "sitar"]


def get_json(session, url, params=None, tries=6):
    for t in range(tries):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            print(f"  http {r.status_code}, retry {t + 1}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  error {type(e).__name__}, retry {t + 1}", flush=True)
        time.sleep(2 * (t + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "data/instruments"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    seen, rows = set(), []
    for q in QUERIES:
        j = get_json(s, f"{API}/search", {"q": q, "departmentId": 18, "hasImages": "true", "isPublicDomain": "true",
                                           "limit": a.per_query * 2})
        ids = (j or {}).get("objectIDs") or []
        kept = 0
        for oid in ids:
            if kept >= a.per_query or oid in seen:
                continue
            o = get_json(s, f"{API_OBJ}/{oid}")
            if not o or not o.get("isPublicDomain") or not o.get("primaryImageSmall") or not o.get("classification"):
                continue
            f = out / "images" / f"{oid}.jpg"
            if not f.exists():
                try:
                    f.write_bytes(s.get(o["primaryImageSmall"], timeout=60).content)
                except Exception:  # noqa: BLE001
                    continue
            cls = o["classification"]
            parts = [p.strip() for p in cls.split("-")]
            rows.append({"objectID": oid, "query": q, "classification": cls, "hs1": parts[0], "hs2": "-".join(parts[:2]),
                         "objectName": o.get("objectName"), "culture": o.get("culture"), "country": o.get("country"),
                         "region": o.get("region"), "year": o.get("objectBeginDate"), "medium": o.get("medium"),
                         "file": f"images/{oid}.jpg"})
            seen.add(oid)
            kept += 1
            time.sleep(0.25)
        print(f"{q:10s} {kept:3d}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "objects.csv", index=False)
    print(df.hs1.value_counts().to_dict())
    print(df.hs2.value_counts().head(15).to_dict())


if __name__ == "__main__":
    main()
