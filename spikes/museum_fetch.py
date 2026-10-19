"""Museum objects spike, step 1: fetch public-domain object images from the Met Open Access API.

Objects are pulled for a set of functional object names (bowl, jar, cup, comb, ...) across several
departments, which at the Met map roughly onto cultures. Two labels are stored per object: the
department (culture proxy) and the functional query term. The question for the spike is which of the
two the image clusters follow.
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
DEPARTMENTS = {10: "Egyptian Art", 13: "Greek and Roman Art", 6: "Asian Art", 14: "Islamic Art",
               5: "Africa, Oceania, Americas", 17: "Medieval Art", 3: "Ancient West Asian Art"}
FUNCTIONS = ["bowl", "jar", "cup", "comb", "lamp", "spoon", "knife", "bottle", "mirror", "bead"]


def get_json(session, url, params=None, tries=6):
    """GET with retry and backoff; the Met API throttles bursts with 403/429 or empty bodies."""
    for t in range(tries):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            print(f"  http {r.status_code} on {url.split('/')[-1][:30]}, retry {t + 1}")
        except Exception as e:  # noqa: BLE001
            print(f"  error {type(e).__name__} on {url.split('/')[-1][:30]}, retry {t + 1}")
        time.sleep(2 * (t + 1))
    return None


def search(session, q, dept, limit, offset=0):
    j = get_json(session, f"{API}/search", {"q": q, "departmentId": dept, "hasImages": "true",
                                            "isPublicDomain": "true", "limit": limit, "offset": offset})
    return (j or {}).get("objectIDs") or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=15, help="max objects per (department, function)")
    ap.add_argument("--out", default=str(ROOT / "data/museum"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    rows = []
    for dept, dname in DEPARTMENTS.items():
        for fn in FUNCTIONS:
            try:
                ids = search(s, fn, dept, a.per_cell * 3)
            except Exception as e:  # noqa: BLE001
                print("search failed", dname, fn, e)
                continue
            kept = 0
            for oid in ids:
                if kept >= a.per_cell:
                    break
                o = get_json(s, f"{API_OBJ}/{oid}")
                if not o:
                    continue
                name = (o.get("objectName") or "").lower()
                img = o.get("primaryImageSmall") or ""
                if fn not in name or not img or not o.get("isPublicDomain"):
                    continue
                f = out / "images" / f"{oid}.jpg"
                if not f.exists():
                    try:
                        f.write_bytes(s.get(img, timeout=60).content)
                    except Exception:  # noqa: BLE001
                        continue
                rows.append({"objectID": oid, "department": dname, "function": fn, "objectName": o.get("objectName"),
                             "culture": o.get("culture"), "period": o.get("period"), "classification": o.get("classification"),
                             "medium": o.get("medium"), "year": o.get("objectBeginDate"), "file": f"images/{oid}.jpg"})
                kept += 1
                time.sleep(0.25)
            print(f"{dname:28s} {fn:8s} {kept:3d}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "objects.csv", index=False)
    print(df.groupby(["department", "function"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
