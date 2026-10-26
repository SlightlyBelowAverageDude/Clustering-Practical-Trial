"""Public-domain instrument photos of the Met (CC0) for the transfer test.

Objects: every record of the Musical Instruments department in dl/MetObjects.csv that is public domain and whose
classification starts with one Hornbostel-Sachs class (Aerophone, Chordophone, Idiophone, Membranophone). The image
URL (primaryImageSmall) comes from the Met collection API, one request per object, at most 10 per second. Photos
already fetched for the 2023 pilot (data/instruments/images/<objectID>.jpg) are reused.
Outputs: data/met_instruments/images/<objectID>.jpg, data/met_instruments/objects.csv. Resumable.
"""
import shutil
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/met_instruments"
OLD = ROOT / "data/instruments/images"
API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"
CLASSES = ("Aerophone", "Chordophone", "Idiophone", "Membranophone")


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    cols = {"Object ID": "objectID", "Object Name": "objectName", "Title": "title", "Classification": "classification", "Culture": "culture",
            "Country": "country", "Region": "region", "Object Begin Date": "year", "Medium": "medium", "Is Public Domain": "isPublicDomain",
            "Department": "department"}
    m = pd.read_csv(ROOT / "dl/MetObjects.csv", usecols=list(cols), dtype=str, low_memory=False).rename(columns=cols).fillna("")
    m = m[(m.department == "Musical Instruments") & (m.isPublicDomain == "True")]
    m = m[m.classification.str.split("-").str[0].isin(CLASSES)].copy()
    m["hs1"] = m.classification.str.split("-").str[0]
    m["hs2"] = m.classification.str.split("-").str[:2].str.join("-")
    print(len(m), "public-domain instruments with a Hornbostel-Sachs class", flush=True)
    done = pd.read_csv(OUT / "objects.csv", dtype=str).fillna("") if (OUT / "objects.csv").exists() else pd.DataFrame(columns=["objectID"])
    have = set(done.objectID)
    rows = done.to_dict("records")
    s = requests.Session()
    s.headers.update({"User-Agent": "clustering-practical-trial (research, contact via GitHub SlightlyBelowAverageDude)"})
    t0 = time.time()
    n_new = 0
    for i, r in enumerate(m.itertuples(index=False)):
        oid = r.objectID
        if oid in have:
            continue
        dst = OUT / "images" / f"{oid}.jpg"
        url = ""
        if (OLD / f"{oid}.jpg").exists():
            if not dst.exists():
                shutil.copy(OLD / f"{oid}.jpg", dst)
            url = "pilot copy"
        else:
            for attempt in range(4):
                try:
                    j = s.get(API.format(oid), timeout=60)
                    if j.status_code == 404:
                        break
                    if j.status_code != 200:
                        time.sleep(2 + 3 * attempt)
                        continue
                    url = j.json().get("primaryImageSmall") or ""
                    if url and not dst.exists():
                        im = s.get(url, timeout=120)
                        if im.status_code == 200 and im.content[:2] == b"\xff\xd8":
                            dst.write_bytes(im.content)
                        else:
                            url = ""
                    break
                except Exception as e:  # noqa: BLE001
                    print("  retry", oid, type(e).__name__, flush=True)
                    time.sleep(2 + 3 * attempt)
            time.sleep(0.1)
        rows.append({"objectID": oid, "objectName": r.objectName, "title": r.title, "classification": r.classification, "hs1": r.hs1, "hs2": r.hs2,
                     "culture": r.culture, "country": r.country, "region": r.region, "year": r.year, "medium": r.medium,
                     "file": f"images/{oid}.jpg" if dst.exists() else "", "source": url})
        n_new += 1
        if n_new % 100 == 0:
            pd.DataFrame(rows).to_csv(OUT / "objects.csv", index=False)
            print(f"{i + 1} of {len(m)}, {n_new} new, {(time.time() - t0) / 60:.1f} min", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "objects.csv", index=False)
    print("done:", len(t), "records,", int((t.file != "").sum()), "with a photo", flush=True)


if __name__ == "__main__":
    main()
