"""Sacred buildings spike, step 1: download Commons images for religious buildings pulled from Wikidata.

Classes (religion proxy): church, mosque, synagogue, Hindu temple, Buddhist temple, Shinto shrine.
Each row has country and inception year. Images are Commons thumbnails (width 512), one per building.
"""
import io  # noqa: F401
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wiki import UA, get, session  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
out = ROOT / "data/sacred"
(out / "images").mkdir(parents=True, exist_ok=True)

rows = []
for f in sorted((ROOT / "dl").glob("wd_*.csv")):
    df = pd.read_csv(f).drop_duplicates("b")
    df["year"] = pd.to_numeric(df.inception.str[:4], errors="coerce")
    df = df.dropna(subset=["year"]).sample(frac=1, random_state=0).head(PER_CLASS)
    df["cls"] = f.stem[3:]
    rows.append(df)
df = pd.concat(rows).reset_index(drop=True)
df["qid"] = df.b.str.rsplit("/", n=1).str[-1]
print(df.groupby("cls").size().to_dict(), flush=True)

s = session()


def fetch(r):
    """Download the original file (thumbnail scaling on Commons is rate-limited) and resize locally."""
    fn = out / "images" / f"{r.qid}.jpg"
    if fn.exists():
        return True
    # curl instead of requests: Commons answers the Python client with 429 even for originals, curl gets 200
    tmp = out / "images" / f"{r.qid}.tmp"
    for attempt in range(3):
        p = subprocess.run(["curl", "-sL", "-m", "120", "-A", UA["User-Agent"], "-o", str(tmp), "-w", "%{http_code}",
                            r.img.replace("http://", "https://")], capture_output=True, text=True)
        code = p.stdout.strip()[-3:]
        if code == "200":
            break
        if code == "429":
            print("  429 from Commons, sleeping 600 s", flush=True)
            time.sleep(600)
            continue
        return False
    else:
        return False
    if not tmp.exists():
        return False
    try:
        im = Image.open(tmp).convert("RGB")
    except Exception:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return False
    tmp.unlink(missing_ok=True)
    im.thumbnail((640, 640))
    im.save(fn, quality=90)
    time.sleep(8.0)
    return True


t0 = time.time()
with ThreadPoolExecutor(1) as ex:
    ok = list(ex.map(fetch, df.itertuples()))
df = df[ok]
print(f"{len(df)} images in {time.time() - t0:.0f}s", flush=True)
pd.DataFrame({"qid": df.qid, "name": df.bLabel, "cls": df.cls, "country": df.countryLabel,
              "year": df.year.astype(int), "file": "images/" + df.qid + ".jpg"}).to_csv(out / "buildings.csv", index=False)
print(df.groupby("cls").size().to_dict())
