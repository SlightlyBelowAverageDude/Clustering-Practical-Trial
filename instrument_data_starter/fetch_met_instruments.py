#!/usr/bin/env python3
"""Fetch a balanced pilot of public-domain musical instruments from The Met.

Data source: https://huggingface.co/datasets/metmuseum/openaccess
Metadata file: openaccess.csv.gz (about 25 MB), not the 419 GB image dataset.
Images: primaryImageSmall URLs or the official Met Collection API.

Examples:
  python fetch_met_instruments.py --per-class 40 --download-images
  python fetch_met_instruments.py --metadata-file /path/to/MetObjects.csv --per-class 40 --download-images

Outputs under data/met_instruments/ by default:
  objects.csv
  images/<objectID>.jpg

No images are bundled in this script. Network access is needed to fetch data.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests

METADATA_URL = "https://huggingface.co/datasets/metmuseum/openaccess/resolve/main/openaccess.csv.gz"
MET_API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"
HS_CLASSES = {"Aerophone", "Chordophone", "Idiophone", "Membranophone", "Electrophone"}
COLUMN_ALIASES = {
    "objectid": "objectID",
    "department": "department",
    "ispublicdomain": "isPublicDomain",
    "classification": "classification",
    "objectname": "objectName",
    "title": "title",
    "culture": "culture",
    "country": "country",
    "region": "region",
    "objectbegindate": "year",
    "primaryimagesmall": "primaryImageSmall",
    "primaryimage": "primaryImage",
}


def canonical(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def download_file(session: requests.Session, url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            with session.get(url, stream=True, timeout=90) as response:
                response.raise_for_status()
                with path.with_suffix(path.suffix + ".part").open("wb") as f:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            f.write(block)
            path.with_suffix(path.suffix + ".part").replace(path)
            return
        except requests.RequestException as exc:
            if attempt == 4:
                raise RuntimeError(f"Cannot download {url}: {exc}") from exc
            time.sleep(2 ** attempt)


def load_metadata(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    use = [c for c in header if canonical(c) in COLUMN_ALIASES]
    raw = pd.read_csv(path, usecols=use, dtype=str, keep_default_na=False, low_memory=False)
    raw = raw.rename(columns={c: COLUMN_ALIASES[canonical(c)] for c in raw.columns})
    required = {"objectID", "department", "isPublicDomain", "classification"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}. Available: {list(raw.columns)}")
    for c in ["objectName", "title", "culture", "country", "region", "year", "primaryImageSmall", "primaryImage"]:
        if c not in raw.columns:
            raw[c] = ""
    df = raw[(raw.department.str.strip() == "Musical Instruments") &
             (raw.isPublicDomain.str.strip().str.lower().isin(["true", "1", "yes"]))].copy()
    df["hs1"] = df.classification.str.split("-").str[0].str.strip()
    df = df[df.hs1.isin(HS_CLASSES) & df.objectID.str.fullmatch(r"\d+")].copy()
    df = df.drop_duplicates("objectID").reset_index(drop=True)
    return df


def get_image_url(session: requests.Session, row) -> str:
    url = str(row.primaryImageSmall or row.primaryImage).strip()
    if url.startswith("https://"):
        return url
    for attempt in range(4):
        try:
            r = session.get(MET_API.format(row.objectID), timeout=30)
            if r.status_code == 404:
                return ""
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            obj = r.json()
            if not obj.get("isPublicDomain", False):
                return ""
            return obj.get("primaryImageSmall") or obj.get("primaryImage") or ""
        except (requests.RequestException, ValueError):
            time.sleep(2 ** attempt)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-file", type=Path, help="Existing MetObjects.csv or openaccess.csv.gz")
    ap.add_argument("--output", type=Path, default=Path("data/met_instruments"))
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--download-images", action="store_true")
    ap.add_argument("--delay", type=float, default=0.35, help="Seconds between image/API requests")
    args = ap.parse_args()
    if args.per_class < 1 or args.delay < 0:
        ap.error("--per-class must be >=1 and --delay must be >=0")
    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "LMU instrument-clustering research pilot/0.1"})
    source = args.metadata_file or (args.output / "openaccess.csv.gz")
    if not source.exists():
        print(f"Downloading compact Met metadata to {source} ...", flush=True)
        download_file(session, METADATA_URL, source)
    df = load_metadata(source)
    print(f"Found {len(df)} public-domain Met musical instruments with a recognized H-S class")
    print("Available classes:", df.hs1.value_counts().to_dict())
    if df.empty:
        raise SystemExit("No qualifying instruments found. Check source schema and filters.")
    parts = []
    for hs, group in df.groupby("hs1", sort=True):
        parts.append(group.sample(n=min(args.per_class, len(group)), random_state=args.seed))
    chosen = pd.concat(parts).sort_values("objectID").reset_index(drop=True)
    chosen["hs2"] = chosen.classification.str.split("-").str[:2].str.join("-")
    chosen["file"] = ""
    chosen["source"] = ""
    if args.download_images:
        (args.output / "images").mkdir(exist_ok=True)
        for i, row in enumerate(chosen.itertuples(index=False)):
            file = args.output / "images" / f"{row.objectID}.jpg"
            url = get_image_url(session, row)
            if file.exists() and file.stat().st_size > 1000:
                chosen.at[i, "file"] = f"images/{row.objectID}.jpg"
                chosen.at[i, "source"] = url
                continue
            if not url:
                continue
            try:
                r = session.get(url, timeout=45)
                r.raise_for_status()
                if r.headers.get("content-type", "").startswith("image/") and len(r.content) > 1000:
                    # Met's primaryImageSmall is documented as a JPEG.
                    if r.content[:3] == b"\xff\xd8\xff":
                        file.write_bytes(r.content)
                        chosen.at[i, "file"] = f"images/{row.objectID}.jpg"
                        chosen.at[i, "source"] = url
            except requests.RequestException as exc:
                print(f"Image failed for {row.objectID}: {exc}")
            time.sleep(args.delay)
            if (i + 1) % 25 == 0:
                print(f"Images checked: {i + 1}/{len(chosen)}", flush=True)
                chosen.to_csv(args.output / "objects.csv", index=False)
    chosen.to_csv(args.output / "objects.csv", index=False)
    print(f"Wrote {len(chosen)} metadata rows to {args.output / 'objects.csv'}")
    print(f"Images downloaded: {(chosen.file != '').sum()}")
    print("Do not mix this dataset with MIMO without checking taxonomic mappings.")


if __name__ == "__main__":
    main()
