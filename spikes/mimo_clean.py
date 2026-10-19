"""Cleaned MIMO record table: one row per usable object with its labels, built from the harvested records.

Removes: the Europeana "no image" placeholder, later byte-identical copies of a shared photo, and catalogue records
of instrument parts (bows, mouthpieces, beaters, plectrums, reeds) by vocabulary name or exact title.
Labels: H-S levels 1 to 6 from common.hs_level (suffix and compound codes handled); instrument name and Western
family from spikes/mimo_names.py with 2 corrections of Wikidata's tree (harmonica is a wind instrument, the Jew's
harp gets no family); country and region recomputed with the fixed matcher in mimo_fetch.country_of.
Output: runs/clean/records.csv plus runs/clean/exclusions.csv (what was dropped and why).
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hs_level  # noqa: E402
from mimo_fetch import country_of  # noqa: E402
from mimo_full import RMAP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/clean"
PART_NAMES = {"bow", "woodwind mouthpiece", "mouthpiece", "brass mouthpiece", "plectrum", "beater", "band reeds", "reed"}
PART_TITLES = {"bow", "beater", "drumstick", "mouthpiece", "strijkstok", "strijkstok voor viool", "strijkstok voor altviool", "strijkstok voor cello",
               "archet", "baguette", "mondstuk", "bell beater / model", "plectrum", "reed", "reeds", "mundstück", "bogen", "schlägel", "stråke", "munstycke"}
FAMILY_FIX = {"harmonica": "woodwind instrument", "Jew's harp": ""}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "runs/mimo_full/sample_full.csv", dtype=str).fillna("")
    names = pd.read_csv(ROOT / "runs/mimo_names_v2/labels.csv", dtype=str).fillna("")
    assert (names.id.values == df.id.values).all()
    df["kw"], df["name"], df["family"] = names.kw.values, names.name.values, names.family.values
    for n, f in FAMILY_FIX.items():
        df.loc[df.name == n, "family"] = f
    for d in range(1, 7):
        df[f"hs{d}"] = hs_level(df.notation, d)
    df["country"] = [country_of([a, b, c]) for a, b, c in zip(df.place_en, df.place, df.places_all)]
    df["region"] = df.country.map(RMAP).fillna("")
    df["hash"] = [hashlib.md5((ROOT / "data/mimo" / f).read_bytes()).hexdigest() for f in df.file]
    counts = df.hash.value_counts()
    reason = pd.Series("", index=df.index)
    reason[df.hash == counts.index[0]] = "placeholder image"
    dup = df.hash.duplicated(keep="first") & (reason == "")
    reason[dup] = "copy of a shared photo"
    by_title = df.title.str.strip().str.lower().isin(PART_TITLES) & (df.name.isin(PART_NAMES) | (df.name == ""))  # a named instrument keeps its row
    parts = (df.name.isin(PART_NAMES) | by_title) & (reason == "")
    reason[parts] = "instrument part"
    df["exclude"] = reason
    excl = df[df.exclude != ""]
    print("exclusions:", excl.exclude.value_counts().to_dict())
    print("   parts by name:", excl[excl.exclude == "instrument part"].name.value_counts().head(8).to_dict())
    print("   parts by title only:", excl[(excl.exclude == "instrument part") & ~excl.name.isin(PART_NAMES)].title.str.lower().value_counts().head(8).to_dict())
    excl[["id", "provider", "title", "name", "exclude"]].to_csv(OUT / "exclusions.csv", index=False)
    keep = df[df.exclude == ""].drop(columns=["exclude", "hash"]).reset_index(drop=True)
    cols = ["id", "file", "provider", "title", "notation", "hs1", "hs2", "hs3", "hs4", "hs5", "hs6", "kw", "name", "family", "country", "region", "year", "preview", "rights"]
    keep = keep[[c for c in cols if c in keep.columns]]
    keep.to_csv(OUT / "records.csv", index=False)
    print(f"kept {len(keep)} of {len(df)} records | coded {int((keep.hs1 != '').sum())} | named {int((keep.name != '').sum())} | with family {int((keep.family != '').sum())}"
          f" | with region {int((keep.region != '').sum())} | Oceania {int((keep.region == 'Oceania').sum())}")


if __name__ == "__main__":
    main()
