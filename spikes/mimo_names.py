"""Instrument name and Western family per MIMO record, from the MIMO keyword id through Wikidata (P3763).

Reproducible version of the labelling that produced runs/mimo_names/labels.csv. When a MIMO id is tied to 2 Wikidata
items (10 ids, for example 4093 transverse flute / Western concert flute), the item with an English Wikipedia page
is taken, then the alphabetically first label; the first version took whichever row came last from SPARQL.
Output: <out>/labels.csv with columns id, kw, name, family (default out: runs/mimo_names_v2).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs/mimo_names_v2")
FAM = ["woodwind instrument", "brass instrument", "string instrument", "percussion instrument", "keyboard instrument", "electronic musical instrument"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "runs/mimo_full/sample_full.csv", dtype=str).fillna("")
    wd = pd.read_csv(ROOT / "dl/wd_mimo_instruments.csv", dtype=str).fillna("")
    anc = pd.read_csv(ROOT / "dl/wd_mimo_ancestors.csv", dtype=str).fillna("")
    wd["item"] = wd.item.str.replace("http://www.wikidata.org/entity/", "")
    wd["has_en"] = (wd.en != "").astype(int)
    pick = wd.sort_values(["mimo", "has_en", "itemLabel"], ascending=[True, False, True]).drop_duplicates("mimo")
    item_of, label_of = dict(zip(pick.mimo, pick.item)), dict(zip(pick.mimo, pick.itemLabel))
    fam = {}
    for it, g in anc.groupby("item"):
        f = [a for a in FAM if a in set(g.ancLabel)]
        fam[it] = f[0] if len(f) == 1 else ""  # 2 families (keyboard and string for a piano-like item): no family
    df["kw"] = df.keyword_uri.str.extract(r"/(\d+)$")[0].fillna("")
    df["name"] = df.kw.map(label_of).fillna("")
    df["family"] = df.kw.map(item_of).map(fam).fillna("")
    df[["id", "kw", "name", "family"]].to_csv(OUT / "labels.csv", index=False)
    print("names:", int((df.name != "").sum()), "| families:", int((df.family != "").sum()), "| distinct names:", df.name[df.name != ""].nunique())
    old = ROOT / "runs/mimo_names/labels.csv"
    if old.exists():
        o = pd.read_csv(old, dtype=str).fillna("")
        print("vs first version: names differ on", int((o.name != df.name).sum()), "rows; families differ on", int((o.family != df.family).sum()), "rows")


if __name__ == "__main__":
    main()
