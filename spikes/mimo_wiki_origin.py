"""MIMO: origin of the instrument type from Wikipedia categories ("X musical instruments"), scored against image clusters.

Record -> MIMO keyword id -> Wikidata item (P3763) -> English Wikipedia page -> geographic categories -> region.
A type gets a region only when all its geographic categories fall in one region. Compared with the museum's own
place-of-manufacture region on the same rows. Inputs: runs/mimo_full (embeddings, records), dl/wd_mimo_instruments.csv,
dl/wiki_instrument_cats.csv. Outputs: runs/mimo_wiki_origin.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/mimo_wiki_origin"
EUROPE = ("German American Italian Russian Ukrainian French English Czech Portuguese Serbian Swiss Greek Celtic Romanian Spanish Romani "
          "Irish Hungarian Basque Norwegian Bulgarian Belgian Slovak Moldovan Scottish Albanian Bosnian Croatian Polish Swedish Lithuanian Welsh "
          "Slovenian Belarusian Pontic-Greek Catalan Austrian Aragonese Macedonian Montenegrin Laz European Finnish Danish Latvian Breton Dutch "
          "Sardinian Asturian Castilian Andalusian Estonian Cypriot Balearic Maltese Galician Icelandic Andorran Calabrian Medieval Ancient-Greek "
          "Ancient-Roman Bashkir Tatar Mari Pre-Norman-Invasion").split()
ASIA = ("Chinese Indian Indonesian Arabic Japanese Korean Azerbaijani Armenian Persian Philippine Turkish Thai Hindustani Pakistani Turkmen "
        "Yemeni Uzbekistani Iranian Burmese Kurdish Tibetan Afghan Malaysian Kazakhstani Cambodian Kyrgyz Bangladeshi Nepalese Mongolian Tajik "
        "Bruneian Vietnamese Syrian Tuvan Uyghur Laotian Ancient-Hebrew Dagestanian Buryat Altai Lebanese Israeli Sri-Lankan Ainu").split()
AFRICA = ("West-African African Ancient-Egyptian Ethiopian Ugandan Egyptian Malian Hausa North-African Burkinabe Algerian Ghanaian Gambian "
          "Guinean Senegalese Eritrean Nigerien Chadian Cape-Verdean Bissau-Guinean Ivorian Yoruba Nigerian Zimbabwean Malagasy Equatoguinean "
          "Central-African-Republic Moroccan Berber Mauritanian Mozambican Tuareg Sudanese Tanzanian Southern-African Precolonial-African Gabonese "
          "Democratic-Republic-of-the-Congo Kenyan Somali Botswana Zambian Libyan").split()
AMERICAS = ("Argentine Cuban Brazilian Canadian Bolivian Guatemalan Mexican Venezuelan Chilean Peruvian Cajun Salvadoran American-Indian Hawaiian "
            "Indigenous-South-American South-American Costa-Rican Ecuadorian Trinidad-and-Tobago Dominican-Republic Inuit Mesoamerican "
            "Pre-Columbian-North-American").split()
OCEANIA = "Australian Papua-New-Guinean Fijian Polynesian Melanesian".split()
# "American" is ambiguous between the USA and the continent; the USA sits in the Americas here, as in the museum place map.
REGION = {d.replace("-", " "): r for r, ds in [("Europe", EUROPE), ("Asia", ASIA), ("Africa", AFRICA), ("Americas", AMERICAS), ("Oceania", OCEANIA)] for d in ds}
REGION["American"] = "Americas"
REGION.pop("Hawaiian", None); REGION["Hawaiian"] = "Oceania"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "runs/mimo_full/sample_full.csv", dtype={"notation": str}).fillna("")
    X = np.load(ROOT / "runs/mimo_full/emb/dinov2_b.npy")
    assert len(X) == len(df)
    Z = PCA(50, random_state=0).fit_transform(X)
    wd = pd.read_csv(ROOT / "dl/wd_mimo_instruments.csv", dtype=str).fillna("")
    wk = pd.read_csv(ROOT / "dl/wiki_instrument_cats.csv", dtype=str).fillna("")
    geo_of_page = {}
    unmapped = set()
    for p, cats in zip(wk.en, wk.instrument_cats):
        regs = set()
        for c in cats.split("|"):
            m = re.match(r"^([A-Z][A-Za-z\- ]+) musical instruments$", c)
            if not m:
                continue
            d = m.group(1)
            if d in REGION:
                regs.add(REGION[d])
            elif d not in ("Early", "Sacred", "Electronic", "Bamboo", "Mechanical", "Electric", "Experimental", "Improvised", "Gourd",
                           "Wind-activated", "Lost and extinct"):
                unmapped.add(d)
        geo_of_page[p] = regs
    print("unmapped demonyms:", sorted(unmapped))
    page_of_kw = dict(zip(wd.mimo, wd.en))
    df["kw"] = df.keyword_uri.str.extract(r"/(\d+)$")[0].fillna("")
    df["page"] = df.kw.map(page_of_kw).fillna("")
    regs = df.page.map(geo_of_page)
    df["type_region"] = [next(iter(r)) if isinstance(r, set) and len(r) == 1 else ("mixed" if isinstance(r, set) and len(r) > 1 else "") for r in regs]
    HS1 = {"1": "Idiophone", "2": "Membranophone", "3": "Chordophone", "4": "Aerophone", "5": "Electrophone"}
    df["hs1"] = df.notation.str.replace(".", "", regex=False).str[:1].map(HS1).fillna("")
    print("records with a page:", (df.page != "").sum(), "; type region:", (df.type_region.isin(["Europe", "Asia", "Africa", "Americas", "Oceania"])).sum(),
          "; mixed:", (df.type_region == "mixed").sum())
    print(df.type_region.value_counts().to_string())
    both = (df.type_region.isin(["Europe", "Asia", "Africa", "Americas", "Oceania"])) & (df.region != "")
    agree = (df.type_region[both] == df.region[both]).mean()
    print(f"rows with both origin labels: {both.sum()}; agreement type-region vs museum-place region: {agree:.3f}")
    print(pd.crosstab(df.type_region[both], df.region[both]).to_string())
    df[["id", "kw", "page", "type_region", "region", "hs1", "provider"]].to_csv(OUT / "labels.csv", index=False)

    def run(idx, col, min_count, ks, tag):
        y = np.array(df.loc[idx, col].tolist(), dtype=object)
        vc = pd.Series(y).value_counts()
        keep = (pd.Series(y).map(vc) >= min_count).values & (y != "") & (y != "mixed")
        ii = np.asarray(idx)[keep]; yc = y[keep]; k = len(set(yc))
        if k < 2:
            return None
        ami = {kk: round(score(yc, KMeans(kk, n_init=3, random_state=0).fit_predict(Z[ii]))["AMI"], 3) for kk in sorted(set(ks) | {k})}
        pr = cross_val_score(LogisticRegression(max_iter=1000, C=0.5), X[ii], yc, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        r = {"set": tag, "label": col, "n": len(yc), "classes": k, "AMI_at_classes": ami[k], "best_AMI": max(ami.values()), "best_k": max(ami, key=ami.get),
             "probe": round(float(pr), 3), "majority": round(float(pd.Series(yc).value_counts(normalize=True).iloc[0]), 3)}
        print(r, flush=True)
        return r

    rows = []
    idx = df.index[both].values
    print("\n== same rows, type region vs museum-place region")
    rows.append(run(idx, "type_region", 30, (5, 20, 50), "both"))
    rows.append(run(idx, "region", 30, (5, 20, 50), "both"))
    print("\n== type region inside each H-S class")
    for cls, g in df[both & (df.hs1 != "")].groupby("hs1"):
        rows.append(run(g.index.values, "type_region", 100, (), f"hs1={cls}"))
        rows.append(run(g.index.values, "region", 100, (), f"hs1={cls}"))
    print("\n== type region inside each museum with 2+ regions of 100+ objects")
    for prov, g in df[both].groupby("provider"):
        vc = g.type_region.value_counts(); vc = vc[(vc >= 100) & (vc.index != "mixed")]
        if len(vc) < 2:
            continue
        rows.append(run(g.index.values, "type_region", 100, (), f"museum={prov[:40]}"))
    pd.DataFrame([r for r in rows if r]).to_csv(OUT / "scores.csv", index=False)


if __name__ == "__main__":
    main()
