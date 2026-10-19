"""MIMO, every record with a thumbnail (63,478; 41,210 of them carry an H-S code): fetch, embed with DINOv2-B, score
against the H-S tree to 6 digits. Uncoded records stay in the set on purpose (they carry names and origin); the
depth table uses only the coded ones.

Reuses the parser and region map from mimo_fetch.py. Images go to data/mimo/images (shared with the samples).
"""
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings, hs_level, score  # noqa: E402
from labels import run_labels  # noqa: E402
from mimo_fetch import DATA, UA, country_of  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs/mimo_full"
ENCODER = sys.argv[1] if len(sys.argv) > 1 else "dinov2_b"
REGION = {"Europe": ["France", "Belgium", "Spain", "Sweden", "Austria", "Germany", "Netherlands", "Czechia", "United Kingdom", "Italy", "Switzerland",
                     "Denmark", "Ireland", "Portugal", "Poland", "Hungary", "Russia", "Greece", "Norway", "Finland", "Europe"],
          "Africa": ["Congo", "Nigeria", "Egypt", "Senegal", "Togo", "Tanzania", "Ivory Coast", "Guinea", "Burkina Faso", "Uganda", "Malawi", "Sudan",
                     "Kenya", "Madagascar", "Benin", "Gabon", "Cameroon", "Morocco", "Tunisia", "Algeria", "Ethiopia", "Rwanda", "Burundi", "Angola",
                     "Zambia", "Zimbabwe", "Mozambique", "South Africa", "Namibia", "Liberia", "Sierra Leone", "Gambia", "Niger", "Chad", "Mali", "Ghana",
                     "Central African Republic", "Africa"],
          "Asia": ["Indonesia", "India", "China", "Japan", "Vietnam", "Sri Lanka", "Pakistan", "Afghanistan", "Bangladesh", "Cambodia", "Laos", "Malaysia",
                   "Mongolia", "Iraq", "Syria", "Israel", "Saudi Arabia", "Yemen", "Oman", "Uzbekistan", "Iran", "Turkey", "Thailand", "Myanmar", "Korea",
                   "Tibet", "Nepal", "Philippines", "Asia"],
          "Americas": ["Mexico", "United States", "Peru", "Bolivia", "Brazil", "Argentina", "Chile", "Colombia", "Ecuador", "Venezuela", "Guatemala", "Cuba",
                       "Haiti", "Jamaica", "Canada"],
          "Oceania": ["New Guinea", "Oceania", "Australia", "New Zealand", "Fiji", "Samoa", "Tonga", "Solomon Islands", "Vanuatu"]}
RMAP = {c: r for r, cs in REGION.items() for c in cs}


def purity(y, lab):
    return sum(pd.Series(y[lab == c]).value_counts().iloc[0] for c in np.unique(lab)) / len(y)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(DATA / "records.csv").fillna("")
    voc = pd.read_csv(DATA / "hs_vocab.csv").fillna("")
    df = rec.merge(voc, on="hs_uri", how="left")
    df = df[(df.notation.astype(str) != "") & (df.preview != "")].copy().reset_index(drop=True)
    df["notation"] = df.notation.astype(str)
    df["country"] = [country_of([a, b, c]) for a, b, c in zip(df.place_en, df.place, df.places_all)]
    df["region"] = df.country.map(RMAP).fillna("")
    df["file"] = ["images/" + re.sub(r"[^A-Za-z0-9]+", "_", i.strip("/")) + ".jpg" for i in df.id]
    print(len(df), "records; with country", (df.country != "").sum(), "; with region", (df.region != "").sum(), flush=True)
    (DATA / "images").mkdir(exist_ok=True)
    sess = requests.Session()
    sess.headers.update(UA)

    def fetch(row):
        fn = DATA / row.file
        if fn.exists():
            return True
        for attempt in range(3):
            try:
                r = sess.get(row.preview, timeout=60)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                    fn.write_bytes(r.content)
                    return True
                if r.status_code == 429:
                    time.sleep(30)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2 * (attempt + 1))
        return False

    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        ok = []
        for i, res in enumerate(ex.map(fetch, df.itertuples())):
            ok.append(res)
            if i % 5000 == 0:
                print(f"  {i} fetched, {time.time() - t0:.0f}s", flush=True)
    df = df[ok].reset_index(drop=True)
    df.to_csv(RUNS / "sample_full.csv", index=False)
    print(f"{len(df)} images in {time.time() - t0:.0f}s", flush=True)

    X = cached_embeddings(RUNS / "emb", ENCODER, [str(DATA / f) for f in df.file])
    Z = PCA(50, random_state=0).fit_transform(X)
    link = linkage(Z, method="ward")
    print(f"\n== {ENCODER}, n={len(df)}: H-S depth, classes with 30+ objects")
    print(f"{'level':>6} {'classes':>8} {'objects':>8} {'AMI kmeans':>11} {'AMI dendro':>11} {'purity':>7} {'majority':>9} {'probe':>6}", flush=True)
    rows = []
    for d in range(1, 7):
        lab = pd.Series(hs_level(df.notation, d))
        vc = lab[lab != ""].value_counts(); keep = lab.isin(vc[vc >= 30].index).values
        y = np.array(lab[keep].tolist(), dtype=object); k = len(set(y))
        if k < 2:
            break
        km = MiniBatchKMeans(k, n_init=5, random_state=0, batch_size=4096).fit_predict(Z[keep])
        dc = fcluster(link, k, criterion="maxclust")[keep]
        pr = cross_val_score(LogisticRegression(max_iter=2000, C=0.5), X[keep], y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        row = {"level": d, "classes": k, "objects": int(keep.sum()), "AMI_kmeans": score(y, km)["AMI"], "AMI_dendro": score(y, dc)["AMI"],
               "purity": purity(y, km), "majority": float(pd.Series(y).value_counts(normalize=True).iloc[0]), "probe": pr}
        rows.append(row)
        print(f"{d:>6} {k:>8} {keep.sum():>8} {row['AMI_kmeans']:>11.3f} {row['AMI_dendro']:>11.3f} {row['purity']:>7.2f} {row['majority']:>9.2f} {pr:>6.2f}", flush=True)
    pd.DataFrame(rows).to_csv(RUNS / "depth.csv", index=False)

    print("\n== region, country, museum on the full set (classes with 30+)")
    r = run_labels(X, df, ["region", "country", "provider"], ks=(5, 20, 50, 150), encoder_name=ENCODER, min_count=30)
    print(r.drop(columns=["encoder"]).to_string(index=False), flush=True)
    r.to_csv(RUNS / "origin.csv", index=False)
    print("\n== region inside each museum with 2+ regions of 100+ objects")
    for prov, g in df.groupby("provider"):
        vc = g.region[g.region != ""].value_counts(); keep = vc[vc >= 100]
        if len(keep) < 2:
            continue
        gg = g[g.region.isin(keep.index)]
        rr = run_labels(X[gg.index.values], gg.reset_index(drop=True), ["region"], ks=(len(keep),), encoder_name="", min_count=100)
        a = rr.iloc[0]
        print(f"  {prov[:45]:45s} n={len(gg):5d} regions={dict(keep)} AMI={a.best_AMI} probe={a.probe_acc} majority={a.majority}", flush=True)


if __name__ == "__main__":
    main()
