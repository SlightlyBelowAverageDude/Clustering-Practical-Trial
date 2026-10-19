"""MIMO, all records: region, country and museum scores on the full set, from the cached DINOv2-B embeddings.

Second half of mimo_full.py, rerun on its own. Ward linkage on 63k points is too heavy, so only k-means is used
(KMeans on the 50-dim PCA projection, n_init=10). Outputs go to runs/mimo_full_origin.
"""
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
SRC, RUNS = ROOT / "runs/mimo_full", ROOT / "runs/mimo_full_origin"
ENCODER = sys.argv[1] if len(sys.argv) > 1 else "dinov2_b"
HS1 = {"1": "Idiophone", "2": "Membranophone", "3": "Chordophone", "4": "Aerophone", "5": "Electrophone"}


def kmeans_scores(X, Z, y, ks, min_count, seed=0):
    """Best k-means AMI over ks, AMI at k = classes, linear probe, majority. Rare label values dropped."""
    y = np.array(pd.Series(y).astype(str).tolist(), dtype=object)
    vc = pd.Series(y).value_counts()
    keep = (pd.Series(y).map(vc) >= min_count).values & (y != "") & (y != "nan")
    Xc, Zc, yc = X[keep], Z[keep], y[keep]
    k_cls = len(set(yc))
    if k_cls < 2:
        return {"n": int(keep.sum()), "classes": k_cls}
    res = {k: score(yc, KMeans(k, n_init=10, random_state=seed).fit_predict(Zc))["AMI"] for k in sorted(set(ks) | {k_cls})}
    best_k = max(res, key=res.get)
    probe = cross_val_score(LogisticRegression(max_iter=3000, C=0.5), Xc, yc, cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean()
    return {"n": int(keep.sum()), "classes": k_cls, "best_AMI": round(res[best_k], 3), "best_k": best_k,
            "AMI_at_k_classes": round(res[k_cls], 3), "probe_acc": round(float(probe), 3),
            "majority": round(float(pd.Series(yc).value_counts(normalize=True).iloc[0]), 3)}


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC / "sample_full.csv", dtype={"notation": str}).fillna("")
    X = np.load(SRC / "emb" / f"{ENCODER}.npy")
    assert len(X) == len(df), (len(X), len(df))
    df["hs1"] = df.notation.str[:1].map(HS1).fillna("")
    Z = PCA(50, random_state=0).fit_transform(X)
    print(f"{len(df)} images; with region {(df.region != '').sum()}; with H-S {(df.hs1 != '').sum()}; museums {df.provider.nunique()}", flush=True)

    print(f"\n== {ENCODER}: origin and museum on the full set (classes with 30+), k-means only")
    rows = []
    for col in ("region", "country", "provider", "hs1"):
        r = {"label": col, **kmeans_scores(X, Z, df[col], ks=(5, 20, 50, 150), min_count=30)}
        rows.append(r)
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(RUNS / "origin.csv", index=False)

    print("\n== region inside each H-S class (regions with 100+ objects in that class)")
    rows = []
    for cls, g in df[df.hs1 != ""].groupby("hs1"):
        vc = g.region[g.region != ""].value_counts(); keep = vc[vc >= 100]
        if len(keep) < 2:
            continue
        gg = g[g.region.isin(keep.index)]
        r = {"hs1": cls, **kmeans_scores(X[gg.index.values], Z[gg.index.values], gg.region, ks=(len(keep),), min_count=100)}
        rows.append(r)
        print(f"  {cls:14s} regions={dict(keep)} AMI={r['best_AMI']} probe={r['probe_acc']} majority={r['majority']}", flush=True)
    pd.DataFrame(rows).to_csv(RUNS / "region_within_hs1.csv", index=False)

    print("\n== region inside each museum with 2+ regions of 100+ objects")
    rows = []
    for prov, g in df.groupby("provider"):
        vc = g.region[g.region != ""].value_counts(); keep = vc[vc >= 100]
        if len(keep) < 2:
            continue
        gg = g[g.region.isin(keep.index)]
        r = {"provider": prov, **kmeans_scores(X[gg.index.values], Z[gg.index.values], gg.region, ks=(len(keep),), min_count=100)}
        rows.append(r)
        print(f"  {prov[:45]:45s} n={len(gg):5d} regions={dict(keep)} AMI={r['best_AMI']} probe={r['probe_acc']} majority={r['majority']}", flush=True)
    pd.DataFrame(rows).to_csv(RUNS / "region_within_museum.csv", index=False)

    print("\n== H-S level 1 inside each museum with 2+ classes of 100+ objects (photography control)")
    rows = []
    for prov, g in df[df.hs1 != ""].groupby("provider"):
        vc = g.hs1.value_counts(); keep = vc[vc >= 100]
        if len(keep) < 2:
            continue
        gg = g[g.hs1.isin(keep.index)]
        r = {"provider": prov, **kmeans_scores(X[gg.index.values], Z[gg.index.values], gg.hs1, ks=(len(keep),), min_count=100)}
        rows.append(r)
        print(f"  {prov[:45]:45s} n={len(gg):5d} AMI={r['best_AMI']} probe={r['probe_acc']} majority={r['majority']}", flush=True)
    pd.DataFrame(rows).to_csv(RUNS / "hs1_within_museum.csv", index=False)


if __name__ == "__main__":
    main()
