"""Do the blind groups fit non-Western instruments? Region of origin is the museum's place of origin (runs/clean/records.csv).

  refit     PCA 50 and k-means fit on the non-European objects alone: H-S levels, family, name, region, country and
            museum (classes with 30+ objects inside that set), probe and majority baseline; the same for Europe
  restrict  the k-means partition of the 27,348 same rows (all regions), scored on its non-European, European and
            unplaced rows separately
  regions   refit inside Africa, Asia and the Americas: H-S levels 1 and 3, name
Outputs: runs/tests/nonwestern/refit.csv, restrict.csv, regions.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/tests/nonwestern"
MIN = 30
LEVELS = [f"hs{d}" for d in range(1, 7)]


def kept(y, min_count=MIN):
    y = np.array(list(y), dtype=object)
    vc = pd.Series(y).value_counts()
    return (pd.Series(y).map(vc).values >= min_count) & (y != ""), y


def probe(X, y, seed=0):
    pred = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return round(float((pred == y).mean()), 3), round(float(ami(y, pred)), 3)


def evaluate(X, Z, y, tag, with_probe=True):
    keep, y = kept(y)
    yk = y[keep]
    k = len(set(yk))
    if k < 2:
        return None
    p = KMeans(k, n_init=10, random_state=0).fit_predict(Z[keep])
    r = {"label": tag, "n": int(keep.sum()), "classes": k, "AMI": round(float(ami(yk, p)), 3),
         "majority": round(float(pd.Series(yk).value_counts(normalize=True).iloc[0]), 3)}
    if with_probe:
        r["probe_acc"], r["probe_AMI"] = probe(X[keep], yk)
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    X = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")
    V = {c: np.array(rec[c].tolist(), dtype=object) for c in LEVELS + ["family", "name", "region", "country", "provider"]}
    region = V["region"]
    non_eu = (region != "") & (region != "Europe")
    eu = region == "Europe"
    print("non-European", int(non_eu.sum()), "European", int(eu.sum()), "unplaced", int((region == "").sum()), flush=True)

    rows = []
    for tag, sel in (("non-European", non_eu), ("Europe", eu)):
        Z = PCA(50, random_state=0).fit_transform(X[sel])
        for c in LEVELS + ["family", "name", "region", "country", "provider"]:
            if c == "region" and tag == "Europe":
                continue
            r = evaluate(X[sel], Z, V[c][sel], c, with_probe=c not in ("country", "provider"))
            if r:
                r["set"] = tag
                rows.append(r)
                print(tag, r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "refit.csv", index=False)

    ids = pd.read_csv(ROOT / "runs/clean/same_rows_ids.csv", dtype=str).id
    pos = dict(zip(rec.id, range(len(rec))))
    idx = np.array([pos[i] for i in ids])
    Zs = PCA(50, random_state=0).fit_transform(X[idx])
    rows = []
    for c in ("hs1", "hs3", "hs6", "family", "name"):
        keep, y = kept(V[c][idx])
        yk = y[keep]
        p = KMeans(len(set(yk)), n_init=10, random_state=0).fit_predict(Zs[keep])
        for tag, sel in (("non-European", non_eu), ("Europe", eu), ("unplaced", region == "")):
            pm = sel[idx][keep]
            if pm.sum() < 100:
                continue
            r = {"label": c, "rows": tag, "n": int(pm.sum()), "classes_present": len(set(yk[pm])), "AMI": round(float(ami(yk[pm], p[pm])), 3),
                 "majority": round(float(pd.Series(yk[pm]).value_counts(normalize=True).iloc[0]), 3)}
            rows.append(r)
            print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "restrict.csv", index=False)

    rows = []
    for reg in ("Africa", "Asia", "Americas", "Oceania"):
        sel = region == reg
        if sel.sum() < 300:
            continue
        Z = PCA(50, random_state=0).fit_transform(X[sel])
        for c in ("hs1", "hs3", "family", "name", "country", "provider"):
            r = evaluate(X[sel], Z, V[c][sel], c, with_probe=c not in ("country", "provider"))
            if r:
                r["region"] = reg
                rows.append(r)
                print(reg, r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "regions.csv", index=False)


if __name__ == "__main__":
    main()
