"""Level-by-level splitting against flat clustering, on the 30,050 objects with 6-digit codes (as in test_tree).

divisive  k-means into k1 groups at level 1 (k1 = number of level-1 classes with 30+ objects); at every deeper level each
          group is split by k-means into a number of parts proportional to its size (at least 1, groups under 5 objects
          per part are not split), so the partitions are nested and the total is about k_d. 2 variants: the global PCA 50
          for every split, and a fresh PCA 50 inside each branch.
flat      k-means at the same k per level (not nested), and its nestedness: the share of objects whose group at level d
          sits under the parent group that most of its group mates have at level d-1.
Ward and average-link cuts (nested) come from runs/tests/tree/cuts.csv.
Output: runs/tests/hier/levels.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score as ami

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/tests/hier"
LEVELS = [f"hs{d}" for d in range(1, 7)]
MIN = 30


def kept(y):
    vc = pd.Series(y).value_counts()
    return (pd.Series(y).map(vc).values >= MIN) & (y != "")


def allocate(groups, k_total):
    sizes = pd.Series(groups).value_counts()
    raw = sizes / sizes.sum() * k_total
    alloc = np.maximum(1, np.floor(raw)).astype(int)
    rem = int(k_total - alloc.sum())
    if rem > 0:
        for g in (raw - np.floor(raw)).sort_values(ascending=False).index[:rem]:
            alloc[g] += 1
    return alloc


def divisive(X, Z, ks, local_pca):
    parts = []
    cur = KMeans(ks[0], n_init=10, random_state=0).fit_predict(Z)
    parts.append(cur.copy())
    for k in ks[1:]:
        alloc = allocate(cur, k)
        new = np.zeros_like(cur)
        nxt = 0
        for g, kg in alloc.items():
            ii = np.where(cur == g)[0]
            if kg == 1 or len(ii) < 5 * kg:
                new[ii] = nxt
                nxt += 1
                continue
            F = PCA(min(50, len(ii) - 1), random_state=0).fit_transform(X[ii]) if local_pca else Z[ii]
            new[ii] = nxt + KMeans(kg, n_init=5, random_state=0).fit_predict(F)
            nxt += kg
        cur = new
        parts.append(cur.copy())
    return parts


def nestedness(parent, child):
    t = pd.DataFrame({"p": parent, "c": child})
    maj = t.groupby("c").p.agg(lambda s: s.value_counts().index[0])
    return float((parent == maj.loc[child].values).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    X = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")
    sel = (rec.hs6 != "").values
    X = X[sel]
    Y = {lv: np.array(rec[lv][sel].tolist(), dtype=object) for lv in LEVELS}
    keep = {lv: kept(Y[lv]) for lv in LEVELS}
    ks = [len(set(Y[lv][keep[lv]])) for lv in LEVELS]
    print(len(X), "objects; k per level", ks, flush=True)
    Z = PCA(50, random_state=0).fit_transform(X)
    flat = [KMeans(k, n_init=10, random_state=0).fit_predict(Z) for k in ks]
    print("flat done", flush=True)
    div_g = divisive(X, Z, ks, local_pca=False)
    print("divisive, global PCA done", flush=True)
    div_l = divisive(X, Z, ks, local_pca=True)
    print("divisive, local PCA done", flush=True)
    cuts = pd.read_csv(ROOT / "runs/tests/tree/cuts.csv").set_index("level") if (ROOT / "runs/tests/tree/cuts.csv").exists() else None
    rows = []
    for d, lv in enumerate(LEVELS):
        kp = keep[lv]
        r = {"level": lv, "k": ks[d], "groups_divisive_global": len(set(div_g[d])), "groups_divisive_local": len(set(div_l[d])),
             "AMI_flat_kmeans": round(float(ami(Y[lv][kp], flat[d][kp])), 3),
             "AMI_divisive_global_pca": round(float(ami(Y[lv][kp], div_g[d][kp])), 3),
             "AMI_divisive_local_pca": round(float(ami(Y[lv][kp], div_l[d][kp])), 3),
             "nestedness_flat": round(nestedness(flat[d - 1], flat[d]), 3) if d else 1.0}
        if cuts is not None and lv in cuts.index:
            r["AMI_Ward_cut"] = cuts.loc[lv, "AMI_Ward"]
            r["AMI_average_link_cut"] = cuts.loc[lv, "AMI_average-link cosine"]
        rows.append(r)
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "levels.csv", index=False)


if __name__ == "__main__":
    main()
