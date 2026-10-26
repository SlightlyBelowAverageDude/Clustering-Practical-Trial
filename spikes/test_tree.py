"""Tree against tree: one blind hierarchy over the instrument photos, scored against the Hornbostel-Sachs tree as a whole.

Objects: the 30,050 clean records with a code of at least 6 digits, so every object has all 6 levels.
Blind trees: Ward on PCA 50 of padded DINOv2-B, and average-link with cosine distance on the same projection.
Scores:
  purity   dendrogram purity per H-S level (also family and name): for pairs of objects of the same class, the share of
           that class among the leaves under the pair's lowest common ancestor; 1 for a tree that keeps every class
           together, the shuffled-label value is the chance level on the same tree
  rank     Spearman correlation over 200,000 random pairs between the height at which the blind tree joins the pair and
           the H-S distance (6 minus the number of shared leading levels); references: plain cosine distance without a
           tree, and permuted codes
  cuts     AMI of the tree cut at k = number of classes (30+ objects) at each level, next to k-means
  stable   10 subsamples of 80 percent: Ward tree and k-means on each, AMI against H-S per subsample, and AMI between
           the cuts of every 2 subsamples on their shared objects
Outputs: runs/tests/tree/*.csv
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/tests/tree"
LEVELS = [f"hs{d}" for d in range(1, 7)]
MIN = 30


def lca(Zl, N, qi, qj):
    """Lowest common ancestor (node id N..2N-2) of each leaf pair, offline: union-find over the merges, pending
    queries kept per cluster and merged small into large."""
    parent = np.arange(N)

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    pending = {}
    for q, (a, b) in enumerate(zip(qi, qj)):
        pending.setdefault(int(a), []).append(q)
        pending.setdefault(int(b), []).append(q)
    root_of = {i: i for i in range(N)}  # node id -> union-find root
    res = np.full(len(qi), -1, dtype=np.int64)
    size = np.ones(N, dtype=np.int64)
    for m in range(N - 1):
        a, b = int(Zl[m, 0]), int(Zl[m, 1])
        ra, rb = root_of.pop(a), root_of.pop(b)
        la, lb = pending.pop(a, []), pending.pop(b, [])
        if len(la) < len(lb):
            la, lb, ra, rb = lb, la, rb, ra
        for q in lb:
            if res[q] < 0:
                fx, fy = find(int(qi[q])), find(int(qj[q]))
                if (fx == ra and fy == rb) or (fx == rb and fy == ra):
                    res[q] = N + m
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        root_of[N + m] = ra
        la.extend(q for q in lb if res[q] < 0)
        pending[N + m] = la
    return res


def class_counts(Zl, N, y_int, C):
    cnt = np.zeros((2 * N - 1, C), dtype=np.int32)
    ok = y_int >= 0
    cnt[np.where(ok)[0], y_int[ok]] = 1
    for m in range(N - 1):
        cnt[N + m] = cnt[int(Zl[m, 0])] + cnt[int(Zl[m, 1])]
    return cnt


def node_sizes(Zl, N):
    return np.concatenate([np.ones(N), Zl[:, 3]])


def same_class_pairs(y_int, n, rng):
    cls, counts = np.unique(y_int[y_int >= 0], return_counts=True)
    w = counts * (counts - 1.0)
    w /= w.sum()
    members = {c: np.where(y_int == c)[0] for c in cls}
    pick = rng.choice(len(cls), n, p=w)
    qi, qj = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    for t, ci in enumerate(pick):
        a, b = rng.choice(members[cls[ci]], 2, replace=False)
        qi[t], qj[t] = a, b
    return qi, qj


def encode(y):
    y = pd.Series(y).astype(str)
    vc = y.value_counts()
    ok = (y != "") & (y.map(vc) >= 2)
    codes = {c: i for i, c in enumerate(sorted(set(y[ok])))}
    return np.array([codes.get(v, -1) if o else -1 for v, o in zip(y, ok)]), len(codes)


def purity(Zl, N, y, rng, n_pairs=50000):
    y_int, C = encode(y)
    qi, qj = same_class_pairs(y_int, n_pairs, rng)
    node = lca(Zl, N, qi, qj)
    cnt = class_counts(Zl, N, y_int, C)
    return float((cnt[node, y_int[qi]] / node_sizes(Zl, N)[node]).mean())


def kept(y):
    y = np.array(pd.Series(y).astype(str).tolist(), dtype=object)
    vc = pd.Series(y).value_counts()
    return (pd.Series(y).map(vc).values >= MIN) & (y != ""), y


def hs_distance(V, i, j):
    shared = np.zeros(len(i), dtype=int)
    alive = np.ones(len(i), dtype=bool)
    for lv in LEVELS:
        v = V[lv]
        alive &= v[i] == v[j]
        shared += alive
    return 6 - shared


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    X = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")
    sel = (rec.hs6 != "").values
    df, X = rec[sel].reset_index(drop=True), X[sel]
    V = {c: np.array(df[c].tolist(), dtype=object) for c in df.columns}
    N = len(df)
    print(N, "objects with 6-digit codes", flush=True)
    Z = PCA(50, random_state=0).fit_transform(X)
    trees = {"Ward": linkage(Z, method="ward")}
    print("Ward done", flush=True)
    trees["average-link cosine"] = linkage(Z, method="average", metric="cosine")
    print("average-link done", flush=True)
    rng = np.random.default_rng(0)

    # purity
    rows = []
    for name, Zl in trees.items():
        for lab in LEVELS + ["family", "name"]:
            y = V[lab]
            r = {"tree": name, "label": lab, "classes": int(len(set(y[y != ""]))), "purity": round(purity(Zl, N, y, rng), 3),
                 "purity_shuffled_labels": round(purity(Zl, N, rng.permutation(y), rng), 3)}
            rows.append(r)
            print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "purity.csv", index=False)

    # rank correlation with the H-S distance
    n = 200000
    i, j = rng.integers(0, N, n), rng.integers(0, N, n)
    m = i != j
    i, j = i[m], j[m]
    d_hs = hs_distance(V, i, j)
    perm = rng.permutation(N)
    d_perm = hs_distance({lv: V[lv][perm] for lv in LEVELS}, i, j)
    rows = []
    for name, Zl in trees.items():
        h = Zl[lca(Zl, N, i, j) - N, 2]
        rows.append({"distance": f"{name} tree height", "spearman_vs_HS_distance": round(float(spearmanr(h, d_hs)[0]), 3),
                     "spearman_vs_permuted_codes": round(float(spearmanr(h, d_perm)[0]), 3)})
        print(rows[-1], flush=True)
    U = normalize(X)
    cosd = 1 - (U[i] * U[j]).sum(1)
    rows.append({"distance": "cosine distance, no tree", "spearman_vs_HS_distance": round(float(spearmanr(cosd, d_hs)[0]), 3),
                 "spearman_vs_permuted_codes": round(float(spearmanr(cosd, d_perm)[0]), 3)})
    nm = (V["name"][i] != "") & (V["name"][j] != "")
    diff_name = (V["name"][i] != V["name"][j]).astype(float)
    rows.append({"distance": "different instrument name (0 or 1), reference", "spearman_vs_HS_distance": round(float(spearmanr(diff_name[nm], d_hs[nm])[0]), 3),
                 "spearman_vs_permuted_codes": round(float(spearmanr(diff_name[nm], d_perm[nm])[0]), 3)})
    print(rows[-2:], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "rank.csv", index=False)

    # cuts at k = classes, next to k-means
    rows = []
    for lv in LEVELS:
        keep, y = kept(V[lv])
        k = len(set(y[keep]))
        r = {"level": lv, "classes": k, "objects": int(keep.sum())}
        for name, Zl in trees.items():
            r[f"AMI_{name}"] = round(float(ami(y[keep], fcluster(Zl, k, criterion="maxclust")[keep])), 3)
        r["AMI_k-means"] = round(float(ami(y[keep], KMeans(k, n_init=10, random_state=0).fit_predict(Z[keep]))), 3)
        rows.append(r)
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "cuts.csv", index=False)
    del trees

    # stability over 10 subsamples of 80 percent
    ks = {lv: len(set(kept(V[lv])[1][kept(V[lv])[0]])) for lv in LEVELS}
    subs, cuts, kms, per = [], [], [], []
    for s in range(10):
        r_ = np.random.default_rng(100 + s)
        idx = np.sort(r_.choice(N, int(0.8 * N), replace=False))
        Zs = PCA(50, random_state=s).fit_transform(X[idx])
        Zl = linkage(Zs, method="ward")
        c_s, k_s = {}, {}
        for lv in LEVELS:
            keep, y = kept(V[lv][idx])
            c_s[lv] = fcluster(Zl, ks[lv], criterion="maxclust")
            k_s[lv] = KMeans(ks[lv], n_init=10, random_state=s).fit_predict(Zs)
            per.append({"subsample": s, "level": lv, "AMI_Ward": round(float(ami(y[keep], c_s[lv][keep])), 3),
                        "AMI_k-means": round(float(ami(y[keep], k_s[lv][keep])), 3),
                        "purity_Ward": round(purity(Zl, len(idx), V[lv][idx], r_, 20000), 3)})
        subs.append(idx)
        cuts.append(c_s)
        kms.append(k_s)
        print("subsample", s, "done", flush=True)
    per = pd.DataFrame(per)
    per.to_csv(OUT / "stability_per_subsample.csv", index=False)
    rows = []
    for lv in LEVELS:
        st_w, st_k = [], []
        for a, b in combinations(range(10), 2):
            common_ids, ia, ib = np.intersect1d(subs[a], subs[b], return_indices=True)
            st_w.append(ami(cuts[a][lv][ia], cuts[b][lv][ib]))
            st_k.append(ami(kms[a][lv][ia], kms[b][lv][ib]))
        p = per[per.level == lv]
        rows.append({"level": lv, "k": ks[lv],
                     "Ward_AMI_vs_HS_mean": round(p.AMI_Ward.mean(), 3), "Ward_AMI_vs_HS_min": p.AMI_Ward.min(), "Ward_AMI_vs_HS_max": p.AMI_Ward.max(),
                     "kmeans_AMI_vs_HS_mean": round(p["AMI_k-means"].mean(), 3), "kmeans_AMI_vs_HS_min": p["AMI_k-means"].min(), "kmeans_AMI_vs_HS_max": p["AMI_k-means"].max(),
                     "Ward_cut_agreement_mean": round(float(np.mean(st_w)), 3), "Ward_cut_agreement_min": round(float(np.min(st_w)), 3),
                     "kmeans_agreement_mean": round(float(np.mean(st_k)), 3), "kmeans_agreement_min": round(float(np.min(st_k)), 3),
                     "Ward_purity_mean": round(p.purity_Ward.mean(), 3), "Ward_purity_min": p.purity_Ward.min(), "Ward_purity_max": p.purity_Ward.max()})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "stability.csv", index=False)


if __name__ == "__main__":
    main()
