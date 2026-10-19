"""Second-pass checks on the clean results. Output: runs/clean_recheck/.

1. Probe leakage: near-duplicate photos (sets of identical objects, the same object photographed twice) can sit in both the
   training and the test fold. Rerun the probe with grouped folds, groups = connected components of cosine > 0.95.
2. Purity baseline for blind-then-label: the same cluster sizes with shuffled assignments.
3. Figure consistency: reproduce the k = 20 clustering of clean_analysis block 5 and the k-means/average-link 6-group
   clusterings of the methods subset, save the assignments so the figures can use exactly the analysed partitions.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
RUN, OUT = ROOT / "runs/clean", ROOT / "runs/clean_recheck"
HS1 = {"1": "Idiophone", "2": "Membranophone", "3": "Chordophone", "4": "Aerophone", "5": "Electrophone"}


def near_dup_groups(X, thr=0.95, block=4096):
    n = len(X); rows, cols = [], []
    for s in range(0, n, block):
        S = X[s:s + block] @ X.T
        i, j = np.where(S > thr)
        rows.append(i + s); cols.append(j)
    r, c = np.concatenate(rows), np.concatenate(cols)
    _, lab = connected_components(csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n)), directed=False)
    return lab


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RUN / "records.csv", dtype=str).fillna("")
    X = np.load(RUN / "emb/dinov2_b_pad.npy")
    df["hs1n"] = df.hs1.map(HS1).fillna("")

    # same rows as clean_analysis block 2
    m = (df.hs3 != "") & (df.family != "") & (df.name != "")
    sub = df[m]
    for col in ("hs3", "family", "name"):
        vc = sub[col].value_counts(); sub = sub[sub[col].map(vc) >= 30]
    idx = sub.index.values; Xs = X[idx]
    groups = near_dup_groups(Xs)
    sizes = pd.Series(groups).value_counts()
    print(f"same rows {len(sub)}: near-duplicate components {int((sizes > 1).sum())}, objects in them {int(sizes[sizes > 1].sum())}, largest {int(sizes.max())}", flush=True)
    rows = []
    for col in ("hs3", "family", "name"):
        y = np.array(sub[col].tolist(), dtype=object)
        lr = LogisticRegression(max_iter=2000, C=0.5)
        p1 = cross_val_predict(lr, Xs, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
        p2 = cross_val_predict(lr, Xs, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0), groups=groups)
        r = {"label": col, "probe_acc_plain": round(float((p1 == y).mean()), 3), "probe_AMI_plain": round(float(ami(y, p1)), 3),
             "probe_acc_grouped": round(float((p2 == y).mean()), 3), "probe_AMI_grouped": round(float(ami(y, p2)), 3)}
        rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "probe_grouped.csv", index=False)

    # blind then label: reproduce block 5 and add a shuffled-assignment baseline
    m2 = (df.name != "") & (df.hs3 != "")
    s2 = df[m2]; vc = s2.name.value_counts(); s2 = s2[s2.name.map(vc) >= 30]
    Z2 = PCA(50, random_state=0).fit_transform(X[s2.index.values])
    rng = np.random.default_rng(0)
    rows = []
    for k in (20, 100, 400):
        c = KMeans(k, n_init=3, random_state=0).fit_predict(Z2)
        if k == 20:
            pd.DataFrame({"id": s2.id.values, "cluster": c}).to_csv(OUT / "k20_assignments.csv", index=False)
        for tag, cc in (("k-means", c), ("shuffled, same sizes", rng.permutation(c))):
            t = pd.DataFrame({"c": cc, "name": s2.name.values, "hs3": s2.hs3.values})
            g = t.groupby("c")
            share = g.name.agg(lambda s: s.value_counts(normalize=True).iloc[0])
            dom, dom3 = g.name.agg(lambda s: s.value_counts().index[0]), g.hs3.agg(lambda s: s.value_counts().index[0])
            r = {"k": k, "assignment": tag, "objects_right_name": round(float((t.name.values == dom[cc].values).mean()), 3),
                 "objects_right_hs3": round(float((t.hs3.values == dom3[cc].values).mean()), 3), "groups_nameable": int((share >= 0.5).sum())}
            rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "blind_then_label_baseline.csv", index=False)
    old = pd.read_csv(RUN / "crossing_names.csv")
    t = pd.DataFrame({"c": pd.read_csv(OUT / "k20_assignments.csv").cluster.values, "name": s2.name.values})
    chk = t.groupby("name").c.agg(lambda s: s.value_counts().index[0])
    same = (old.set_index("name").main_cluster.reindex(chk.index) == chk).mean()
    print(f"k20 reproduction: main cluster per name identical for {same:.3f} of names", flush=True)

    # methods subset: reproduce the 6-group partitions used for the family numbers
    ids = pd.read_csv(RUN / "methods_subset_ids.csv", dtype=str).id
    pos = {i: p for p, i in enumerate(sub.id.values)}
    pick = np.array([pos[i] for i in ids])
    Zm = PCA(50, random_state=0).fit_transform(Xs[pick])
    fam = sub.family.values[pick]
    km6 = KMeans(6, n_init=10, random_state=0).fit_predict(Zm)
    al6 = AgglomerativeClustering(6, linkage="average", metric="cosine").fit_predict(Zm)
    print(f"methods subset {len(pick)}: family AMI k-means 6 groups {ami(fam, km6):.3f}, average-link 6 groups {ami(fam, al6):.3f}", flush=True)
    pd.DataFrame({"id": ids.values, "km6": km6, "al6": al6}).to_csv(OUT / "methods_k6_assignments.csv", index=False)


if __name__ == "__main__":
    main()
