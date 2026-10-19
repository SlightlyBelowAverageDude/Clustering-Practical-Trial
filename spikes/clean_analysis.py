"""All instrument analyses on the cleaned MIMO set (runs/clean/records.csv, padded DINOv2-B embeddings).

Outputs in runs/clean/: depth.csv (H-S levels 1 to 6, coded rows), same_rows.csv (H-S levels, family, name on the rows
that carry all three), taxonomy_vs_taxonomy.csv, origin.csv (region, country, museum, hs1 on the full clean set),
within_museum.csv (hs1 and region inside each museum), blind_then_label.csv, crossing_names.csv (k = 20),
methods.csv (k-means, Ward, average-link cosine, UMAP + k-means and the probe ceiling on a 12,000-row subset).
Pass 'kmeans' as the first argument to skip the Ward dendrogram on the full coded set.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/clean"
WITH_DENDRO = not (len(sys.argv) > 1 and sys.argv[1] == "kmeans")
HS1 = {"1": "Idiophone", "2": "Membranophone", "3": "Chordophone", "4": "Aerophone", "5": "Electrophone"}
MIN = 30


def kept(y, min_count=MIN):
    y = np.array(pd.Series(y).astype(str).tolist(), dtype=object)
    vc = pd.Series(y).value_counts()
    return (pd.Series(y).map(vc).values >= min_count) & (y != "") & (y != "nan"), y


def probe(X, y, seed=0):
    pred = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return round(float((pred == y).mean()), 3), round(float(ami(y, pred)), 3)


def evaluate(X, Z, y, ks=(), tag="", with_probe=True, n_init=10):
    keep, y = kept(y)
    yk, Xk, Zk = y[keep], X[keep], Z[keep]; k = len(set(yk))
    if k < 2:
        return None
    res = {kk: round(score(yk, KMeans(kk, n_init=n_init, random_state=0).fit_predict(Zk))["AMI"], 3) for kk in sorted(set(ks) | {k})}
    r = {"label": tag, "n": int(keep.sum()), "classes": k, "AMI_at_classes": res[k], "best_AMI": max(res.values()), "best_k": max(res, key=res.get),
         "majority": round(float(pd.Series(yk).value_counts(normalize=True).iloc[0]), 3)}
    if with_probe:
        r["probe_acc"], r["probe_AMI"] = probe(Xk, yk)
    print(r, flush=True)
    return r


def main():
    df = pd.read_csv(OUT / "records.csv", dtype=str).fillna("")
    X = np.load(OUT / "emb/dinov2_b_pad.npy")
    assert len(X) == len(df), (len(X), len(df))
    df["hs1n"] = df.hs1.map(HS1).fillna("")
    print(f"{len(df)} clean records, embeddings {X.shape}", flush=True)

    # 1. depth table on the coded rows
    coded = (df.hs1 != "").values
    Xc, dfc = X[coded], df[coded].reset_index(drop=True)
    Zc = PCA(50, random_state=0).fit_transform(Xc)
    link = linkage(Zc, method="ward") if WITH_DENDRO else None
    print("\n== depth table on", len(dfc), "coded objects", flush=True)
    rows = []
    for d in range(1, 7):
        keep, y = kept(dfc[f"hs{d}"])
        yk = y[keep]; k = len(set(yk))
        km = KMeans(k, n_init=5, random_state=0).fit_predict(Zc[keep])
        r = {"level": d, "classes": k, "objects": int(keep.sum()), "AMI_kmeans": round(score(yk, km)["AMI"], 3),
             "AMI_dendro": round(score(yk, fcluster(link, k, criterion="maxclust")[keep])["AMI"], 3) if WITH_DENDRO else None,
             "majority": round(float(pd.Series(yk).value_counts(normalize=True).iloc[0]), 3)}
        r["probe_acc"], r["probe_AMI"] = probe(Xc[keep], yk)
        rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "depth.csv", index=False)

    # 2. same rows: H-S levels, family and name on the rows that carry all three (classes with 30+ inside that set)
    m = (df.hs3 != "") & (df.family != "") & (df.name != "")
    sub = df[m]
    for col in ("hs3", "family", "name"):
        vc = sub[col].value_counts(); sub = sub[sub[col].map(vc) >= MIN]
    idx = sub.index.values
    sub.to_csv(OUT / "same_rows_ids.csv", index=False)
    Xs = X[idx]; Zs = PCA(50, random_state=0).fit_transform(Xs)
    print("\n== same rows:", len(sub), flush=True)
    rows = [evaluate(Xs, Zs, sub[c].values, tag=c) for c in ("hs1", "hs2", "hs3", "hs4", "hs5", "hs6", "family", "name")]
    pd.DataFrame([r for r in rows if r]).to_csv(OUT / "same_rows.csv", index=False)
    pairs = []
    for a, b in (("name", "hs3"), ("name", "hs4"), ("name", "family"), ("family", "hs1"), ("family", "hs2"), ("hs6", "hs3")):
        ka, ya = kept(sub[a].values); kb, yb = kept(sub[b].values); both = ka & kb
        pairs.append({"a": a, "b": b, "n": int(both.sum()), "classes_a": len(set(ya[both])), "classes_b": len(set(yb[both])), "AMI": round(float(ami(ya[both], yb[both])), 3)})
        print(pairs[-1], flush=True)
    pd.DataFrame(pairs).to_csv(OUT / "taxonomy_vs_taxonomy.csv", index=False)

    # 3. origin and museum on the full clean set
    Z = PCA(50, random_state=0).fit_transform(X)
    print("\n== origin and museum, full clean set", flush=True)
    rows = [evaluate(X, Z, df[c].values, ks=(5, 20, 50, 150), tag=c) for c in ("region", "country", "provider", "hs1n")]
    pd.DataFrame([r for r in rows if r]).to_csv(OUT / "origin.csv", index=False)

    # 4. within each museum: hs1 and region
    print("\n== within museums", flush=True)
    rows = []
    for prov, g in df.groupby("provider"):
        for col in ("hs1n", "region"):
            vc = g[col][g[col] != ""].value_counts(); vc = vc[vc >= 100]
            if len(vc) < 2:
                continue
            gg = g[g[col].isin(vc.index)]
            r = evaluate(X[gg.index.values], Z[gg.index.values], gg[col].values, tag=col, with_probe=True)
            if r:
                r["provider"] = prov; rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "within_museum.csv", index=False)

    # 5. blind groups, then label by majority (rows with name and hs3)
    m2 = (df.name != "") & (df.hs3 != "")
    s2 = df[m2]; vc = s2.name.value_counts(); s2 = s2[s2.name.map(vc) >= MIN]
    Z2 = PCA(50, random_state=0).fit_transform(X[s2.index.values])
    print("\n== blind then label:", len(s2), "objects,", s2.name.nunique(), "names", flush=True)
    rows = []
    for k in (20, 50, 100, 200, 400):
        c = KMeans(k, n_init=3, random_state=0).fit_predict(Z2)
        t = pd.DataFrame({"c": c, "name": s2.name.values, "hs3": s2.hs3.values})
        g = t.groupby("c")
        dom, share, size = g.name.agg(lambda s: s.value_counts().index[0]), g.name.agg(lambda s: s.value_counts(normalize=True).iloc[0]), g.size()
        dom3 = g.hs3.agg(lambda s: s.value_counts().index[0])
        r = {"k": k, "objects_right_name": round(float((t.name.values == dom[c].values).mean()), 3), "objects_right_hs3": round(float((t.hs3.values == dom3[c].values).mean()), 3),
             "groups_nameable": int((share >= 0.5).sum()), "objects_in_nameable_groups": round(float(size[share >= 0.5].sum() / len(t)), 3), "distinct_names_used": int(dom.nunique())}
        rows.append(r); print(r, flush=True)
        if k == 20:
            dom1 = t.assign(hs1=s2.hs1n.values).groupby("c").hs1.agg(lambda s: s.value_counts().index[0])
            cr = []
            for n, gn in t.assign(hs1=s2.hs1n.values).groupby("name"):
                top = gn.c.value_counts(); cc = top.index[0]
                cr.append({"name": n, "n": len(gn), "hs1": gn.hs1.value_counts().index[0], "main_cluster": int(cc), "share_in_main": round(top.iloc[0] / len(gn), 2),
                           "cluster_dominant_hs1": dom1[cc]})
            cr = pd.DataFrame(cr); cr["crosses"] = cr.hs1 != cr.cluster_dominant_hs1
            cr.sort_values(["crosses", "n"], ascending=False).to_csv(OUT / "crossing_names.csv", index=False)
            print("   names (50+) whose main cluster is dominated by another H-S class:", int(cr[(cr.n >= 50)].crosses.sum()), "of", int((cr.n >= 50).sum()), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "blind_then_label.csv", index=False)

    # 6. methods on a 12,000-row subset of the same rows
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(np.arange(len(sub)), min(12000, len(sub)), replace=False))
    sm, Xm, Zm = sub.iloc[pick], Xs[pick], PCA(50, random_state=0).fit_transform(Xs[pick])
    pd.DataFrame({"id": sm.id.values}).to_csv(OUT / "methods_subset_ids.csv", index=False)
    print("\n== methods on", len(sm), "rows", flush=True)
    import umap
    U = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.0, random_state=0).fit_transform(Zm)
    labels = {c: kept(sm[c].values, 5) for c in ("hs3", "family", "name")}
    rows = []
    for method in ("k-means", "Ward", "average-link cosine", "UMAP 10-d + k-means", "probe (supervised ceiling)"):
        r = {"method": method}
        for c, (keep, y) in labels.items():
            yk = y[keep]; k = len(set(yk))
            if method == "k-means":
                p = KMeans(k, n_init=10, random_state=0).fit_predict(Zm[keep])
            elif method == "Ward":
                p = AgglomerativeClustering(k, linkage="ward").fit_predict(Zm[keep])
            elif method == "average-link cosine":
                p = AgglomerativeClustering(k, linkage="average", metric="cosine").fit_predict(Zm[keep])
            elif method == "UMAP 10-d + k-means":
                p = KMeans(k, n_init=10, random_state=0).fit_predict(U[keep])
            else:
                p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), Xm[keep], yk, cv=StratifiedKFold(5, shuffle=True, random_state=0))
            r[f"AMI_{c}"] = round(float(ami(yk, p)), 3)
        rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "methods.csv", index=False)


if __name__ == "__main__":
    main()
