"""MIMO: depth table for the Wikidata instrument taxonomy (P279 tree under "musical instrument", Q34379), on the same
rows as the H-S depth table, plus taxonomy-vs-taxonomy agreement as the reference for "how good is AMI x".

Level d of an item = its ancestor at shortest-path depth d from the root (the most frequent one when the DAG offers
several). Inputs: runs/mimo_full, runs/mimo_names/labels.csv, dl/wd_mimo_instruments.csv, dl/wd_mimo_ancestors.csv,
dl/wd_instrument_edges.csv. Outputs: runs/mimo_wd_depth.
"""
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hs_level, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs/mimo_wd_depth")
ROOT_ITEM = "Q34379"
MIN_COUNT = 30


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "runs/mimo_full/sample_full.csv", dtype={"notation": str}).fillna("")
    lab = pd.read_csv(ROOT / "runs/mimo_names/labels.csv", dtype=str).fillna("")
    assert (lab.id.values == df.id.values).all()
    df["kw"], df["name"] = lab.kw.values, lab.name.values
    X = np.load(ROOT / "runs/mimo_full/emb/dinov2_b.npy")
    Z = PCA(50, random_state=0).fit_transform(X)
    wd = pd.read_csv(ROOT / "dl/wd_mimo_instruments.csv", dtype=str).fillna("")
    anc = pd.read_csv(ROOT / "dl/wd_mimo_ancestors.csv", dtype=str).fillna("")
    edges = pd.read_csv(ROOT / "dl/wd_instrument_edges.csv", dtype=str).fillna("")
    label_of = dict(zip(anc.anc, anc.ancLabel))
    label_of.update(dict(zip(wd.item.str.replace("http://www.wikidata.org/entity/", ""), wd.itemLabel)))
    label_of.update(dict(zip(edges.o, edges.oLabel)))
    label_of[ROOT_ITEM] = "musical instrument"

    children = defaultdict(set)
    for s, o in zip(edges.s, edges.o):
        children[o].add(s)
    depth = {ROOT_ITEM: 0}
    q = deque([ROOT_ITEM])
    while q:
        u = q.popleft()
        for c in children[u]:
            if c not in depth:
                depth[c] = depth[u] + 1
                q.append(c)
    ancestors = defaultdict(set)
    for it, a in zip(anc.item, anc.anc):
        ancestors[it].add(a)
    item_of = dict(zip(wd.mimo, wd.item.str.replace("http://www.wikidata.org/entity/", "")))
    df["item"] = df.kw.map(item_of).fillna("")
    print("items with depth:", sum(1 for i in set(df.item) if i in depth), "of", df.item[df.item != ""].nunique(),
          "; max depth", max(depth.values()))

    item_counts = df.item.value_counts()
    pop = defaultdict(int)  # how many records sit under each node
    for it, n in item_counts.items():
        if it == "":
            continue
        for a in ancestors[it] | {it}:
            pop[a] += n
    max_d = 7
    for d in range(1, max_d + 1):
        col = []
        for it in df.item:
            if it == "" or it not in depth:
                col.append(""); continue
            cands = [a for a in ancestors[it] | {it} if depth.get(a) == d]
            if not cands or depth[it] < d:
                col.append(""); continue
            col.append(max(cands, key=lambda a: pop[a]))
        df[f"wd{d}"] = col
    for d in range(1, 7):
        df[f"hs{d}"] = hs_level(df.notation, d)

    def run(col, idx, tag):
        y = np.array(df.loc[idx, col].tolist(), dtype=object)
        vc = pd.Series(y).value_counts()
        keep = (pd.Series(y).map(vc) >= MIN_COUNT).values & (y != "")
        ii = np.asarray(idx)[keep]; yc = y[keep]; k = len(set(yc))
        if k < 2:
            return None
        km = KMeans(k, n_init=3, random_state=0).fit_predict(Z[ii])
        pr = cross_val_score(LogisticRegression(max_iter=1000, C=0.5), X[ii], yc, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        r = {"set": tag, "label": col, "n": len(yc), "classes": k, "AMI": round(score(yc, km)["AMI"], 3), "probe": round(float(pr), 3),
             "majority": round(float(pd.Series(yc).value_counts(normalize=True).iloc[0]), 3)}
        print(r, flush=True)
        return r, ii, yc, km

    # rows with both taxonomies to depth 3
    both = (df.wd3 != "") & (df.hs3 != "")
    idx = df.index[both].values
    print(f"\n== rows with Wikidata depth 3 and H-S depth 3: {len(idx)}")
    rows, parts = [], {}
    for d in range(1, max_d + 1):
        r = run(f"wd{d}", idx, "same rows")
        if r:
            rows.append(r[0]); parts[f"wd{d}"] = r
    for d in range(1, 7):
        r = run(f"hs{d}", idx, "same rows")
        if r:
            rows.append(r[0]); parts[f"hs{d}"] = r
    r = run("name", idx, "same rows")
    if r:
        rows.append(r[0]); parts["name"] = r
    pd.DataFrame(rows).to_csv(OUT / "depth_same_rows.csv", index=False)

    print("\n== taxonomy vs taxonomy (AMI between label sets, rows where both are defined with 30+ per class)")
    pairs = [("wd1", "hs1"), ("wd2", "hs2"), ("wd3", "hs3"), ("wd4", "hs4"), ("wd5", "hs5"), ("wd2", "hs1"), ("wd3", "hs2"), ("wd4", "hs3"),
             ("name", "hs3"), ("name", "hs4"), ("name", "wd3"), ("name", "wd4"), ("wd3", "hs4"), ("wd2", "hs3")]
    out = []
    for a, b in pairs:
        if a not in parts or b not in parts:
            continue
        ia, ib = parts[a][1], parts[b][1]
        common = np.intersect1d(ia, ib)
        ya = pd.Series(parts[a][2], index=ia).loc[common].values
        yb = pd.Series(parts[b][2], index=ib).loc[common].values
        v = round(float(ami(ya, yb)), 3)
        out.append({"a": a, "b": b, "n": len(common), "classes_a": len(set(ya)), "classes_b": len(set(yb)), "AMI": v})
        print(f"  {a:5s} vs {b:5s} n={len(common):6d} classes {len(set(ya)):4d} / {len(set(yb)):4d}  AMI {v:.3f}")
    pd.DataFrame(out).to_csv(OUT / "taxonomy_vs_taxonomy.csv", index=False)

    print("\n== Wikidata level labels and their sizes (level 2 and 3)")
    for d in (2, 3):
        vc = df.loc[idx, f"wd{d}"].value_counts()
        vc = vc[vc >= MIN_COUNT]
        print(f"  level {d}: " + "; ".join(f"{label_of.get(k, k)} {v}" for k, v in vc.items()))
    df[["id", "item", "name"] + [f"wd{d}" for d in range(1, max_d + 1)]].to_csv(OUT / "labels.csv", index=False)
    pd.Series(label_of).to_csv(OUT / "node_labels.csv", header=["label"])


if __name__ == "__main__":
    main()
