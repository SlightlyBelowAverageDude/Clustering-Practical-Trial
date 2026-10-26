"""Per-node map: which Hornbostel-Sachs nodes and which instrument names does a blind tree reproduce as one subtree?

30,050 objects with 6-digit codes (as test_tree). Trees: average-link with cosine distance and Ward, both on PCA 50 of
the padded DINOv2-B embeddings; the linkage matrices are saved to runs/tests/tree/linkage_<tree>.npy and reused.
For every H-S node with 30+ objects at levels 1 to 6, and every name with 30+ objects, the tree node whose leaf set
matches the class best: F1 of precision (share of the subtree's leaves in the class) and recall (share of the class under
the subtree). Reference: the best single k-means cluster at k = number of classes of that level, scored the same way.
Outputs: runs/tests/nodes/nodes.csv (one row per class and method), summary.csv (per level and method: share of classes
and of objects with F1 at or above 0.5 and 0.7, mean F1 weighted by objects).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tree import class_counts, node_sizes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/tests/nodes"
TREE = ROOT / "runs/tests/tree"
LEVELS = [f"hs{d}" for d in range(1, 7)]
MIN = 30
HS_NAMES = {"1": "idiophones", "11": "struck idiophones", "111": "directly struck", "112": "indirectly struck", "12": "plucked idiophones",
            "121": "in the form of a frame (jaw harps)", "122": "in board or comb form (lamellophones)", "13": "friction idiophones", "14": "blown idiophones",
            "2": "membranophones", "21": "struck drums", "211": "directly struck", "212": "rattle drums", "22": "plucked drums", "23": "friction drums",
            "24": "singing membranes", "3": "chordophones", "31": "simple chordophones (zithers)", "311": "bar zithers", "312": "tube zithers",
            "313": "raft zithers", "314": "board zithers", "315": "trough zithers", "316": "frame zithers", "32": "composite chordophones",
            "321": "lutes", "322": "harps", "323": "harp-lutes", "4": "aerophones", "41": "free aerophones", "411": "displacement free aerophones",
            "412": "interruptive free aerophones (free reeds)", "413": "plosive aerophones", "42": "wind instruments proper",
            "421": "edge-blown (flutes)", "422": "reed pipes", "423": "trumpets (lip-vibrated)", "5": "electrophones", "51": "electromechanical",
            "52": "electronic", "53": "radioelectric"}


def classes_of(y, min_count=MIN):
    vc = pd.Series(y).value_counts()
    keep = sorted(c for c, n in vc.items() if c != "" and n >= min_count)
    cls = {c: i for i, c in enumerate(keep)}
    return np.array([cls.get(v, -1) for v in y]), keep


def best_subtree(Zl, N, y_int, C):
    cnt = class_counts(Zl, N, y_int, C).astype(np.float64)
    sizes = node_sizes(Zl, N)
    tot = np.bincount(y_int[y_int >= 0], minlength=C).astype(np.float64)
    f1 = 2 * cnt / (sizes[:, None] + tot[None, :])
    best = f1.argmax(0)
    j = np.arange(C)
    return f1[best, j], cnt[best, j] / sizes[best], cnt[best, j] / tot, sizes[best].astype(int), tot.astype(int)


def best_cluster(p, y_int, C):
    k = p.max() + 1
    cnt = np.zeros((k, C))
    m = y_int >= 0
    np.add.at(cnt, (p[m], y_int[m]), 1)
    sizes = np.bincount(p, minlength=k).astype(np.float64)
    tot = np.bincount(y_int[m], minlength=C).astype(np.float64)
    f1 = 2 * cnt / (sizes[:, None] + tot[None, :])
    best = f1.argmax(0)
    j = np.arange(C)
    return f1[best, j], cnt[best, j] / sizes[best], cnt[best, j] / tot, sizes[best].astype(int), tot.astype(int)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    X = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")
    sel = (rec.hs6 != "").values
    X = X[sel]
    N = len(X)
    V = {c: np.array(rec[c][sel].tolist(), dtype=object) for c in LEVELS + ["name"]}
    Z = PCA(50, random_state=0).fit_transform(X)
    trees = {}
    for name, kw in (("average-link cosine", {"method": "average", "metric": "cosine"}), ("Ward", {"method": "ward"})):
        f = TREE / f"linkage_{name.split()[0].lower()}.npy"
        if f.exists():
            trees[name] = np.load(f)
        else:
            trees[name] = linkage(Z, **kw)
            np.save(f, trees[name])
        print(name, "tree ready", flush=True)
    rows = []
    for lab in LEVELS + ["name"]:
        y_int, keep = classes_of(V[lab])
        C = len(keep)
        level = lab if lab == "name" else int(lab[2])
        results = {name: best_subtree(Zl, N, y_int, C) for name, Zl in trees.items()}
        p = KMeans(C, n_init=10, random_state=0).fit_predict(Z)
        results["k-means, k = classes"] = best_cluster(p, y_int, C)
        for method, (f1, prec, rec_, size, tot) in results.items():
            for j, c in enumerate(keep):
                rows.append({"method": method, "level": level, "code": c, "hs_name": HS_NAMES.get(c, ""), "objects": int(tot[j]), "F1": round(float(f1[j]), 3),
                             "precision": round(float(prec[j]), 3), "recall": round(float(rec_[j]), 3), "subtree_size": int(size[j])})
        print(lab, C, "classes done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "nodes.csv", index=False)
    s = []
    for (method, level), g in t.groupby(["method", "level"], sort=False):
        w = g.objects / g.objects.sum()
        s.append({"method": method, "level": level, "classes": len(g), "classes_F1_at_least_0.5": int((g.F1 >= 0.5).sum()), "classes_F1_at_least_0.7": int((g.F1 >= 0.7).sum()),
                  "objects_in_classes_F1_at_least_0.5": round(float(g.objects[g.F1 >= 0.5].sum() / g.objects.sum()), 3),
                  "mean_F1_weighted_by_objects": round(float((g.F1 * w).sum()), 3), "median_F1": round(float(g.F1.median()), 3)})
    s = pd.DataFrame(s)
    s.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
