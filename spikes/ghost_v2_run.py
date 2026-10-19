"""Ghost Borders, attempt 2, scoring.

For each border and set (real, placebo east, placebo west): AMI of 2-cluster and best-k clusterings against
side, a balanced linear probe (5-fold, 10 resamples), and the AUC of each explicit field feature. Image
variants: RGB early summer, RGB late summer, NDVI grayscale late summer, and the concatenation of both RGB
embeddings. The 2 positive controls (cz_at, inner_de) must clear their placebos before pl_partition means anything.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["clip_b32", "dinov2_s"]
DATA = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "data/ghost_v2")
RUNS = ROOT / "runs" / DATA.name
VARIANTS = {"rgb_d1": "{}_d1_rgb.png", "rgb_d2": "{}_d2_rgb.png", "ndvi_d2": "{}_d2_ndvi.png"}
FEATS = ["edge_density", "mean_seg_ha", "median_seg_ha", "n_segments", "ndvi_std"]


def cluster_scores(X, y, seed=0):
    Z = PCA(min(50, X.shape[0] - 1, X.shape[1]), random_state=seed).fit_transform(X)
    k2 = KMeans(2, n_init=10, random_state=seed).fit_predict(Z)
    best = max(score(y, AgglomerativeClustering(k, linkage="ward").fit_predict(Z))["AMI"] for k in (2, 3, 4, 6, 8))
    return score(y, k2)["AMI"], best


def probe(X, y, reps=10):
    accs = []
    for r in range(reps):
        pred = cross_val_predict(LogisticRegression(max_iter=3000, C=0.5), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=r))
        accs.append(balanced_accuracy_score(y, pred))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    df = pd.read_csv(DATA / "tiles.csv")
    print(df.groupby(["border", "set", "side"]).size().unstack(fill_value=0).to_string())
    RUNS.mkdir(parents=True, exist_ok=True)
    embs = {}
    for enc in ENCODERS:
        for var, pat in VARIANTS.items():
            paths = [str(DATA / "tiles" / pat.format(n)) for n in df.name]
            embs[(enc, var)] = cached_embeddings(RUNS / f"emb_{var}", enc, paths)
        embs[(enc, "rgb_d1+d2")] = np.concatenate([embs[(enc, "rgb_d1")], embs[(enc, "rgb_d2")]], axis=1)
    rows = []
    for (border, sname), g in df.groupby(["border", "set"]):
        y = (g.side == "A").astype(int).values
        idx = g.index.values
        if min((y == 1).sum(), (y == 0).sum()) < 20:
            continue  # set still being fetched
        row = {"border": border, "set": sname, "n": len(g)}
        for f in FEATS:
            for dk in ("d1", "d2"):
                v = g[f"{dk}_{f}"].values
                auc = roc_auc_score(y, v)
                row[f"auc_{dk}_{f}"] = round(max(auc, 1 - auc), 3)
        for (enc, var), X in embs.items():
            ami2, amib = cluster_scores(X[idx], y)
            pm, ps = probe(X[idx], y)
            row[f"{enc}/{var} AMI k2"] = round(ami2, 3)
            row[f"{enc}/{var} AMI best"] = round(amib, 3)
            row[f"{enc}/{var} probe"] = f"{pm:.3f}±{ps:.3f}"
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(RUNS / "scores.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n== explicit field features, AUC side A vs B (0.5 = nothing). Real must beat both placebos.")
    cols = ["border", "set", "n"] + [c for c in res.columns if c.startswith("auc_d2_")]
    print(res[cols].to_string(index=False))
    for enc in ENCODERS:
        print(f"\n== {enc}: embeddings, AMI (k=2 / best k) and balanced probe accuracy")
        cols = ["border", "set"] + [c for c in res.columns if c.startswith(enc + "/")]
        print(res[cols].to_string(index=False))


if __name__ == "__main__":
    main()
