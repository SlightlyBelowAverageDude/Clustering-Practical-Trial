"""Ghost Borders spike, step 2: embed the tiles, cluster, score against side-of-border.

For each tile set (real border, 2 placebo lines) and each encoder: AMI/NMI/ARI of k-means, agglomerative,
GMM and HDBSCAN clusters against the side label. A cross-validated logistic regression is printed as a
diagnostic: it says whether the side is linearly present in the embedding at all, which is the ceiling
for what clustering could recover.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings, cluster_and_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ghost_borders"
RUNS = ROOT / "runs/ghost_borders"
ENCODERS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["clip_b32", "dinov2_s"]


def main():
    df = pd.read_csv(DATA / "tiles.csv")
    paths = [str(DATA / f) for f in df.file]
    results = []
    for enc in ENCODERS:
        X_all = cached_embeddings(RUNS / "emb", enc, paths)
        for set_name, sub in df.groupby("set"):
            X, y = X_all[sub.index.values], (sub.side == "east").astype(int).values
            res = cluster_and_score(X, y, ks=(2, 3, 4, 6, 8), hdbscan_min_cluster_size=10)
            res.insert(0, "set", set_name)
            res.insert(0, "encoder", enc)
            probe = cross_val_score(LogisticRegression(max_iter=2000, C=0.5), X, y, cv=5).mean()
            res["probe_acc"] = probe
            results.append(res)
            print(f"\n== {enc} / {set_name}: n={len(sub)}, linear probe acc={probe:.3f}")
            print(res.drop(columns=["encoder", "set", "probe_acc"]).round(3).to_string(index=False))
    out = pd.concat(results)
    RUNS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RUNS / "scores.csv", index=False)
    print("\n== best AMI per encoder and set")
    print(out.sort_values("AMI", ascending=False).groupby(["encoder", "set"]).head(1)
          [["encoder", "set", "method", "k", "AMI", "ARI", "probe_acc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
