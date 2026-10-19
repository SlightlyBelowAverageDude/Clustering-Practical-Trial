"""Museum objects spike, step 2: do image clusters follow function or department (culture)?"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings, cluster_and_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RUNS = ROOT / "data/museum", ROOT / "runs/museum"
ENCODERS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["clip_b32", "dinov2_s"]


def main():
    df = pd.read_csv(DATA / "objects.csv")
    paths = [str(DATA / f) for f in df.file]
    out = []
    for enc in ENCODERS:
        X = cached_embeddings(RUNS / "emb", enc, paths)
        for label in ("function", "department"):
            res = cluster_and_score(X, df[label].values, ks=(5, 7, 10, 15, 25), hdbscan_min_cluster_size=8)
            res.insert(0, "label", label)
            res.insert(0, "encoder", enc)
            out.append(res)
            print(f"\n== {enc} / clusters vs {label} ({df[label].nunique()} classes, n={len(df)})")
            print(res.drop(columns=["encoder", "label"]).round(3).to_string(index=False))
    out = pd.concat(out)
    RUNS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RUNS / "scores.csv", index=False)


if __name__ == "__main__":
    main()
