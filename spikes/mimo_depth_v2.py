"""Full-set H-S depth table, recomputed with the hyphen suffix of the codes dropped (common.hs_level) and the majority
taken over the kept rows. Same embeddings and rows as runs/mimo_full/depth.csv. Output: runs/mimo_full_v2/depth.csv.
Ward linkage on 41k points is heavy; pass 'kmeans' as the first argument to skip the dendrogram column."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hs_level, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/mimo_full_v2"
WITH_DENDRO = not (len(sys.argv) > 1 and sys.argv[1] == "kmeans")


def purity(y, lab):
    return sum(pd.Series(y[lab == c]).value_counts().iloc[0] for c in np.unique(lab)) / len(y)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "runs/mimo_full/sample_full.csv", dtype={"notation": str}).fillna("")
    X = np.load(ROOT / "runs/mimo_full/emb/dinov2_b.npy")
    assert len(X) == len(df)
    coded = hs_level(df.notation, 1) != ""
    Xc, dfc = X[coded], df[coded].reset_index(drop=True)
    Z = PCA(50, random_state=0).fit_transform(Xc)
    link = linkage(Z, method="ward") if WITH_DENDRO else None
    print(f"{len(dfc)} coded objects; {'with' if WITH_DENDRO else 'without'} dendrogram", flush=True)
    rows = []
    for d in range(1, 7):
        lab = pd.Series(hs_level(dfc.notation, d))
        vc = lab[lab != ""].value_counts(); keep = lab.isin(vc[vc >= 30].index).values
        y = np.array(lab[keep].tolist(), dtype=object); k = len(set(y))
        if k < 2:
            break
        km = MiniBatchKMeans(k, n_init=5, random_state=0, batch_size=4096).fit_predict(Z[keep])
        dc = fcluster(link, k, criterion="maxclust")[keep] if WITH_DENDRO else None
        pr = cross_val_score(LogisticRegression(max_iter=2000, C=0.5), Xc[keep], y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        row = {"level": d, "classes": k, "objects": int(keep.sum()), "AMI_kmeans": round(score(y, km)["AMI"], 3),
               "AMI_dendro": round(score(y, dc)["AMI"], 3) if WITH_DENDRO else None, "purity": round(purity(y, km), 3),
               "majority": round(float(pd.Series(y).value_counts(normalize=True).iloc[0]), 3), "probe": round(float(pr), 3)}
        rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "depth.csv", index=False)


if __name__ == "__main__":
    main()
