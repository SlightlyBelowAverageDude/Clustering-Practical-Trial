"""Score image clusters against several candidate labels at once."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cluster_and_score  # noqa: E402


def run_labels(X: np.ndarray, df: pd.DataFrame, label_cols, ks=(5, 10, 20), encoder_name: str = "",
               min_count: int = 5) -> pd.DataFrame:
    """For each label column: best AMI over k-means and agglomerative clusterings and the given ks, the AMI
    at k = number of classes (if that k is in ks), a cross-validated linear-probe accuracy (ceiling for what
    is linearly present in the embedding), and the majority-class baseline. Rows whose label value occurs
    fewer than min_count times are dropped for that label."""
    rows = []
    for col in label_cols:
        y = np.array(df[col].astype(str).tolist(), dtype=object)  # plain numpy, not a pyarrow-backed array
        counts = pd.Series(y).value_counts()
        keep = (pd.Series(y).map(counts) >= min_count).values & (y != "") & (y != "nan")
        Xc, yc = X[keep], y[keep]
        n_cls = len(set(yc))
        if n_cls < 2:
            rows.append({"encoder": encoder_name, "label": col, "n": len(yc), "classes": n_cls})
            continue
        res = cluster_and_score(Xc, yc, ks=ks, methods=("kmeans", "agglo"), hdbscan_min_cluster_size=None)
        best = res.sort_values("AMI", ascending=False).iloc[0]
        at_k = res[(res.k == n_cls) & (res.method == "kmeans")]
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        probe = cross_val_score(LogisticRegression(max_iter=3000, C=0.5), Xc, yc, cv=cv).mean()
        rows.append({"encoder": encoder_name, "label": col, "n": len(yc), "classes": n_cls,
                     "best_AMI": round(float(best.AMI), 3), "best_method": f"{best.method} k={int(best.k)}",
                     "AMI_at_k_classes": round(float(at_k.AMI.iloc[0]), 3) if len(at_k) else None,
                     "probe_acc": round(float(probe), 3), "majority": round(float(pd.Series(yc).value_counts(normalize=True).iloc[0]), 3)})
    return pd.DataFrame(rows)
