"""Robustness of the main numbers on the 27,348 same rows (runs/clean/same_rows_ids.csv; classes with 30+ objects).

Blocks (pass any as arguments; default all):
  subsample  20 subsamples of 80 percent, PCA and k-means refit on each: spread of the AMI per label
  evalboot   the full-set k-means partition held fixed, objects resampled with replacement 200 times: spread of the score
  probe      the probe with 3 different fold splits for hs3, family and name
  museum     leave one museum out (museums with 900+ objects): k-means AMI without that museum; and the probe trained
             on the other museums and tested on the held-out museum, next to the pooled cross-validated probe on the
             same objects
  balance    each museum capped at 500 objects, 10 draws, against random draws of the same size
Outputs: runs/tests/robust/*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/tests/robust"
LABELS = ("hs1", "hs2", "hs3", "hs4", "hs5", "hs6", "family", "name")
MIN = 30


def kept(y, min_count=MIN):
    y = np.array(pd.Series(y).astype(str).tolist(), dtype=object)
    vc = pd.Series(y).value_counts()
    return (pd.Series(y).map(vc).values >= min_count) & (y != ""), y


def kmeans_ami(Z, df, labels=LABELS, seed=0, min_count=MIN):
    out = {}
    for c in labels:
        keep, y = kept(df[c].values, min_count)
        k = len(set(y[keep]))
        out[c] = float(ami(y[keep], KMeans(k, n_init=10, random_state=seed).fit_predict(Z[keep])))
    return out


def probe_pred(X, y, seed=0):
    return cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))


def summary(vals):
    v = np.array(vals)
    return {"mean": round(float(v.mean()), 3), "p2.5": round(float(np.percentile(v, 2.5)), 3), "p97.5": round(float(np.percentile(v, 97.5)), 3),
            "min": round(float(v.min()), 3), "max": round(float(v.max()), 3)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = sys.argv[1:] or ["subsample", "evalboot", "probe", "museum", "balance"]
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    ids = pd.read_csv(ROOT / "runs/clean/same_rows_ids.csv", dtype=str).id
    pos = dict(zip(rec.id, range(len(rec))))
    idx = np.array([pos[i] for i in ids])
    df = rec.iloc[idx].reset_index(drop=True)
    X = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")[idx]
    Z = PCA(50, random_state=0).fit_transform(X)
    full = kmeans_ami(Z, df)
    print(len(df), "rows; full-set k-means AMI", {c: round(v, 3) for c, v in full.items()}, flush=True)

    if "subsample" in blocks:
        draws = []
        for s in range(20):
            sel = np.sort(np.random.default_rng(s).choice(len(df), int(0.8 * len(df)), replace=False))
            Zs = PCA(50, random_state=s).fit_transform(X[sel])
            draws.append(kmeans_ami(Zs, df.iloc[sel].reset_index(drop=True), seed=s))
            print("subsample", s, {c: round(v, 3) for c, v in draws[-1].items()}, flush=True)
        rows = [{"label": c, "full_set": round(full[c], 3), **summary([d[c] for d in draws])} for c in LABELS]
        pd.DataFrame(rows).to_csv(OUT / "subsample.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)

    if "evalboot" in blocks:
        rows = []
        rng = np.random.default_rng(0)
        for c in LABELS:
            keep, y = kept(df[c].values)
            yk = y[keep]
            p = KMeans(len(set(yk)), n_init=10, random_state=0).fit_predict(Z[keep])
            vals = []
            for _ in range(200):
                b = rng.integers(0, len(yk), len(yk))
                vals.append(ami(yk[b], p[b]))
            rows.append({"label": c, "full_set": round(full[c], 3), **summary(vals)})
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "evalboot.csv", index=False)

    if "probe" in blocks:
        rows = []
        for c in ("hs3", "family", "name"):
            keep, y = kept(df[c].values)
            for seed in (0, 1, 2):
                p = probe_pred(X[keep], y[keep], seed)
                rows.append({"label": c, "fold_seed": seed, "accuracy": round(float((p == y[keep]).mean()), 3), "AMI": round(float(ami(y[keep], p)), 3)})
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "probe_folds.csv", index=False)

    if "museum" in blocks:
        big = [m for m, n in df.provider.value_counts().items() if n >= 900]
        pooled = {}
        for c in ("hs1", "hs3", "family", "name"):
            keep, y = kept(df[c].values)
            p = np.full(len(df), "", dtype=object)
            p[keep] = probe_pred(X[keep], y[keep])
            pooled[c] = (keep, y, p)
        rows = []
        for m in big:
            out = (df.provider != m).values
            r = {"museum_left_out": m, "objects_left_out": int((~out).sum())}
            Zs = PCA(50, random_state=0).fit_transform(X[out])
            r.update({f"kmeans_AMI_without_{c}": round(v, 3) for c, v in kmeans_ami(Zs, df[out].reset_index(drop=True), ("hs1", "hs3", "family", "name")).items()})
            for c in ("hs1", "hs3", "family"):
                keep, y, p_pooled = pooled[c]
                tr = keep & out
                te = keep & ~out
                clf = LogisticRegression(max_iter=2000, C=0.5).fit(X[tr], y[tr])
                pt = clf.predict(X[te])
                r[f"probe_acc_{c}_trained_without_museum"] = round(float((pt == y[te]).mean()), 3)
                r[f"probe_acc_{c}_pooled_folds"] = round(float((p_pooled[te] == y[te]).mean()), 3)
            rows.append(r)
            print(r, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "leave_museum_out.csv", index=False)

    if "balance" in blocks:
        rows = []
        for s in range(10):
            rng = np.random.default_rng(1000 + s)
            capped = np.sort(np.concatenate([rng.choice(g, min(len(g), 500), replace=False) for g in df.groupby("provider").indices.values()]))
            rand = np.sort(rng.choice(len(df), len(capped), replace=False))
            for tag, sel in (("capped at 500 per museum", capped), ("random, same size", rand)):
                d = df.iloc[sel].reset_index(drop=True)
                Zs = PCA(50, random_state=s).fit_transform(X[sel])
                r = {"draw": s, "sample": tag, "objects": len(sel), "museums": d.provider.nunique()}
                r.update({c: round(v, 3) for c, v in kmeans_ami(Zs, d, ("hs1", "hs3", "family", "name", "provider"), seed=s, min_count=10).items()})
                rows.append(r)
                print(r, flush=True)
        t = pd.DataFrame(rows)
        t.to_csv(OUT / "balance_draws.csv", index=False)
        print(t.groupby("sample")[["hs1", "hs3", "family", "name", "provider"]].agg(["mean", "min", "max"]).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
