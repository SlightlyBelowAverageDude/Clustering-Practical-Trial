"""MIMO step 3: score image clusters against H-S levels 1 to 3, country of manufacture, museum and era.

Same scoring as the Met runs, so the 2 collections can be compared directly. Also the within-class country
test and a museum (photography) control: if museum is as recoverable as country, the 2 are confounded.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings, score  # noqa: E402
from labels import run_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RUNS = ROOT / "data/mimo", ROOT / "runs/mimo"
ENCODERS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["clip_b32", "dinov2_s", "dinov2_b"]
HS1 = {"1": "Idiophone", "2": "Membranophone", "3": "Chordophone", "4": "Aerophone", "5": "Electrophone"}


def main():
    df = pd.read_csv(DATA / "sample.csv", dtype={"notation": str, "hs1": str, "hs2": str, "hs3": str}).fillna("")
    df["hs1"] = df.hs1.map(HS1).fillna(df.hs1)
    df["era"] = pd.cut(pd.to_numeric(df.year_num, errors="coerce"), [0, 1700, 1800, 1850, 1900, 2100],
                       labels=["<1700", "1700-1799", "1800-1849", "1850-1899", "1900+"]).astype(str)
    print(len(df), "images;", df.hs1.value_counts().to_dict())
    print("countries with 30+:", (df.country.value_counts() >= 30).sum(), "| museums:", df.provider.nunique(), "| with era:", (df.era != "nan").sum())
    paths = [str(DATA / f) for f in df.file]
    RUNS.mkdir(parents=True, exist_ok=True)
    out = []
    for enc in ENCODERS:
        X = cached_embeddings(RUNS / "emb", enc, paths)
        r = run_labels(X, df, ["hs1", "hs2", "hs3", "country", "region", "provider", "era"], ks=(5, 10, 20, 40, 80, 150), encoder_name=enc, min_count=30)
        out.append(r)
        print(f"\n== {enc}, n={len(df)}")
        print(r.drop(columns=["encoder"]).to_string(index=False))
        Z = PCA(50, random_state=0).fit_transform(X)
        link = linkage(Z, method="ward")
        for lvl in ("hs1", "hs2", "hs3"):
            vc = df[lvl].value_counts(); keep = df[lvl].isin(vc[vc >= 30].index).values
            y = np.array(df[lvl][keep].tolist(), dtype=object); k = len(set(y))
            lab = fcluster(link, k, criterion="maxclust")[keep]
            print(f"   dendrogram cut at {lvl} ({k} classes): AMI {score(y, lab)['AMI']:.3f}")
        for cls, g in df.groupby("hs1"):
            vc = g.country.value_counts(); gg = g[(g.country.isin(vc[vc >= 30].index) & (g.country != "")) | (g.region != "")]
            if gg.country.nunique() < 3:
                continue
            rr = run_labels(X[gg.index.values], gg.reset_index(drop=True), ["country", "region", "provider", "hs2"], ks=(3, 5, 8, 15), encoder_name=enc, min_count=30)
            print(f"   within {cls} (n={len(gg)}): " + "; ".join(f"{a.label} AMI={a.best_AMI} probe={a.probe_acc} majority={a.majority} ({a.classes} cl)"
                                                             for a in rr.itertuples() if a.classes >= 2))
    pd.concat(out).to_csv(RUNS / "scores.csv", index=False)


if __name__ == "__main__":
    main()
