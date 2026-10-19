"""Instruments at Met scale through the Met's published DINOv2-giant embeddings (CC0, no images).

All public-domain objects of the Musical Instruments department with an H-S classification. Scores clusters
against H-S level 1, 2 and 3, region and era; then per-level tree agreement: for each H-S level, the AMI of
an agglomerative cut with as many clusters as that level has classes, plus the purity of a Ward dendrogram cut.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import score  # noqa: E402
from labels import run_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/instruments_met_large"


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    cols = {"Object ID": "objectID", "Department": "department", "Object Name": "objectName", "Culture": "culture", "Country": "country",
            "Region": "region", "Classification": "classification", "Object Begin Date": "year", "Is Public Domain": "isPublicDomain"}
    meta = pd.read_csv(ROOT / "dl/MetObjects.csv", usecols=list(cols), dtype=str, low_memory=False).rename(columns=cols)
    meta["objectID"] = meta.objectID.astype(int)
    meta = meta[(meta.department == "Musical Instruments") & meta.classification.notna()].copy()
    emb = load_dataset("metmuseum/openaccess-embeddings-dinov2-giant", split="train")
    ids = np.array(emb["objectID"])
    pos = {int(o): i for i, o in enumerate(ids.tolist())}
    meta = meta[meta.objectID.isin(pos)].reset_index(drop=True)
    parts = meta.classification.str.split("-")
    meta["hs1"] = parts.str[0].str.strip()
    meta["hs2"] = parts.str[:2].str.join("-").str.strip()
    meta["hs3"] = parts.str[:3].str.join("-").str.strip()
    meta = meta[meta.hs1.isin(["Aerophone", "Chordophone", "Idiophone", "Membranophone"])].reset_index(drop=True)
    meta["culture1"] = meta.culture.fillna("").str.split(",|;|\\(").str[0].str.strip()
    meta["region1"] = meta.country.fillna("").where(meta.country.fillna("") != "", meta.culture1)
    meta["era"] = pd.cut(pd.to_numeric(meta.year, errors="coerce"), [-5000, 1600, 1800, 1900, 2100], labels=["<1600", "1600-1799", "1800-1899", "1900+"]).astype(str)
    X = np.array(emb.select([pos[o] for o in meta.objectID])["embedding"], dtype=np.float32)
    print(len(meta), "instruments with embeddings")
    print("hs1:", meta.hs1.value_counts().to_dict())
    print("hs2 classes:", meta.hs2.nunique(), "| hs3 classes:", meta.hs3.nunique(), "| regions with 20+:", (meta.region1.value_counts() >= 20).sum())
    print("era:", meta.era.value_counts().to_dict())
    meta.to_csv(RUNS / "sample.csv", index=False); np.save(RUNS / "X.npy", X)

    r = run_labels(X, meta, ["hs1", "hs2", "hs3", "region1", "era"], ks=(4, 10, 20, 40, 80, 150), encoder_name="dinov2_giant_met", min_count=20)
    print("\n== overall"); print(r.drop(columns=["encoder"]).to_string(index=False))

    print("\n== tree agreement: one Ward dendrogram, cut at the number of classes of each H-S level")
    Z = PCA(50, random_state=0).fit_transform(X)
    link = linkage(Z, method="ward")
    for lvl in ("hs1", "hs2", "hs3"):
        vc = meta[lvl].value_counts(); keep = meta[lvl].isin(vc[vc >= 20].index).values
        y = np.array(meta[lvl][keep].tolist(), dtype=object); k = len(set(y))
        lab = fcluster(link, k, criterion="maxclust")[keep]
        purity = sum(pd.Series(y[lab == c]).value_counts().iloc[0] for c in np.unique(lab)) / len(y)
        print(f"   {lvl}: {k} classes with 20+, cut AMI {score(y, lab)['AMI']:.3f}, purity {purity:.3f}, majority {vc.max() / keep.sum():.3f}")

    print("\n== region inside each hs1 class (regions with 20+)")
    for cls, g in meta.groupby("hs1"):
        vc = g.region1.value_counts(); gg = g[g.region1.isin(vc[vc >= 20].index) & (g.region1 != "")]
        if gg.region1.nunique() < 3:
            continue
        rr = run_labels(X[gg.index.values], gg.reset_index(drop=True), ["region1", "era", "hs2"], ks=(3, 5, 8, 15), encoder_name="", min_count=20)
        print(f"   {cls} (n={len(gg)}): " + "; ".join(f"{a.label} AMI={a.best_AMI} probe={a.probe_acc} majority={a.majority} ({a.classes} cl)" for a in rr.itertuples() if a.classes >= 2))


if __name__ == "__main__":
    main()
