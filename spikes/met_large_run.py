"""Museum objects at scale: the Met's own DINOv2-giant embeddings (CC0) joined to MetObjects metadata.

No images are downloaded. For objects whose objectName matches a functional vocabulary and whose culture is
set, scores clusters against function, culture (first token), department and period, overall and inside
each function. Also the residual test: subtract function class means, rescore culture.

Usage: python spikes/met_large_run.py [max_per_function]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import score  # noqa: E402
from labels import run_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/met_large"
MAX_PER_FN = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
FUNCTIONS = ["bowl", "jar", "cup", "comb", "lamp", "spoon", "knife", "bottle", "mirror", "bead", "vase", "plate", "dish",
             "pitcher", "bell", "ring", "pendant", "amulet", "seal", "coin", "axe", "spearhead", "arrowhead", "needle", "pin",
             "box", "basket", "mask", "helmet", "sword", "dagger", "shield", "flute", "drum", "whistle"]


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    # MetObjects.csv from github.com/metmuseum/openaccess (the Hugging Face metadata set carries image bytes, 52 GB)
    cols = {"Object ID": "objectID", "Department": "department", "Object Name": "objectName", "Culture": "culture",
            "Period": "period", "Classification": "classification", "Object Begin Date": "objectBeginDate", "Is Public Domain": "isPublicDomain"}
    meta = pd.read_csv(ROOT / "dl/MetObjects.csv", usecols=list(cols), dtype=str, low_memory=False).rename(columns=cols)
    meta["objectID"] = meta.objectID.astype(int)
    print(len(meta), "objects in metadata")
    emb = load_dataset("metmuseum/openaccess-embeddings-dinov2-giant", split="train")
    ids = np.array(emb["objectID"])
    print(len(ids), "embeddings")
    meta = meta[meta.objectID.isin(set(ids.tolist()))].copy()
    name = meta.objectName.fillna("").str.lower()
    meta["function"] = ""
    for fn in FUNCTIONS:  # first matching whole word wins
        m = (meta.function == "") & name.str.contains(rf"\b{fn}\b", regex=True)
        meta.loc[m, "function"] = fn
    meta["culture1"] = meta.culture.fillna("").str.split(",|;|\\(|or ").str[0].str.strip()
    meta["culture1"] = meta.culture1.str.replace(r"^(probably|possibly|attributed to)\s+", "", regex=True)
    # the Met mixes noun and adjective forms; merge the frequent pairs
    merge = {"Japanese": "Japan", "Chinese": "China", "Korean": "Korea", "Iranian": "Iran", "Persian": "Iran", "Indian": "India",
             "Egyptian": "Egypt", "Greek": "Greece", "Roman": "Rome", "Etruscan": "Etruria", "Cypriot": "Cyprus", "Byzantine": "Byzantium",
             "British": "Britain", "English": "Britain", "French": "France", "German": "Germany", "Italian": "Italy", "Spanish": "Spain",
             "Dutch": "Netherlands", "Flemish": "Flanders", "American": "United States", "Mexican": "Mexico", "Peruvian": "Peru",
             "Turkish": "Turkey", "Syrian": "Syria", "Iraqi": "Iraq", "Tibetan": "Tibet", "Thai": "Thailand", "Nepalese": "Nepal",
             "Russian": "Russia", "Swiss": "Switzerland", "Austrian": "Austria", "Swedish": "Sweden", "Danish": "Denmark"}
    meta["culture1"] = meta.culture1.replace(merge)
    sel = meta[(meta.function != "") & (meta.culture1 != "")]
    sel = pd.concat([g.sample(min(len(g), MAX_PER_FN), random_state=0) for _, g in sel.groupby("function")]).reset_index(drop=True)
    print(len(sel), "objects with function and culture;", sel.function.nunique(), "functions;", sel.culture1.nunique(), "cultures")
    print(sel.function.value_counts().to_dict())
    print(sel.culture1.value_counts().head(15).to_dict())
    print(sel.department.value_counts().head(10).to_dict())
    pos = {oid: i for i, oid in enumerate(ids.tolist())}
    rows = [pos[o] for o in sel.objectID]
    X = np.array(emb.select(rows)["embedding"], dtype=np.float32)
    sel["era"] = pd.cut(pd.to_numeric(sel.objectBeginDate, errors="coerce"), [-5000, 0, 1000, 1500, 1800, 2100],
                        labels=["BCE", "1-1000", "1000-1499", "1500-1799", "1800+"]).astype(str)
    sel.to_csv(RUNS / "sample.csv", index=False)
    np.save(RUNS / "X_dinov2_giant.npy", X)

    r = run_labels(X, sel, ["function", "culture1", "department", "era"], ks=(10, 20, 35, 50, 100), encoder_name="dinov2_giant_met", min_count=30)
    print("\n== overall"); print(r.drop(columns=["encoder"]).to_string(index=False))

    print("\n== culture inside each function (cultures with 30+ objects)")
    out = []
    for fn, g in sel.groupby("function"):
        vc = g.culture1.value_counts()
        keep = g.culture1.isin(vc[vc >= 30].index)
        gg = g[keep]
        if gg.culture1.nunique() < 3:
            continue
        rr = run_labels(X[gg.index.values], gg.reset_index(drop=True), ["culture1"], ks=(gg.culture1.nunique(),), encoder_name="", min_count=30)
        out.append({"function": fn, "n": len(gg), "cultures": gg.culture1.nunique(), "AMI": rr.best_AMI.iloc[0], "probe": rr.probe_acc.iloc[0], "majority": rr.majority.iloc[0]})
    o = pd.DataFrame(out).sort_values("AMI", ascending=False)
    print(o.to_string(index=False))
    o.to_csv(RUNS / "within_function.csv", index=False)

    print("\n== residual test: remove function class means, rescore culture and department (k = number of classes kept)")
    Z = X.copy()
    for fn in sel.function.unique():
        m = (sel.function == fn).values
        Z[m] -= X[m].mean(0)
    for col in ("culture1", "department"):
        vc = sel[col].value_counts()
        keep = sel[col].isin(vc[vc >= 30].index).values
        y = np.array(sel[col][keep].tolist(), dtype=object)
        k = len(set(y))
        for nm, M in (("raw", X[keep]), ("residual", Z[keep])):
            P = PCA(50, random_state=0).fit_transform(M)
            ami = score(y, KMeans(k, n_init=5, random_state=0).fit_predict(P))["AMI"]
            print(f"   {col:10s} {nm:8s} k={k}: AMI {ami:.3f}")


if __name__ == "__main__":
    main()
