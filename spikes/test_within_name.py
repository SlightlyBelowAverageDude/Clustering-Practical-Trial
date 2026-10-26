"""Within one instrument name: what do the blind sub-groups follow, the maker, the period or the museum?

Names: violin, clarinet, Western concert flute, guitar (the 4 largest). Objects: clean records with that name.
Labels: maker (dcCreator from the harvest and the record API; anonymous and unknown variants dropped; makers with 5+
objects of that name), decade (Europeana's year, or the first 4-digit year in the record's free-text date), museum
(museums with 10+ objects of that name). Sub-groups: k-means on PCA 50 of the padded DINOv2-B embeddings of the name's
objects at k = 5, 10, 20 and k = number of classes, scored with AMI; the probe and the majority baseline; the AMI between
the labels themselves. Outputs: runs/tests/within_name/scores.csv, label_agreement.csv, coverage.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/tests/within_name"
NAMES = ("violin", "clarinet", "Western concert flute", "guitar")
ANON = re.compile(r"^(anonym|anonim|anonyme|unknown|okänd|ez dakigu|nieznany|inconnu|sconosciuto|unbekannt|anoniem|not known|\?|-)", re.I)


def clean_maker(s):
    """First non-anonymous maker string; MIMO maker ids are kept as ids."""
    for v in str(s).split(" | "):
        v = v.strip()
        if not v or ANON.match(v):
            continue
        return v
    return ""


def decade_of(year, date_text):
    m = re.search(r"\b(1[2-9]\d\d|20[0-2]\d)\b", str(year)) or re.search(r"\b(1[2-9]\d\d|20[0-2]\d)\b", str(date_text))
    return f"{int(m.group(1)) // 10 * 10}s" if m else ""


def kept(y, min_count):
    y = np.array(list(y), dtype=object)
    vc = pd.Series(y).value_counts()
    return (pd.Series(y).map(vc).values >= min_count) & (y != ""), y


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    X = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")
    harvest = pd.read_csv(ROOT / "data/mimo/records.csv", dtype=str, usecols=["id", "creator", "year"]).fillna("")
    extra = pd.read_csv(ROOT / "data/mimo/record_extra.csv", dtype=str).fillna("") if (ROOT / "data/mimo/record_extra.csv").exists() else pd.DataFrame(columns=["id", "creators", "date_text"])
    rows, agree, cover = [], [], []
    for name in NAMES:
        sel = np.where((rec.name == name).values)[0]
        d = rec.iloc[sel][["id", "provider", "year"]].merge(harvest, on="id", how="left", suffixes=("", "_h")).merge(extra[["id", "creators", "date_text"]], on="id", how="left").fillna("")
        maker = np.array([clean_maker(c2 if c2 else c1) for c1, c2 in zip(d.creator, d.creators)], dtype=object)
        decade = np.array([decade_of(y, t) for y, t in zip(d.year, d.date_text)], dtype=object)
        museum = np.array(d.provider.tolist(), dtype=object)
        Z = PCA(50, random_state=0).fit_transform(X[sel])
        labels = {"maker": (maker, 5), "decade": (decade, 10), "museum": (museum, 10)}
        cover.append({"name": name, "objects": len(sel), "with_maker": int((maker != "").sum()), "distinct_makers": len(set(maker) - {""}),
                      "with_decade": int((decade != "").sum()), "museums": len(set(museum))})
        print(cover[-1], flush=True)
        kept_ = {}
        for lab, (y, mc) in labels.items():
            keep, y = kept(y, mc)
            kept_[lab] = (keep, y)
            yk = y[keep]
            k = len(set(yk))
            if k < 2:
                continue
            r = {"name": name, "label": lab, "n": int(keep.sum()), "classes": k, "majority": round(float(pd.Series(yk).value_counts(normalize=True).iloc[0]), 3)}
            for kk in (5, 10, 20, k):
                r[f"AMI_k{kk}" if kk in (5, 10, 20) else "AMI_k_classes"] = round(float(ami(yk, KMeans(kk, n_init=10, random_state=0).fit_predict(Z[keep]))), 3)
            if (pd.Series(yk).value_counts() >= 5).all():
                pr = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), X[sel][keep], yk, cv=StratifiedKFold(5, shuffle=True, random_state=0))
                r["probe_acc"], r["probe_AMI"] = round(float((pr == yk).mean()), 3), round(float(ami(yk, pr)), 3)
            rows.append(r)
            print(r, flush=True)
        for a, b in (("maker", "museum"), ("maker", "decade"), ("decade", "museum")):
            (ka, ya), (kb, yb) = kept_[a], kept_[b]
            both = ka & kb
            if both.sum() >= 30:
                agree.append({"name": name, "a": a, "b": b, "n": int(both.sum()), "AMI": round(float(ami(ya[both], yb[both])), 3)})
                print(agree[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(agree).to_csv(OUT / "label_agreement.csv", index=False)
    pd.DataFrame(cover).to_csv(OUT / "coverage.csv", index=False)


if __name__ == "__main__":
    main()
