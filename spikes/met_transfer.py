"""Transfer test on the Met: do the MIMO blind groups carry over to another collection, photographed by other people?

Met objects: data/met_instruments/objects.csv (public-domain instruments with a Hornbostel-Sachs-style classification
such as "Chordophone-Lute-plucked-fretted": level 1 = the class word, level 2 = the first 2 parts, full = the whole
string). MIMO partitions, padded DINOv2-B with PCA 50 fitted on the MIMO rows:
  hs3    k-means, k = 13, on the 27,348 same rows (as runs/clean/same_rows.csv)
  name   k-means, k = 129, same rows
  k100   k-means, k = 100, on the 31,402 blind-then-label rows (as runs/clean/blind_then_label.csv)
Met photos are embedded the same way, projected with MIMO's PCA and assigned to the nearest centroid.
Scores: AMI of the assigned group against the Met level 1, level 2 and full classification, next to k-means refit on the
Met alone with the same k; the MIMO majority label of each group carried to the Met object (level 1: the class word;
name: the MIMO majority name against the Met object name, exact and as a substring of one another); a logistic probe
trained on MIMO (level 1, name) applied to the Met photos; the Met's own probe (5-fold) as the reference.
Outputs: runs/tests/met/transfer.csv, name_transfer.csv, emb/dinov2_b_pad.npy
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "data/met_instruments"
OUT = ROOT / "runs/tests/met"
HS1 = {"1": "Idiophone", "2": "Membranophone", "3": "Chordophone", "4": "Aerophone", "5": "Electrophone"}
MIN = 30


def arr(s):
    return np.array(list(s), dtype=object)


def kept(y, min_count=MIN):
    vc = pd.Series(y).value_counts()
    return (pd.Series(y).map(vc).values >= min_count) & (y != "")


def majority(groups, labels, k):
    t = pd.DataFrame({"g": groups, "l": labels})
    t = t[t.l != ""]
    m = t.groupby("g").l.agg(lambda s: s.value_counts().index[0])
    out = np.full(k, "", dtype=object)
    out[m.index.values] = m.values
    return out


def name_match(a, b):
    a, b = a.lower().strip(), b.lower().strip()
    return a == b, bool(a) and bool(b) and (a in b or b in a)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    met = pd.read_csv(MET / "objects.csv", dtype=str).fillna("")
    met = met[met.file != ""].reset_index(drop=True)
    XM = cached_embeddings(OUT / "emb", "dinov2_b", [str(MET / f) for f in met.file], pad_square=True)
    L = {"level 1": arr(met.hs1), "level 2": arr(met.hs2), "full": arr(met.classification)}
    obj_name = arr(met.objectName)
    print(len(met), "Met objects;", {k: len(set(v)) for k, v in L.items()}, "classes", flush=True)

    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    X = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")
    pos = dict(zip(rec.id, range(len(rec))))
    same = np.array([pos[i] for i in pd.read_csv(ROOT / "runs/clean/same_rows_ids.csv", dtype=str).id])
    m2 = (rec.name != "") & (rec.hs3 != "")
    s2 = rec[m2]
    vc = s2.name.value_counts()
    btl = s2[s2.name.map(vc) >= MIN].index.values
    parts = {}
    for tag, rows, col, k_fixed, n_init in (("hs3", same, "hs3", None, 10), ("name", same, "name", None, 10), ("k100", btl, "name", 100, 3)):
        pca = PCA(50, random_state=0).fit(X[rows])
        Z = pca.transform(X[rows])
        y = arr(rec[col].iloc[rows])
        keep = kept(y) if k_fixed is None else np.ones(len(rows), dtype=bool)
        k = k_fixed or len(set(y[keep]))
        km = KMeans(k, n_init=n_init, random_state=0).fit(Z[keep])
        g_mimo = km.labels_
        parts[tag] = {"k": k, "assign": km.predict(pca.transform(XM)),
                      "maj_hs1": majority(g_mimo, arr(rec.hs1.iloc[rows][keep]).astype(object), k),
                      "maj_name": majority(g_mimo, arr(rec.name.iloc[rows][keep]), k)}
        print(tag, "k =", k, "MIMO rows", int(keep.sum()), flush=True)

    ZM = PCA(50, random_state=0).fit_transform(XM)
    rows = []
    for tag, p in parts.items():
        a = p["assign"]
        r = {"partition": f"MIMO {tag}, k = {p['k']}", "met_groups_used": int(len(set(a)))}
        for lab, y in L.items():
            r[f"AMI_{lab}_transferred"] = round(float(ami(y, a)), 3)
            r[f"AMI_{lab}_refit_on_Met"] = round(float(ami(y, KMeans(p["k"], n_init=10, random_state=0).fit_predict(ZM))), 3)
        t1 = np.array([HS1.get(h, "") for h in p["maj_hs1"][a]], dtype=object)
        r["level1_accuracy_from_MIMO_majority"] = round(float((t1 == L["level 1"]).mean()), 3)
        rows.append(r)
        print(r, flush=True)
    base = {"partition": "reference", "level1_majority_class_share": round(float(pd.Series(L["level 1"]).value_counts(normalize=True).iloc[0]), 3)}
    for lab, y in L.items():
        keep = kept(y, 10)
        if len(set(y[keep])) < 2:
            continue
        pr = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), XM[keep], y[keep], cv=StratifiedKFold(5, shuffle=True, random_state=0))
        base[f"Met_probe_acc_{lab}"] = round(float((pr == y[keep]).mean()), 3)
        base[f"Met_probe_AMI_{lab}"] = round(float(ami(y[keep], pr)), 3)
    y1 = arr(rec.hs1.iloc[same])
    clf = LogisticRegression(max_iter=2000, C=0.5).fit(X[same], y1)
    p1 = np.array([HS1[h] for h in clf.predict(XM)], dtype=object)
    base["level1_accuracy_MIMO_probe_on_Met"] = round(float((p1 == L["level 1"]).mean()), 3)
    rows.append(base)
    print(base, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "transfer.csv", index=False)

    nrows = []
    yn = arr(rec.name.iloc[same])
    keep = kept(yn)
    clf = LogisticRegression(max_iter=2000, C=0.5).fit(X[same][keep], yn[keep])
    probe_names = clf.predict(XM)
    for tag, names in (("MIMO k100 majority name", parts["k100"]["maj_name"][parts["k100"]["assign"]]),
                       ("MIMO name partition (k = 129) majority name", parts["name"]["maj_name"][parts["name"]["assign"]]),
                       ("MIMO name probe", probe_names)):
        ex, sub = zip(*[name_match(a, b) for a, b in zip(names, obj_name)])
        nrows.append({"source": tag, "exact": round(float(np.mean(ex)), 3), "substring": round(float(np.mean(sub)), 3)})
        print(nrows[-1], flush=True)
    top = pd.Series(parts["k100"]["maj_name"][parts["k100"]["assign"]]).value_counts().index[0]
    ex, sub = zip(*[name_match(top, b) for b in obj_name])
    nrows.append({"source": f"every Met object called '{top}' (the most frequent transferred name)", "exact": round(float(np.mean(ex)), 3), "substring": round(float(np.mean(sub)), 3)})
    nrows.append({"source": "Met majority object name share", "exact": round(float(pd.Series(obj_name).value_counts(normalize=True).iloc[0]), 3), "substring": None})
    pd.DataFrame(nrows).to_csv(OUT / "name_transfer.csv", index=False)
    ex_tab = pd.DataFrame({"met_object_name": obj_name, "transferred_name": parts["k100"]["maj_name"][parts["k100"]["assign"]], "met_class": L["full"]})
    ex_tab.groupby(["met_object_name", "transferred_name"]).size().sort_values(ascending=False).head(60).to_csv(OUT / "name_pairs_top60.csv")


if __name__ == "__main__":
    main()
