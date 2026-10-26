"""Can the blind groups be named without labels?

Rows and blind groups as in the blind-then-label table (runs/clean/blind_then_label.csv): clean records with a name and
a level-3 code, names with 30+ objects, k-means on PCA 50 of padded DINOv2-B, k = 20, 50, 100, 200, 400.
Naming without labels: SigLIP B/16 image embeddings (padded, runs/tests/naming/emb) and text embeddings of
"a photo of a <name>, a musical instrument".
  group name   the name closest to the mean of the group's image embeddings
  vote name    the most frequent per-object nearest name in the group; a group is nameable without labels when that
               name covers at least half of the group
2 vocabularies: all 945 instrument names of the Wikidata items linked to MIMO (label-free), and the 157 names that occur
as labels on these rows (label-informed: it tests the naming step with the vocabulary problem removed).
Scored against the labels: group name equals the majority label name; objects whose own name equals their group's name;
objects whose level-3 code equals the Wikidata H-S code of their group's name (where the name has one).
Outputs: runs/tests/naming/naming.csv, groups_k100_<vocabulary>.csv, per_object_zero_shot.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import hs_level  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/tests/naming"
MIN = 30
PROMPT = "a photo of a {}, a musical instrument"


def arr(s):
    return np.array(list(s), dtype=object)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    X = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")
    paths = [str(ROOT / "data/mimo" / f) for f in rec.file]
    common.ENCODERS["siglip_b16"] = ("open_clip", "ViT-B-16-SigLIP", "webli")
    XS = common.cached_embeddings(OUT / "emb", "siglip_b16", paths, pad_square=True)
    assert len(XS) == len(rec) == len(X)

    wd = pd.read_csv(ROOT / "dl/wd_mimo_instruments.csv", dtype=str).fillna("")
    wd["label"] = wd.itemLabel.str.strip()
    wd = wd[(wd.label != "") & ~wd.label.str.startswith("Q")]
    vocab_all = sorted(set(wd.label))
    code = {}
    for n, g in wd.groupby("label"):
        c = pd.Series(hs_level(g.hs, 3))
        c = c[c != ""]
        code[n] = c.value_counts().index[0] if len(c) else ""
    enc = common.Encoder("siglip_b16")

    m = (rec.name != "") & (rec.hs3 != "")
    s2 = rec[m]
    vc = s2.name.value_counts()
    s2 = s2[s2.name.map(vc) >= MIN]
    idx = s2.index.values
    name, hs3 = arr(s2.name), arr(s2.hs3)
    vocab_labels = sorted(set(name))
    missing = sorted(set(name) - set(vocab_all))
    print(len(s2), "objects,", len(vocab_labels), "names;", len(vocab_all), "vocabulary names; label names not in the vocabulary:", missing, flush=True)
    Z2 = PCA(50, random_state=0).fit_transform(X[idx])
    XSi = XS[idx]
    parts = {k: KMeans(k, n_init=3, random_state=0).fit_predict(Z2) for k in (20, 50, 100, 200, 400)}

    rows, per_obj = [], []
    for vname, voc in (("all 945 Wikidata names", vocab_all), ("the 157 label names", vocab_labels)):
        T = np.concatenate([enc.embed_texts([PROMPT.format(v) for v in voc[i:i + 128]]) for i in range(0, len(voc), 128)])
        V = arr(voc)
        S = XSi @ T.T
        obj_name = V[S.argmax(1)]
        obj_code = arr(code.get(n, "") for n in obj_name)
        has = obj_code != ""
        per_obj.append({"vocabulary": vname, "objects": len(s2), "names": len(vocab_labels), "vocabulary_size": len(voc),
                        "per_object_zero_shot_name_accuracy": round(float((obj_name == name).mean()), 3),
                        "per_object_zero_shot_hs3_accuracy": round(float((obj_code[has] == hs3[has]).mean()), 3),
                        "objects_whose_zero_shot_name_has_a_code": round(float(has.mean()), 3),
                        "majority_name_share": round(float(pd.Series(name).value_counts(normalize=True).iloc[0]), 3)})
        print(per_obj[-1], flush=True)
        for k, c in parts.items():
            t = pd.DataFrame({"c": c, "name": name, "hs3": hs3, "zs": obj_name})
            g = t.groupby("c")
            maj = arr(g.name.agg(lambda s: s.value_counts().index[0]))
            maj_share = g.name.agg(lambda s: s.value_counts(normalize=True).iloc[0]).values
            maj3 = arr(g.hs3.agg(lambda s: s.value_counts().index[0]))
            size = g.size().values
            vote = arr(g.zs.agg(lambda s: s.value_counts().index[0]))
            vote_share = g.zs.agg(lambda s: s.value_counts(normalize=True).iloc[0]).values
            cen = normalize(np.vstack([XSi[c == j].mean(0) for j in range(k)]))
            sim = cen @ T.T
            order = np.argsort(-sim, 1)
            gname = V[order[:, 0]]
            margin = sim[np.arange(k), order[:, 0]] - sim[np.arange(k), order[:, 1]]
            gcode = arr(code.get(n, "") for n in gname)
            per_obj_name, per_obj_code = gname[c], gcode[c]
            hc = per_obj_code != ""
            r = {"vocabulary": vname, "k": k,
                 "groups_nameable_by_labels": int((maj_share >= 0.5).sum()),
                 "groups_nameable_without_labels": int((vote_share >= 0.5).sum()),
                 "groups_name_equals_majority_name": int((gname == maj).sum()),
                 "groups_vote_equals_majority_name": int((vote == maj).sum()),
                 "objects_right_name_majority_labelling": round(float((name == maj[c]).mean()), 3),
                 "objects_right_name_group_name": round(float((name == per_obj_name).mean()), 3),
                 "objects_right_name_vote_name": round(float((name == vote[c]).mean()), 3),
                 "objects_right_hs3_majority_labelling": round(float((hs3 == maj3[c]).mean()), 3),
                 "objects_right_hs3_group_name_code": round(float((hs3[hc] == per_obj_code[hc]).mean()), 3),
                 "objects_whose_group_name_has_a_code": round(float(hc.mean()), 3),
                 "distinct_group_names": int(len(set(gname))),
                 "distinct_majority_names": int(len(set(maj)))}
            rows.append(r)
            print(r, flush=True)
            if k == 100:
                tag = "all945" if voc is vocab_all else "labels157"
                pd.DataFrame({"group": range(k), "size": size, "majority_name": maj, "majority_share": np.round(maj_share, 2), "group_name": gname,
                              "second_name": V[order[:, 1]], "margin": np.round(margin, 3), "vote_name": vote, "vote_share": np.round(vote_share, 2)
                              }).to_csv(OUT / f"groups_k100_{tag}.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "naming.csv", index=False)
    pd.DataFrame(per_obj).to_csv(OUT / "per_object_zero_shot.csv", index=False)


if __name__ == "__main__":
    main()
