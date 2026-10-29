"""Follow-ups on the closeness of names (test_name_similarity.py). Same rows (27,348 objects, 129 names) unless stated.

repr      which language representation knows instruments: MiniLM on name + Wikipedia intro, MiniLM on the intro alone,
          bge-base on name + intro, SigLIP's text tower on "a photo of a <name>, a musical instrument", the Wikidata tree,
          the expert code. Yardstick at the name level: AUC that 2 names with the same level-3 code are closer than 2 names
          with different codes (8,256 name pairs). Then the object-level pair test (different-name pairs, photo groups
          k = 129) and name groups at K = 20 as in the first test.
graded    agreement as closeness: each object's name against its group's majority name (tree and MiniLM text), mean over all
          objects and over the wrongly named ones, at 20 to 400 groups, against shuffled groups of the same sizes; the
          probe's own errors scored the same way.
zeroshot  the 100 zero-shot group names of runs/tests/naming/groups_k100_labels157.csv against the majority names, by
          closeness instead of exact match; random name pairs as the baseline.
wdnodes   node map on the names' own tree: every Wikidata ancestor class of the names (30+ objects, below the root) and
          every H-S node, scored on the named objects of the 30,050 six-digit set by the best matching subtree of the
          saved average-link tree (F1 over named leaves).
Outputs: runs/tests/names/repr.csv, graded.csv, zeroshot_graded.csv, wdnodes.csv, wdnodes_summary.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score as ami, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from test_name_similarity import code_digits, embed_texts, fetch_intros, items_for_names  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/tests/names"
MIN = 30


def arr(s):
    return np.array(list(s), dtype=object)


def kept(y):
    vc = pd.Series(y).value_counts()
    return (pd.Series(y).map(vc).values >= MIN) & (y != "")


def embed_bge(texts):
    from transformers import AutoModel, AutoTokenizer
    name = "BAAI/bge-base-en-v1.5"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModel.from_pretrained(name).eval().cuda()
    out = []
    for i in range(0, len(texts), 32):
        b = tok(texts[i:i + 32], padding=True, truncation=True, max_length=384, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out.append(m(**b).last_hidden_state[:, 0].cpu().numpy())
    return normalize(np.concatenate(out))


def shared_prefix(a, b):
    if not a or not b:
        return 0.0
    n = 0
    for x, z in zip(a, b):
        if x != z:
            break
        n += 1
    return min(n, 6) / 6


def tree_similarity(it, anc_of):
    sets = [({r.qid} | anc_of.get(r.qid, set())) if r.qid else set() for r in it.itertuples()]
    S = np.array([[len(a & b) / len(a | b) if a and b else float(i == j) for j, b in enumerate(sets)] for i, a in enumerate(sets)])
    np.fill_diagonal(S, 1.0)
    return S


def similarities(names, anc_of, with_models=True):
    it = items_for_names(names)
    intros = fetch_intros(it.title.tolist())
    it["intro"] = [intros.get(t, "") for t in it.title]
    S = {}
    if with_models:
        E = embed_texts([f"{r.name}. {r.intro[:1200]}" if r.intro else r.name for r in it.itertuples()])
        S["MiniLM, name + intro"] = E @ E.T
        E = embed_texts([r.intro[:1200] if r.intro else r.name for r in it.itertuples()])
        S["MiniLM, intro only"] = E @ E.T
        E = embed_bge([f"{r.name}. {r.intro[:1500]}" if r.intro else r.name for r in it.itertuples()])
        S["bge-base, name + intro"] = E @ E.T
        common.ENCODERS["siglip_b16"] = ("open_clip", "ViT-B-16-SigLIP", "webli")
        enc = common.Encoder("siglip_b16")
        E = enc.embed_texts([f"a photo of a {n}, a musical instrument" for n in it.name])
        S["SigLIP text"] = E @ E.T
    S["Wikidata tree"] = tree_similarity(it, anc_of)
    codes = [code_digits(h) for h in it.hs]
    S["expert code"] = np.array([[shared_prefix(a, b) for b in codes] for a in codes])
    for k in S:
        np.fill_diagonal(S[k], 1.0)
    return it, S


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = sys.argv[1:] or ["repr", "graded", "zeroshot", "wdnodes"]
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    X = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")
    pos = dict(zip(rec.id, range(len(rec))))
    idx = np.array([pos[i] for i in pd.read_csv(ROOT / "runs/clean/same_rows_ids.csv", dtype=str).id])
    name_all = arr(rec.name.iloc[idx])
    keep = kept(name_all)
    idx, name = idx[keep], name_all[keep]
    hs3 = arr(rec.hs3.iloc[idx])
    names = sorted(set(name))
    nid = {n: i for i, n in enumerate(names)}
    y = np.array([nid[n] for n in name])
    anc = pd.read_csv(ROOT / "dl/wd_mimo_ancestors.csv", dtype=str).fillna("")
    anc_of = anc.groupby("item").anc.agg(set).to_dict()
    it, S = similarities(names, anc_of)
    Z = PCA(50, random_state=0).fit_transform(X[idx])
    print(len(idx), "objects,", len(names), "names;", list(S), flush=True)

    if "repr" in blocks:
        codes3 = np.array([code_digits(h)[:3] for h in it.hs], dtype=object)
        iu, ju = np.triu_indices(len(names), 1)
        ok = (codes3[iu] != "") & (codes3[ju] != "")
        same_code = (codes3[iu] == codes3[ju]).astype(int)[ok]
        p = KMeans(129, n_init=10, random_state=0).fit_predict(Z)
        rng = np.random.default_rng(0)
        i = rng.integers(0, len(idx), 400000)
        j = rng.integers(0, len(idx), 400000)
        d = y[i] != y[j]
        i, j = i[d], j[d]
        same_photo = (p[i] == p[j]).astype(int)
        rows = []
        for kind, M in S.items():
            r = {"representation": kind, "name_pairs_same_code_AUC": round(float(roc_auc_score(same_code, M[iu, ju][ok])), 3),
                 "object_pairs_photo_group_k129_AUC": round(float(roc_auc_score(same_photo, M[y[i], y[j]])), 3)}
            D = 1 - M
            np.fill_diagonal(D, 0)
            cl = AgglomerativeClustering(20, metric="precomputed", linkage="average").fit_predict(D)
            lab = cl[y]
            pk = KMeans(20, n_init=10, random_state=0).fit_predict(Z)
            r["name_groups_20_AMI_vs_photo_groups"] = round(float(ami(lab, pk)), 3)
            r["name_groups_20_largest_share"] = round(float(pd.Series(lab).value_counts(normalize=True).iloc[0]), 3)
            r["nearest_to_harmonica"] = ", ".join(names[t] for t in np.argsort(-M[nid["harmonica"]])[1:4]) if "harmonica" in nid else ""
            r["nearest_to_violin"] = ", ".join(names[t] for t in np.argsort(-M[nid["violin"]])[1:4]) if "violin" in nid else ""
            rows.append(r)
            print(r, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "repr.csv", index=False)

    if "graded" in blocks:
        T, W = S["Wikidata tree"], S["MiniLM, name + intro"]
        rows = []
        rng = np.random.default_rng(0)

        def score(assign_names, tag, k):
            right = assign_names == y
            r = {"grouping": tag, "k": k, "objects_right_name": round(float(right.mean()), 3),
                 "closeness_tree_all": round(float(T[y, assign_names].mean()), 3), "closeness_text_all": round(float(W[y, assign_names].mean()), 3),
                 "closeness_tree_wrong_only": round(float(T[y, assign_names][~right].mean()), 3), "closeness_text_wrong_only": round(float(W[y, assign_names][~right].mean()), 3),
                 "wrong_with_tree_closeness_at_least_0.5": round(float((T[y, assign_names][~right] >= 0.5).mean()), 3)}
            rows.append(r)
            print(r, flush=True)

        for k in (20, 50, 100, 200, 400):
            c = KMeans(k, n_init=3, random_state=0).fit_predict(Z)
            maj = pd.Series(y).groupby(c).agg(lambda s: s.value_counts().index[0])
            score(maj.loc[c].values, "blind groups, majority name", k)
            cs = rng.permutation(c)
            majs = pd.Series(y).groupby(cs).agg(lambda s: s.value_counts().index[0])
            score(majs.loc[cs].values, "shuffled groups of the same sizes, majority name", k)
        pr = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), X[idx], y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
        score(pr, "probe, predicted name", 129)
        rp = rng.permutation(y)
        score(rp, "random name", 129)
        pd.DataFrame(rows).to_csv(OUT / "graded.csv", index=False)

    if "zeroshot" in blocks:
        g = pd.read_csv(ROOT / "runs/tests/naming/groups_k100_labels157.csv", dtype=str).fillna("")
        names157 = sorted(set(g.majority_name) | set(g.group_name) | set(g.vote_name))
        it2, S2 = similarities(names157, anc_of, with_models=True)
        n2 = {n: i for i, n in enumerate(names157)}
        rng = np.random.default_rng(0)
        rows = []
        for col in ("group_name", "vote_name"):
            a = np.array([n2[v] for v in g[col]])
            b = np.array([n2[v] for v in g.majority_name])
            size = g["size"].astype(int).values
            for kind in ("Wikidata tree", "MiniLM, name + intro", "expert code"):
                M = S2[kind]
                cl = M[a, b]
                ra = rng.integers(0, len(names157), 10000)
                rb = rng.integers(0, len(names157), 10000)
                base = M[ra, rb][ra != rb]
                rows.append({"zero_shot_name": col, "closeness": kind, "groups_exact": int((a == b).sum()), "mean_closeness": round(float(cl.mean()), 3),
                             "mean_closeness_weighted_by_objects": round(float((cl * size).sum() / size.sum()), 3),
                             "groups_closeness_at_least_0.5": int((cl >= 0.5).sum()), "random_name_pair_mean": round(float(base.mean()), 3),
                             "random_pairs_at_least_0.5": round(float((base >= 0.5).mean()), 3)})
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "zeroshot_graded.csv", index=False)

    if "wdnodes" in blocks:
        from test_tree import node_sizes
        Zl = np.load(sorted((ROOT / "runs/tests/tree").glob("linkage_average*.npy"))[0])
        sel = np.where((rec.hs6 != "").values)[0]
        N = len(sel)
        nm = arr(rec.name.iloc[sel])
        named = nm != ""
        it6 = items_for_names(sorted(set(nm[named])))
        qid_of = dict(zip(it6.name, it6.qid))
        wd = pd.read_csv(ROOT / "dl/wd_mimo_instruments.csv", dtype=str).fillna("")
        label_of = dict(zip(wd.item.str.replace("http://www.wikidata.org/entity/", "", regex=False), wd.itemLabel))
        label_of.update(dict(zip(anc.anc, anc.ancLabel)))
        members = {}
        for o, n in enumerate(nm):
            q = qid_of.get(n, "") if n else ""
            if not q:
                continue
            for a in ({q} | anc_of.get(q, set())):
                members.setdefault(a, set()).add(o)
        n_named = int(named.sum())
        classes, seen = [], {}
        for q, mem in members.items():
            key = frozenset(mem)
            if MIN <= len(mem) <= 0.95 * n_named and key not in seen:
                seen[key] = q
                classes.append(("Wikidata", q, label_of.get(q, q), mem))
        for d in range(1, 7):
            col = arr(rec[f"hs{d}"].iloc[sel])
            for c in sorted(set(col[named])):
                mem = set(np.where((col == c) & named)[0])
                if len(mem) >= MIN:
                    classes.append(("Hornbostel-Sachs", c, f"level {d}", mem))
        C = len(classes)
        M = np.zeros((2 * N - 1, C + 1), dtype=np.int32)
        for j, (_, _, _, mem) in enumerate(classes):
            M[list(mem), j] = 1
        M[np.where(named)[0], C] = 1
        for m in range(N - 1):
            a, b = int(Zl[m, 0]), int(Zl[m, 1])
            M[N + m] = M[a] + M[b]
        sizes_named = M[:, C].astype(np.float64)
        rows = []
        for j, (tax, code, label, mem) in enumerate(classes):
            cnt = M[:, j].astype(np.float64)
            f1 = 2 * cnt / (sizes_named + len(mem))
            b = int(f1.argmax())
            rows.append({"taxonomy": tax, "code": code, "label": label, "objects": len(mem), "F1": round(float(f1[b]), 3),
                         "precision": round(float(cnt[b] / sizes_named[b]), 3), "recall": round(float(cnt[b] / len(mem)), 3), "subtree_named_leaves": int(sizes_named[b])})
        t = pd.DataFrame(rows).sort_values(["taxonomy", "objects"], ascending=[True, False])
        t.to_csv(OUT / "wdnodes.csv", index=False)
        bins = [(30, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10**6)]
        srows = []
        for tax, g in t.groupby("taxonomy"):
            for lo, hi in bins:
                h = g[(g.objects >= lo) & (g.objects < hi)]
                if len(h):
                    srows.append({"taxonomy": tax, "objects_from": lo, "objects_below": hi, "nodes": len(h), "mean_F1": round(float(h.F1.mean()), 3),
                                  "median_F1": round(float(h.F1.median()), 3), "nodes_F1_at_least_0.5": int((h.F1 >= 0.5).sum())})
            srows.append({"taxonomy": tax, "objects_from": 30, "objects_below": 10**6, "nodes": len(g), "mean_F1": round(float(g.F1.mean()), 3),
                          "median_F1": round(float(g.F1.median()), 3), "nodes_F1_at_least_0.5": int((g.F1 >= 0.5).sum())})
        s = pd.DataFrame(srows)
        s.to_csv(OUT / "wdnodes_summary.csv", index=False)
        pd.set_option("display.width", 200)
        print(s.to_string(index=False), flush=True)
        print(t[t.taxonomy == "Wikidata"].head(25).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
