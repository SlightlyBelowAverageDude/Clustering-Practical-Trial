"""More on graded agreement (after test_name_similarity2.py).

hsgraded  graded agreement with the expert code: shared leading digits between an object's code and its group's majority
          code (0 to 6), as a distribution over objects, for 20 to 400 blind groups, shuffled groups and the probe. Rows:
          the 27,348 same rows with their full code.
medoid    naming a group through the knowledge graph: the per-object zero-shot names (SigLIP, 157-name vocabulary, as in
          test_naming.py) are combined by choosing the vocabulary name with the highest summed Wikidata-tree closeness to
          the votes, against the plain vote and the majority label; 100 blind groups on the 31,402 blind-then-label rows.
triangle  on the 868 recorded objects with a name: Spearman correlation across object pairs with different names between
          photo similarity (DINOv2 cosine), sound similarity (CLAP, and timbre statistics) and name closeness (tree); AUC
          that name closeness of at least 0.5 predicts sound similarity.
groups    the 100 blind groups sorted: nameable (one name covers half), coherent (mean pairwise tree closeness of the
          members at least 0.4 but no single name), crossing (neither); with the names of each crossing group.
Outputs: runs/tests/names/hs_graded.csv, medoid_naming.csv, triangle.csv, groups_sorted.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from test_name_similarity import code_digits, items_for_names  # noqa: E402
from test_name_similarity2 import tree_similarity  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/tests/names"
MIN = 30
PROMPT = "a photo of a {}, a musical instrument"


def arr(s):
    return np.array(list(s), dtype=object)


def kept(y):
    vc = pd.Series(y).value_counts()
    return (pd.Series(y).map(vc).values >= MIN) & (y != "")


def shared_digits(a, b):
    n = 0
    for x, z in zip(a, b):
        if x != z:
            break
        n += 1
    return min(n, 6)


def tree_for(names, anc_of):
    it = items_for_names(names)
    return tree_similarity(it, anc_of)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = sys.argv[1:] or ["hsgraded", "medoid", "triangle", "groups"]
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    X = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")
    pos = dict(zip(rec.id, range(len(rec))))
    anc = pd.read_csv(ROOT / "dl/wd_mimo_ancestors.csv", dtype=str).fillna("")
    anc_of = anc.groupby("item").anc.agg(set).to_dict()
    rng = np.random.default_rng(0)

    if "hsgraded" in blocks:
        idx = np.array([pos[i] for i in pd.read_csv(ROOT / "runs/clean/same_rows_ids.csv", dtype=str).id])
        codes = arr(code_digits(n) for n in rec.notation.iloc[idx])
        Z = PCA(50, random_state=0).fit_transform(X[idx])
        hs3 = arr(rec.hs3.iloc[idx])
        rows = []

        def score(group_code, tag, k):
            d = np.array([shared_digits(a, b) for a, b in zip(codes, group_code)])
            full = np.array([min(len(a), 6) for a in codes])
            r = {"grouping": tag, "k": k, "mean_shared_digits": round(float(d.mean()), 2), "exact_code": round(float((d >= full).mean()), 3)}
            for n in range(0, 7):
                r[f"share_at_least_{n}"] = round(float((d >= n).mean()), 3)
            rows.append(r)
            print(r, flush=True)

        for k in (20, 50, 100, 200, 400):
            c = KMeans(k, n_init=3, random_state=0).fit_predict(Z)
            maj = pd.Series(codes).groupby(c).agg(lambda s: s.value_counts().index[0])
            score(maj.loc[c].values, "blind groups, majority code", k)
            cs = rng.permutation(c)
            majs = pd.Series(codes).groupby(cs).agg(lambda s: s.value_counts().index[0])
            score(majs.loc[cs].values, "shuffled groups, majority code", k)
        keep3 = kept(hs3)
        pr = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), X[idx][keep3], hs3[keep3], cv=StratifiedKFold(5, shuffle=True, random_state=0))
        full_pr = np.full(len(idx), "", dtype=object)
        full_pr[keep3] = pr
        score(full_pr, "probe on level-3 codes (3 digits at most)", 13)
        pd.DataFrame(rows).to_csv(OUT / "hs_graded.csv", index=False)

    # rows and groups of the naming test (blind-then-label rows, 100 groups)
    m = (rec.name != "") & (rec.hs3 != "")
    s2 = rec[m]
    vc = s2.name.value_counts()
    s2 = s2[s2.name.map(vc) >= MIN]
    idx2 = s2.index.values
    name2, hs32 = arr(s2.name), arr(s2.hs3)
    vocab = sorted(set(name2))
    vid = {n: i for i, n in enumerate(vocab)}
    y2 = np.array([vid[n] for n in name2])
    T_tree = tree_for(vocab, anc_of)
    c100 = KMeans(100, n_init=3, random_state=0).fit_predict(PCA(50, random_state=0).fit_transform(X[idx2]))
    maj2 = pd.Series(y2).groupby(c100).agg(lambda s: s.value_counts().index[0]).values
    maj_share = pd.Series(y2).groupby(c100).agg(lambda s: s.value_counts(normalize=True).iloc[0]).values
    sizes = np.bincount(c100, minlength=100)

    if "medoid" in blocks:
        common.ENCODERS["siglip_b16"] = ("open_clip", "ViT-B-16-SigLIP", "webli")
        paths = [str(ROOT / "data/mimo" / f) for f in rec.file]
        XS = common.cached_embeddings(ROOT / "runs/tests/naming/emb", "siglip_b16", paths, pad_square=True)[idx2]
        enc = common.Encoder("siglip_b16")
        Tt = np.concatenate([enc.embed_texts([PROMPT.format(v) for v in vocab[i:i + 128]]) for i in range(0, len(vocab), 128)])
        votes = (XS @ Tt.T).argmax(1)
        code_of = {}
        it = items_for_names(vocab)
        for n, h in zip(it.name, it.hs):
            code_of[vid[n]] = code_digits(h)[:3]
        rows = []
        names_out = {}
        for tag in ("plain vote", "tree-smoothed vote"):
            g = np.zeros(100, dtype=int)
            for j in range(100):
                v = votes[c100 == j]
                if tag == "plain vote":
                    g[j] = np.bincount(v, minlength=len(vocab)).argmax()
                else:
                    g[j] = T_tree[v].sum(0).argmax()
            names_out[tag] = g
            cl = T_tree[g, maj2]
            own = g[c100]
            right_code = np.array([code_of.get(a, "") == b and b != "" for a, b in zip(own, hs32)])
            has_code = np.array([code_of.get(a, "") != "" for a in own])
            r = {"naming": tag, "groups_exact": int((g == maj2).sum()), "groups_within_0.5": int((cl >= 0.5).sum()), "mean_closeness_to_majority": round(float(cl.mean()), 3),
                 "objects_right_name": round(float((own == y2).mean()), 3), "objects_right_hs3_via_name_code": round(float(right_code[has_code].mean()), 3),
                 "distinct_names": int(len(set(g)))}
            rows.append(r)
            print(r, flush=True)
        r = {"naming": "majority label (reference)", "groups_exact": 100, "groups_within_0.5": 100, "mean_closeness_to_majority": 1.0,
             "objects_right_name": round(float((maj2[c100] == y2).mean()), 3), "distinct_names": int(len(set(maj2)))}
        rows.append(r)
        pd.DataFrame(rows).to_csv(OUT / "medoid_naming.csv", index=False)
        pd.DataFrame({"group": range(100), "size": sizes, "majority_name": [vocab[i] for i in maj2], "majority_share": np.round(maj_share, 2),
                      "plain_vote": [vocab[i] for i in names_out["plain vote"]], "tree_smoothed": [vocab[i] for i in names_out["tree-smoothed vote"]]
                      }).to_csv(OUT / "medoid_groups_k100.csv", index=False)

    if "triangle" in blocks:
        au = pd.read_csv(ROOT / "runs/clean/audio_dedup/objects.csv", dtype=str).fillna("")
        rows_a = np.array([pos[i] for i in au.id])
        nm = arr(rec.name.iloc[rows_a])
        has = nm != ""
        names_a = sorted(set(nm[has]))
        aid = {n: i for i, n in enumerate(names_a)}
        Ta = tree_for(names_a, anc_of)
        XI = normalize(X[rows_a][has])
        XA = normalize(np.load(ROOT / "runs/clean/audio_dedup/emb_clap.npy")[has])
        XM = np.load(ROOT / "runs/tests/audio/emb_mfcc.npy")[has]
        XM = normalize((XM - XM.mean(0)) / (XM.std(0) + 1e-8))
        ya = np.array([aid[n] for n in nm[has]])
        n = len(ya)
        iu, ju = np.triu_indices(n, 1)
        diff = ya[iu] != ya[ju]
        iu, ju = iu[diff], ju[diff]
        s_img = (XI[iu] * XI[ju]).sum(1)
        s_clap = (XA[iu] * XA[ju]).sum(1)
        s_mfcc = (XM[iu] * XM[ju]).sum(1)
        s_name = Ta[ya[iu], ya[ju]]
        close = (s_name >= 0.5).astype(int)
        rows = [{"objects": int(n), "pairs_different_names": int(len(iu)), "pairs_name_closeness_at_least_0.5": int(close.sum()),
                 "spearman_name_vs_clap": round(float(spearmanr(s_name, s_clap)[0]), 3), "spearman_name_vs_timbre": round(float(spearmanr(s_name, s_mfcc)[0]), 3),
                 "spearman_photo_vs_clap": round(float(spearmanr(s_img, s_clap)[0]), 3), "spearman_photo_vs_timbre": round(float(spearmanr(s_img, s_mfcc)[0]), 3),
                 "spearman_name_vs_photo": round(float(spearmanr(s_name, s_img)[0]), 3),
                 "AUC_close_names_predict_clap": round(float(roc_auc_score(close, s_clap)), 3), "AUC_close_names_predict_timbre": round(float(roc_auc_score(close, s_mfcc)), 3),
                 "AUC_close_names_predict_photo": round(float(roc_auc_score(close, s_img)), 3)}]
        print(rows[0], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "triangle.csv", index=False)

    if "groups" in blocks:
        hs1 = arr(rec.hs1.iloc[idx2])
        rows = []
        for j in range(100):
            mem = np.where(c100 == j)[0]
            a = rng.choice(mem, 3000)
            b = rng.choice(mem, 3000)
            ok = a != b
            coh = float(Ta_mean(T_tree, y2[a[ok]], y2[b[ok]]))
            top = pd.Series(name2[mem]).value_counts(normalize=True).head(3)
            h1 = pd.Series(hs1[mem]).value_counts(normalize=True).iloc[0]
            kind = "nameable" if maj_share[j] >= 0.5 else ("coherent" if coh >= 0.4 else "crossing")
            rows.append({"group": j, "size": int(sizes[j]), "kind": kind, "majority_share": round(float(maj_share[j]), 2), "mean_pairwise_tree_closeness": round(coh, 3),
                         "dominant_hs1_share": round(float(h1), 2), "top_names": "; ".join(f"{k} {v:.2f}" for k, v in top.items())})
        t = pd.DataFrame(rows).sort_values(["kind", "size"], ascending=[True, False])
        t.to_csv(OUT / "groups_sorted.csv", index=False)
        pd.set_option("display.width", 220)
        pd.set_option("display.max_colwidth", 80)
        print(t.kind.value_counts().to_dict(), flush=True)
        print(t.groupby("kind")[["size", "mean_pairwise_tree_closeness", "dominant_hs1_share"]].mean().round(3).to_string(), flush=True)
        print(t[t.kind == "crossing"][["size", "mean_pairwise_tree_closeness", "dominant_hs1_share", "top_names"]].to_string(index=False), flush=True)


def Ta_mean(T, a, b):
    return T[a, b].mean()


if __name__ == "__main__":
    main()
