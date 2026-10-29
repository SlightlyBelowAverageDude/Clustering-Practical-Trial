"""Closeness of instrument names: does sharing a blind photo group predict that 2 objects carry close names, even when
the names differ? And do name clusters built from language fit the photo groups better than the raw names?

Objects: the 27,348 same rows (runs/clean/same_rows_ids.csv), 129 names. Name similarities (129 x 129):
  text   cosine of sentence embeddings (all-MiniLM-L6-v2 through transformers) of "<name>. <first paragraph of the
         English Wikipedia article>", intros fetched once into dl/wiki_name_intros.csv
  word   the same embedding of the bare name
  tree   Jaccard overlap of the names' ancestor sets in Wikidata's instrument tree (dl/wd_mimo_ancestors.csv)
  code   shared leading digits of the names' Wikidata Hornbostel-Sachs codes, divided by 6 (the expert reference)
Tests:
  pairs     pair AUC: for pairs of objects with different names, does sharing a blind photo group (k = 13, 50, 129)
            predict a high name similarity? References: sharing the level-3 code; permuted groups (chance).
  clusters  the 129 names grouped by text or tree similarity into 10, 20, 40 name groups (average-link); AMI of the
            blind photo groups (same k) against the name groups, next to the raw names, the H-S levels and the probe.
Outputs: runs/tests/names/pair_auc.csv, name_clusters.csv, nearest_names.csv
"""
import re
import sys
import urllib.parse
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
import wiki  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/tests/names"
INTROS = ROOT / "dl/wiki_name_intros.csv"
MIN = 30


def arr(s):
    return np.array(list(s), dtype=object)


def kept(y):
    vc = pd.Series(y).value_counts()
    return (pd.Series(y).map(vc).values >= MIN) & (y != "")


def items_for_names(names):
    wd = pd.read_csv(ROOT / "dl/wd_mimo_instruments.csv", dtype=str).fillna("")
    wd["label"] = wd.itemLabel.str.strip()
    wd["qid"] = wd.item.str.replace("http://www.wikidata.org/entity/", "", regex=False)
    rows = []
    for n in names:
        c = wd[wd.label == n]
        c = c.sort_values(["en", "qid"], ascending=[False, True])
        if c.empty:
            rows.append({"name": n, "qid": "", "title": "", "hs": ""})
            continue
        r = c.iloc[0]
        title = urllib.parse.unquote(r.en.rsplit("/", 1)[-1]).replace("_", " ") if r.en else ""
        hs = next((h for h in c.hs if h), "")
        rows.append({"name": n, "qid": r.qid, "title": title, "hs": hs})
    return pd.DataFrame(rows)


def fetch_intros(titles):
    if INTROS.exists():
        have = pd.read_csv(INTROS, dtype=str).fillna("")
    else:
        have = pd.DataFrame(columns=["title", "intro"])
    todo = [t for t in titles if t and t not in set(have.title)]
    s = wiki.session()
    rows = have.to_dict("records")
    for i in range(0, len(todo), 20):
        batch = todo[i:i + 20]
        resp = wiki.get(s, "https://en.wikipedia.org/w/api.php", params={"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                                                       "redirects": 1, "format": "json", "titles": "|".join(batch)})
        j = resp.json() if resp is not None else {}
        norm = {}
        for k in ("normalized", "redirects"):
            for d in (j.get("query", {}).get(k, []) or []):
                norm[d["to"]] = norm.get(d["from"], d["from"])
        for p in j.get("query", {}).get("pages", {}).values():
            t = norm.get(p.get("title", ""), p.get("title", ""))
            rows.append({"title": t, "intro": (p.get("extract") or "").strip()})
        print(f"  intros {min(i + 20, len(todo))} of {len(todo)}", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("title")
    df.to_csv(INTROS, index=False)
    return dict(zip(df.title, df.intro))


def embed_texts(texts):
    from transformers import AutoModel, AutoTokenizer
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModel.from_pretrained(name).eval().cuda()
    out = []
    for i in range(0, len(texts), 32):
        b = tok(texts[i:i + 32], padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = m(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return normalize(np.concatenate(out))


def code_digits(hs):
    return re.split(r"[-+]", str(hs))[0].replace(".", "").strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
    print(len(idx), "objects,", len(names), "names", flush=True)

    it = items_for_names(names)
    intros = fetch_intros(it.title.tolist())
    it["intro"] = [intros.get(t, "") for t in it.title]
    print("names with a Wikipedia intro:", int((it.intro != "").sum()), "with a tree item:", int((it.qid != "").sum()), "with a code:", int((it.hs != "").sum()), flush=True)
    E_text = embed_texts([f"{r.name}. {r.intro[:1200]}" if r.intro else r.name for r in it.itertuples()])
    E_word = embed_texts(list(it.name))
    S = {"text": E_text @ E_text.T, "word": E_word @ E_word.T}
    anc = pd.read_csv(ROOT / "dl/wd_mimo_ancestors.csv", dtype=str).fillna("")
    anc_of = anc.groupby("item").anc.agg(set).to_dict()
    sets = [({r.qid} | anc_of.get(r.qid, set())) if r.qid else set() for r in it.itertuples()]
    S["tree"] = np.array([[len(a & b) / len(a | b) if a and b else float(i == j) for j, b in enumerate(sets)] for i, a in enumerate(sets)])
    codes = [code_digits(h) for h in it.hs]

    def shared(a, b):
        if not a or not b:
            return 0.0
        n = 0
        for x, z in zip(a, b):
            if x != z:
                break
            n += 1
        return min(n, 6) / 6

    S["code"] = np.array([[shared(a, b) for b in codes] for a in codes])
    for k in S:
        np.fill_diagonal(S[k], 1.0)

    ex = []
    for n in ("violin", "Western concert flute", "duct flute", "clarinet", "bell", "xylophone", "guitar", "harmonica"):
        if n in nid:
            i = nid[n]
            for kind in ("text", "tree"):
                order = np.argsort(-S[kind][i])
                ex.append({"name": n, "similarity": kind, "nearest": ", ".join(f"{names[j]} ({S[kind][i, j]:.2f})" for j in order[1:6])})
    pd.DataFrame(ex).to_csv(OUT / "nearest_names.csv", index=False)
    for e in ex:
        print(e, flush=True)

    Z = PCA(50, random_state=0).fit_transform(X[idx])
    groups = {k: KMeans(k, n_init=10, random_state=0).fit_predict(Z) for k in (13, 50, 129)}
    rng = np.random.default_rng(0)
    n_pairs = 400000
    i = rng.integers(0, len(idx), n_pairs)
    j = rng.integers(0, len(idx), n_pairs)
    diff = y[i] != y[j]
    i, j = i[diff], j[diff]
    rows = []
    groupings = {f"photo groups, k = {k}": g for k, g in groups.items()}
    groupings["same level-3 code"] = np.array([hash(h) for h in hs3])
    groupings["permuted photo groups, k = 50"] = rng.permutation(groups[50])
    for gname, g in groupings.items():
        same = (g[i] == g[j]).astype(int)
        if gname == "same level-3 code":
            ok = (hs3[i] != "") & (hs3[j] != "")
        else:
            ok = np.ones(len(i), dtype=bool)
        r = {"grouping": gname, "pairs": int(ok.sum()), "same_pairs": int(same[ok].sum())}
        for kind, M in S.items():
            sim = M[y[i], y[j]]
            r[f"AUC_{kind}"] = round(float(roc_auc_score(same[ok], sim[ok])), 3)
            r[f"mean_{kind}_same"] = round(float(sim[ok][same[ok] == 1].mean()), 3)
            r[f"mean_{kind}_different"] = round(float(sim[ok][same[ok] == 0].mean()), 3)
        rows.append(r)
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "pair_auc.csv", index=False)

    rows = []
    hs = {d: arr(rec[f"hs{d}"].iloc[idx]) for d in (3, 4, 5, 6)}
    for kind in ("text", "tree"):
        D = 1 - S[kind]
        np.fill_diagonal(D, 0)
        for K in (10, 20, 40):
            cl = AgglomerativeClustering(K, metric="precomputed", linkage="average").fit_predict(D)
            lab = cl[y]
            p = KMeans(K, n_init=10, random_state=0).fit_predict(Z)
            r = {"name_similarity": kind, "name_groups": K, "AMI_photo_groups_vs_name_groups": round(float(ami(lab, p)), 3),
                 "AMI_photo_groups_vs_raw_names": round(float(ami(y, p)), 3)}
            for d, h in hs.items():
                m = h != ""
                r[f"AMI_photo_groups_vs_hs{d}"] = round(float(ami(h[m], p[m])), 3)
                r[f"AMI_name_groups_vs_hs{d}"] = round(float(ami(h[m], lab[m])), 3)
            pr = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), X[idx], lab, cv=StratifiedKFold(5, shuffle=True, random_state=0))
            r["probe_AMI_name_groups"] = round(float(ami(lab, pr)), 3)
            r["largest_name_group_share"] = round(float(pd.Series(lab).value_counts(normalize=True).iloc[0]), 3)
            r["example_group"] = "; ".join(names[t] for t in np.where(cl == pd.Series(cl[y]).value_counts().index[1])[0][:8])
            rows.append(r)
            print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "name_clusters.csv", index=False)


if __name__ == "__main__":
    main()
