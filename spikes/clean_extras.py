"""Remaining comparisons on the clean set, so every statement in IDEAS.md comes from the same data.

On the 12,000-row methods subset (runs/clean/methods_subset_ids.csv), padded images:
  seeds (k-means, 3 seeds), per-museum centering, DINOv2-L, CLIP L/14, SigLIP B/16, ensemble DINOv2-L + SigLIP;
  zero-shot H-S level 1 from CLIP L/14 and SigLIP text prompts.
On the full clean set: origin of the instrument type from Wikipedia categories vs the museum's place of origin;
  within-name museum splits at 100 blind groups (violins, clarinets, transverse flutes).
Outputs: runs/clean/extras_*.csv.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import cached_embeddings  # noqa: E402
from mimo_wiki_origin import REGION  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/clean"
common.ENCODERS["siglip_b16"] = ("open_clip", "ViT-B-16-SigLIP", "webli")
common.ENCODERS["dinov2_l"] = ("timm", "vit_large_patch14_dinov2.lvd142m", None)
HS1 = {"1": "Idiophone", "2": "Membranophone", "3": "Chordophone", "4": "Aerophone", "5": "Electrophone"}
HS_PROMPTS = {"Aerophone": "a photo of a wind instrument", "Chordophone": "a photo of a string instrument",
              "Idiophone": "a photo of a percussion instrument that is struck or shaken, such as a bell, rattle or xylophone",
              "Membranophone": "a photo of a drum"}


def kmeans_ami(E, labels, seed=0):
    Z = PCA(50, random_state=0).fit_transform(E)
    out = {}
    for c, y in labels.items():
        out[c] = round(float(ami(y, KMeans(len(set(y)), n_init=10, random_state=seed).fit_predict(Z))), 3)
    return out


def main():
    df = pd.read_csv(RUN / "records.csv", dtype=str).fillna("")
    X = np.load(RUN / "emb/dinov2_b_pad.npy")
    pos = dict(zip(df.id, range(len(df))))
    ids = pd.read_csv(RUN / "methods_subset_ids.csv", dtype=str).id
    rows_idx = np.array([pos[i] for i in ids]); sub = df.iloc[rows_idx].reset_index(drop=True)
    labels = {}
    for c in ("hs3", "family", "name"):
        y = np.array(sub[c].tolist(), dtype=object); vc = pd.Series(y).value_counts(); labels[c] = y
        assert (pd.Series(y).map(vc) >= 5).all(), c
    Xb = X[rows_idx]
    res = []
    for seed in (0, 1, 2):
        r = {"variant": f"DINOv2-B, k-means seed {seed}", **kmeans_ami(Xb, labels, seed)}; res.append(r); print(r, flush=True)
    Xc = Xb.copy()
    for p in sub.provider.unique():
        sel = (sub.provider == p).values
        if sel.sum() >= 20:
            Xc[sel] -= Xc[sel].mean(0)
    r = {"variant": "DINOv2-B, per-museum centering", **kmeans_ami(Xc, labels)}; res.append(r); print(r, flush=True)
    paths = [str(ROOT / "data/mimo" / f) for f in sub.file]
    embs = {"dinov2_b": Xb}
    for enc in ("dinov2_l", "clip_l14", "siglip_b16"):
        embs[enc] = cached_embeddings(RUN / "emb_methods_subset", enc, paths, pad_square=True)
        r = {"variant": enc, **kmeans_ami(embs[enc], labels)}; res.append(r); print(r, flush=True)
    E = np.hstack([normalize(embs["dinov2_l"]), normalize(embs["siglip_b16"])])
    r = {"variant": "ensemble dinov2_l + siglip_b16", **kmeans_ami(E, labels)}; res.append(r); print(r, flush=True)
    pd.DataFrame(res).to_csv(RUN / "extras_encoders.csv", index=False)

    # zero-shot H-S level 1 (4 acoustic classes; electrophones dropped), scored with AMI and accuracy
    zs = []
    y1 = sub.hs1.map(HS1).values; keep = np.isin(y1, list(HS_PROMPTS))
    for enc in ("clip_l14", "siglip_b16"):
        T = common.Encoder(enc).embed_texts(list(HS_PROMPTS.values()))
        pred = np.array(list(HS_PROMPTS))[(embs[enc][keep] @ T.T).argmax(1)]
        r = {"encoder": enc, "n": int(keep.sum()), "AMI": round(float(ami(y1[keep], pred)), 3), "accuracy": round(float((pred == y1[keep]).mean()), 3),
             "majority": round(float(pd.Series(y1[keep]).value_counts(normalize=True).iloc[0]), 3)}
        zs.append(r); print(r, flush=True)
    pd.DataFrame(zs).to_csv(RUN / "extras_zeroshot.csv", index=False)

    # origin of the instrument type from Wikipedia categories, on the clean set
    wd = pd.read_csv(ROOT / "dl/wd_mimo_instruments.csv", dtype=str).fillna("")
    wk = pd.read_csv(ROOT / "dl/wiki_instrument_cats.csv", dtype=str).fillna("")
    wd["item"] = wd.item.str.replace("http://www.wikidata.org/entity/", ""); wd["has_en"] = (wd.en != "").astype(int)
    pick = wd.sort_values(["mimo", "has_en", "itemLabel"], ascending=[True, False, True]).drop_duplicates("mimo")
    page_of = dict(zip(pick.mimo, pick.en))
    regs_of = {}
    for p_, cats in zip(wk.en, wk.instrument_cats):
        regs = {REGION[m.group(1)] for c in cats.split("|") for m in [re.match(r"^([A-Z][A-Za-z\- ]+) musical instruments$", c)] if m and m.group(1) in REGION}
        regs_of[p_] = regs
    regs = df.kw.map(page_of).map(regs_of)
    df["type_region"] = [next(iter(r)) if isinstance(r, set) and len(r) == 1 else "" for r in regs]
    both = (df.type_region != "") & (df.region != "")
    Z = PCA(50, random_state=0).fit_transform(X)
    out = {"records_with_type_region": int((df.type_region != "").sum()), "with_both": int(both.sum()),
           "agreement": round(float((df.type_region[both] == df.region[both]).mean()), 3)}
    for col in ("type_region", "region"):
        y = df.loc[both, col].values; vc = pd.Series(y).value_counts(); k = (pd.Series(y).map(vc) >= 30).values
        out[f"AMI_{col}"] = round(float(ami(y[k], KMeans(len(set(y[k])), n_init=10, random_state=0).fit_predict(Z[both.values][k]))), 3)
    print(out, flush=True)
    pd.DataFrame([out]).to_csv(RUN / "extras_wiki_origin.csv", index=False)

    # within-name splits at 100 blind groups (same rows and clustering as clean_analysis block 5)
    s2 = df[(df.name != "") & (df.hs3 != "")]; vc = s2.name.value_counts(); s2 = s2[s2.name.map(vc) >= 30]
    c = KMeans(100, n_init=3, random_state=0).fit_predict(PCA(50, random_state=0).fit_transform(X[s2.index.values]))
    s2 = s2.assign(c=c)
    sp = []
    for n in ("violin", "clarinet", "transverse flute"):
        g = s2[s2.name == n]
        for grp, gg in list(g.groupby("c"))[:0]:
            pass
        top = g.c.value_counts().head(5)
        for grp in top.index:
            gg = g[g.c == grp]; mv = gg.provider.value_counts()
            sp.append({"name": n, "objects": len(g), "group": int(grp), "in_group": len(gg), "top_museum": mv.index[0], "top_museum_share": round(mv.iloc[0] / len(gg), 2)})
    sp = pd.DataFrame(sp); print(sp.to_string(index=False), flush=True); sp.to_csv(RUN / "extras_name_splits.csv", index=False)


if __name__ == "__main__":
    main()
