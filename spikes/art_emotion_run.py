"""Art emotion spike: which visual clusters of paintings evoke the same emotion in every language group?

Data: ArtELingo annotations (English, Arabic, Chinese emotion labels per WikiArt painting) joined to the
WikiArt images on Hugging Face (Artificio/WikiArt) by artist and title slug.

Per cluster of image embeddings, the emotion distribution of each language is compared with the
Jensen-Shannon divergence (JSD). The reference is the JSD between 2 random halves of the same language
inside the same cluster (annotator noise). A cluster "splits by culture" when the cross-language JSD is
well above that within-language reference.

Usage: python spikes/art_emotion_run.py [n_paintings] [encoders]
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings, cluster_and_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "dl/artelingo_raw/artelingo_release_lite.csv"
RUNS = ROOT / "runs/art_emotion"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
ENCODERS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["clip_b32", "dinov2_s"]
EMOTIONS = ["amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness", "something else"]
LANGS = ["english", "arabic", "chinese"]
MIN_PER_LANG = 5


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s


def painting_key(p: str) -> str:
    """ArtELingo painting id: '<artist-slug>_<title-slug>(-<year>)'. Drop a trailing year."""
    return re.sub(r"-\d{4}$", "", p)


def load_annotations():
    cols = ["art_style", "emotion", "language", "painting"]
    df = pd.read_csv(CSV, usecols=lambda c: c in cols)
    counts = df.groupby(["painting", "language"]).size().unstack(fill_value=0)
    ok = counts[(counts[LANGS] >= MIN_PER_LANG).all(axis=1)].index
    df = df[df.painting.isin(ok)]
    print(f"{df.painting.nunique()} paintings with >= {MIN_PER_LANG} annotations in each of {LANGS}")
    return df


def join_wikiart(paintings):
    ds = load_dataset("Artificio/WikiArt", split="train")
    meta = ds.remove_columns(["image", "embeddings_pca512"]).to_pandas()
    meta["key"] = meta.artist.map(slug) + "_" + meta.title.map(slug)
    meta = meta.drop_duplicates("key")
    want = pd.DataFrame({"painting": paintings})
    want["key"] = want.painting.map(painting_key)
    m = want.merge(meta[["key", "title", "artist", "genre", "style"]].assign(row=meta.index), on="key", how="inner")
    print(f"matched {len(m)} of {len(want)} paintings to WikiArt images ({len(m) / len(want):.1%})")
    return ds, m


def emotion_dist(sub):
    v = sub.emotion.value_counts().reindex(EMOTIONS, fill_value=0).values.astype(float)
    return v / max(v.sum(), 1)


def cluster_report(ann, labels, painting_order, rng, k):
    """Cross-language vs within-language JSD per cluster."""
    lab = pd.Series(labels, index=painting_order, name="cluster")
    a = ann.join(lab, on="painting").dropna(subset=["cluster"])
    rows = []
    for c, g in a.groupby("cluster"):
        dists = {l: emotion_dist(g[g.language == l]) for l in LANGS}
        cross = np.mean([jensenshannon(dists[x], dists[y], base=2) for x, y in
                         (("english", "arabic"), ("english", "chinese"), ("arabic", "chinese"))])
        within = []
        for l in LANGS:
            gl = g[g.language == l]
            idx = rng.permutation(len(gl))
            h1, h2 = gl.iloc[idx[: len(gl) // 2]], gl.iloc[idx[len(gl) // 2:]]
            within.append(jensenshannon(emotion_dist(h1), emotion_dist(h2), base=2))
        top = {l: EMOTIONS[int(np.argmax(dists[l]))] for l in LANGS}
        rows.append({"k": k, "cluster": int(c), "n_paintings": g.painting.nunique(), "n_ann": len(g),
                     "jsd_cross": cross, "jsd_within": float(np.mean(within)), **{f"top_{l[:2]}": top[l] for l in LANGS}})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(0)
    ann = load_annotations()
    paintings = ann.painting.unique()
    ds, m = join_wikiart(paintings)
    m = m.sample(n=min(N, len(m)), random_state=0).reset_index(drop=True)
    ann = ann[ann.painting.isin(m.painting)]
    images = [ds[int(i)]["image"] for i in m.row]
    RUNS.mkdir(parents=True, exist_ok=True)
    m.drop(columns=["row"]).to_csv(RUNS / f"sample_{N}.csv", index=False)

    # painting-level majority emotion per language, as cluster ground truth candidates
    maj = ann.groupby(["painting", "language"]).emotion.agg(lambda s: s.value_counts().idxmax()).unstack()
    maj = maj.loc[m.painting]

    # painting-level cross-language agreement (the ArtELingo / ArtECulture style number)
    same3 = (maj.english == maj.arabic) & (maj.english == maj.chinese)
    print(f"paintings where the majority emotion agrees in all 3 languages: {same3.mean():.1%}")

    # global cross-language divergence over all sampled paintings: the base-rate difference between
    # language groups. A cluster only "splits by culture" beyond this if its JSD exceeds the global one.
    glob = {l: emotion_dist(ann[ann.language == l]) for l in LANGS}
    glob_cross = np.mean([jensenshannon(glob[x], glob[y], base=2) for x, y in
                          (("english", "arabic"), ("english", "chinese"), ("arabic", "chinese"))])
    print(f"global cross-language JSD (all paintings): {glob_cross:.3f}")
    for l in LANGS:
        print(f"  {l:8s} " + " ".join(f"{e[:5]}={v:.2f}" for e, v in zip(EMOTIONS, glob[l])))

    reports = []
    for enc in ENCODERS:
        X = cached_embeddings(RUNS / f"emb_{N}", enc, images)
        print(f"\n== {enc}: clusters vs majority emotion per language, vs genre, vs style")
        for name, y in (("emotion_en", maj.english.values), ("emotion_ar", maj.arabic.values),
                        ("emotion_zh", maj.chinese.values), ("genre", m["genre"].fillna("?").values),
                        ("style", m["style"].fillna("?").values)):
            r = cluster_and_score(X, y, ks=(9, 20), methods=("kmeans",), hdbscan_min_cluster_size=None)
            print(f"  {name:11s} " + "  ".join(f"k={int(k)} AMI={a:.3f}" for k, a in zip(r.k, r.AMI)))
        Z = PCA(n_components=50, random_state=0).fit_transform(X)
        for k in (8, 16, 32):
            labels = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
            rep = cluster_report(ann, labels, m.painting.values, rng, k)
            rep.insert(0, "encoder", enc)
            reports.append(rep)
            rep["jsd_cross_minus_global"] = rep.jsd_cross - glob_cross
            above = (rep.jsd_cross > glob_cross + rep.jsd_within).mean()
            print(f"  k={k:2d}: JSD cross-language mean {rep.jsd_cross.mean():.3f} (sd {rep.jsd_cross.std():.3f},"
                  f" min {rep.jsd_cross.min():.3f}, max {rep.jsd_cross.max():.3f}); within-language {rep.jsd_within.mean():.3f};"
                  f" clusters above global + within noise: {above:.0%}")
    rep = pd.concat(reports)
    rep.to_csv(RUNS / f"cluster_report_{N}.csv", index=False)
    r16 = rep[(rep.k == 16) & (rep.encoder == ENCODERS[0])].sort_values("jsd_cross", ascending=False)
    print(f"\n== {ENCODERS[0]}, k=16, clusters sorted by cross-language divergence (top 5 and bottom 3)")
    print(pd.concat([r16.head(5), r16.tail(3)]).drop(columns=["encoder", "k"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
