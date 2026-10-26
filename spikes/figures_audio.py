"""Figures for the seen-and-heard test (runs/mimo_audio) and the data for the cluster gallery page.

Outputs: runs/figures_pitch/figF_pair_auc.png, figG_audio_vs_image.png, figH_crossmodal.png, runs/mimo_audio/gallery.json.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs/mimo_audio")
OUT = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "runs/figures_pitch")
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e2", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True})


def fig_f():
    t = pd.read_csv(RUN / "pair_auc.csv")
    rows = [("permuted codes (chance)", "hs3 permuted (chance)", C["magenta"]), ("same museum", "provider", C["yellow"]),
            ("same image cluster, 10 clusters", "image cluster k=10", C["blue"]), ("same image cluster, 20 clusters", "image cluster k=20", C["blue"]),
            ("same image cluster, 40 clusters", "image cluster k=40", C["blue"]), ("same H-S class (level 1)", "hs1", C["orange"]),
            ("same H-S code (level 3)", "hs3", C["orange"]), ("same instrument name", "name", C["aqua"])]
    v = t.set_index("grouping").AUC.copy()
    g = ROOT / "runs/tests/audio/pair_auc_granularity.csv"  # photo groups with as many groups as the label has classes
    if g.exists():
        gr = pd.read_csv(g)
        gr = gr[gr.space == "clap"].set_index("label")
        v["image cluster k=17"] = gr.loc["hs3", "AUC_photo_groups_same_k_same_objects"]
        rows.insert(3, ("same image cluster, 17 clusters (as many as level-3 codes)", "image cluster k=17", C["blue"]))
        rows = [(lab.replace("same H-S code (level 3)", "same H-S code (level 3, 17 codes)").replace("same instrument name", "same instrument name (140 names)"), key, col)
                for lab, key, col in rows]
    fig, ax = plt.subplots(figsize=(8.4 + 2.2 * (len(rows) > 8), 4.2 + 0.4 * (len(rows) - 8)))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, key, col) in zip(y, rows):
        ax.barh(yi, v[key], color=col, height=0.6)
        ax.text(v[key] + 0.008, yi, f"{v[key]:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0.4, 1.0); ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    n = len(pd.read_csv(RUN / "objects.csv"))
    ax.set_xlabel(f"AUC: does sharing the group predict high audio similarity? ({n:,} objects, photo and recording of each)", fontsize=9)
    ax.set_title("Instruments that cluster together in photos sound alike", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(OUT / "figF_pair_auc.png", dpi=170); plt.close(fig)


def fig_g():
    t = pd.read_csv(RUN / "labels.csv")
    cls = t[t.space == "audio"].set_index("label").classes
    names = {"hs1": "H-S level 1", "hs2": "H-S level 2", "hs3": "H-S level 3", "name": "instrument name", "provider": "museum", "region": "region"}
    labels = [(k, f"{v}\n({int(cls[k])} classes)" if k == "hs1" else f"{v}\n({int(cls[k])})") for k, v in names.items()]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    w = 0.36
    for j, (space, col, lab) in enumerate([("audio", C["orange"], "sound clusters (CLAP)"), ("image", C["blue"], "photo clusters (DINOv2)")]):
        s = t[t.space == space].set_index("label")
        vals = [s.AMI_at_classes.get(k, np.nan) for k, _ in labels]
        x = np.arange(len(labels)) + (j - 0.5) * w
        ax.bar(x, vals, width=w * 0.94, color=col, label=lab)
        for xi, vv in zip(x, vals):
            ax.text(xi, vv + 0.012, f"{vv:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels([l for _, l in labels], fontsize=8.5)
    ax.set_ylabel("AMI of clusters vs label, k = number of classes"); ax.set_ylim(0, 1.0)
    ax.set_title("Sound follows the acoustic tree more closely than form does (same objects)", loc="left", fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "figG_audio_vs_image.png", dpi=170); plt.close(fig)


def fig_h_and_gallery():
    obj = pd.read_csv(RUN / "objects.csv", dtype=str).fillna("")
    XA = np.load(RUN / "emb_clap.npy"); XA = XA[~np.isnan(XA).any(1)]
    IMG_TABLE = ROOT / (sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].endswith(".csv") else "runs/mimo_full/sample_full.csv")
    IMG_EMB = ROOT / (sys.argv[4] if len(sys.argv) > 4 and sys.argv[4].endswith(".npy") else "runs/mimo_full/emb/dinov2_b.npy")
    full = pd.read_csv(IMG_TABLE, dtype=str).fillna("")
    row_of = dict(zip(full.id, range(len(full))))
    XI = np.load(IMG_EMB)[[row_of[i] for i in obj.id]]
    assert len(XA) == len(obj) == len(XI)
    ZA = PCA(50, random_state=0).fit_transform(XA); ZI = PCA(50, random_state=0).fit_transform(XI)
    ca = KMeans(10, n_init=10, random_state=0).fit_predict(ZA); ci = KMeans(10, n_init=10, random_state=0).fit_predict(ZI)
    # order both so the diagonal carries the mass
    ct = pd.crosstab(ci, ca)
    order_i = list(ct.max(1).sort_values(ascending=False).index)
    used, order_a = set(), []
    for i in order_i:
        best = [a for a in ct.loc[i].sort_values(ascending=False).index if a not in used]
        if best:
            order_a.append(best[0]); used.add(best[0])
    order_a += [a for a in ct.columns if a not in used]
    M = ct.loc[order_i, order_a].values
    fig, ax = plt.subplots(figsize=(6.8, 6))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            if M[r, c] > 0:
                ax.text(c, r, str(M[r, c]), ha="center", va="center", fontsize=8, color=INK if M[r, c] < M.max() * 0.6 else SURF)
    rec_t = pd.read_csv(ROOT / "data/mimo/records.csv", dtype=str).fillna("")
    title_of = dict(zip(rec_t.id, rec_t.title))
    names = pd.Series([n if n else title_of.get(i, "untitled").split(".")[0][:24].lower() for n, i in zip(obj.name, obj.id)])
    obj["display"] = names.values
    ylab = [f"photo {k}: " + names[ci == i].value_counts().index[0][:18] for k, i in enumerate(order_i)]
    xlab = [f"sound {k}" for k in range(len(order_a))]
    ax.set_yticks(range(len(order_i))); ax.set_yticklabels(ylab, fontsize=8); ax.set_xticks(range(len(order_a))); ax.set_xticklabels(xlab, fontsize=8, rotation=45, ha="right")
    from sklearn.metrics import adjusted_mutual_info_score as ami_score
    ax.grid(False); ax.set_title(f"Photo clusters vs sound clusters, same {len(obj):,} objects (AMI {ami_score(ci, ca):.2f})", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "figH_crossmodal.png", dpi=170); plt.close(fig)

    # gallery data: 20 image clusters, each object with preview image, audio URL, name, H-S class, museum, audio cluster
    ci20 = KMeans(20, n_init=10, random_state=0).fit_predict(ZI); ca20 = KMeans(20, n_init=10, random_state=0).fit_predict(ZA)
    rec = pd.read_csv(ROOT / "data/mimo/records.csv", dtype=str).fillna("")
    prev = dict(zip(rec.id, rec.preview))
    aud = pd.read_csv(ROOT / "data/mimo_audio/audio.csv", dtype=str).fillna("")
    aurl = dict(zip(aud.file, aud.audio_url))
    # audio similarity within vs between image clusters, per cluster
    S = XA @ XA.T
    items, clusters = [], []
    for c in range(20):
        ii = np.where(ci20 == c)[0]
        within = S[np.ix_(ii, ii)][np.triu_indices(len(ii), 1)].mean() if len(ii) > 1 else float("nan")
        others = np.setdiff1d(np.arange(len(obj)), ii)
        between = S[np.ix_(ii, others)].mean()
        maj = names.iloc[ii].value_counts(); mh = obj.hs1.iloc[ii].replace("", "no code").value_counts()
        clusters.append({"cluster": c, "n": int(len(ii)), "majority_name": maj.index[0], "majority_share": round(float(maj.iloc[0] / len(ii)), 2),
                         "majority_hs1": mh.index[0], "within_sim": round(float(within), 3), "between_sim": round(float(between), 3),
                         "audio_clusters": int(len(set(ca20[ii])))})
        # order objects inside a cluster by distance to the image centroid
        cen = ZI[ii].mean(0); d = np.linalg.norm(ZI[ii] - cen, axis=1)
        for j in ii[np.argsort(d)]:
            items.append({"cluster": c, "id": obj.id[j], "name": obj.name[j], "title": title_of.get(obj.id[j], ""), "hs1": obj.hs1[j], "hs3": obj.hs3[j], "museum": obj.provider[j],
                          "image": prev.get(obj.id[j], ""), "audio": aurl.get(obj.file_a[j], "").replace("http://www.mimo-international.com", "https://mimo-international.com"),
                          "audio_cluster": int(ca20[j])})
    clusters.sort(key=lambda x: -x["n"])
    json.dump({"clusters": clusters, "items": items}, open(RUN / "gallery.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(pd.DataFrame(clusters).to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = [a for a in sys.argv[3:] if a in ("f", "g", "h")] or ["f", "g", "h"]
    if "f" in which:
        fig_f(); print("F")
    if "g" in which:
        fig_g(); print("G")
    if "h" in which:
        fig_h_and_gallery(); print("H")
