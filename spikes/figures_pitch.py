"""Pitch figures for the instruments idea, from the saved MIMO results. Static PNGs in runs/figures_pitch.

Same style as figures.py: fixed categorical order blue, orange, aqua, yellow, magenta; thin marks; text in ink.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/figures_pitch"
OUT.mkdir(parents=True, exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e2", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True})
HS1 = {"1": "Idiophone", "2": "Membranophone", "3": "Chordophone", "4": "Aerophone", "5": "Electrophone"}
HSC = {"Aerophone": C["blue"], "Chordophone": C["orange"], "Idiophone": C["aqua"], "Membranophone": C["yellow"], "Electrophone": C["magenta"]}
FAMC = {"woodwind instrument": C["blue"], "brass instrument": C["yellow"], "string instrument": C["orange"], "percussion instrument": C["aqua"],
        "keyboard instrument": C["magenta"]}


def load():
    df = pd.read_csv(ROOT / "runs/mimo_full/sample_full.csv", dtype={"notation": str}).fillna("")
    lab = pd.read_csv(ROOT / "runs/mimo_names/labels.csv", dtype=str).fillna("")
    assert (lab.id.values == df.id.values).all()
    df["name"], df["family"] = lab.name.values, lab.family.values
    df["hs1"] = df.notation.str.replace(".", "", regex=False).str[:1].map(HS1).fillna("")
    X = np.load(ROOT / "runs/mimo_full/emb/dinov2_b.npy")
    return df, X


def fig_a_disagreements(df, Z):
    """Thumbnails from 3 image clusters that cut across the Hornbostel-Sachs classes."""
    m = (df.name != "") & (df.hs1 != "")
    vc = df.name[m].value_counts(); m &= df.name.map(vc) >= 50
    sub = df[m].copy()
    sub["c"] = KMeans(20, n_init=3, random_state=0).fit_predict(Z[sub.index.values])
    picks = [  # cluster, title
        (17, "Long thin bodies: tube zithers, Jew's harps, rattles and tube drums (3 other classes) with flutes and pipes (aerophone)"),
        (13, "Rows of tubes and bars: pan flutes (aerophone) with xylophones and lamellophones (idiophone)"),
        (18, "Necked bodies: electric guitars (electrophone) with lutes and guitars (chordophone)"),
    ]
    ncol = 8
    fig, axes = plt.subplots(3, ncol, figsize=(13, 6.3))
    rng = np.random.default_rng(0)
    for row, (c, title) in enumerate(picks):
        g = sub[sub.c == c]
        dom = g.hs1.value_counts().index[0]
        counts = g.name.value_counts()
        cross = [n for n in counts.index if g[g.name == n].hs1.iloc[0] != dom][:4]
        fill = [n for n in counts.index if n not in cross][: ncol - len(cross)]
        names = cross + fill
        print(f"  cluster {c}: dominant {dom}; showing {names}")
        for k, n in enumerate(names):
            gg = g[g.name == n]
            r = gg.iloc[rng.integers(len(gg))]
            ax = axes[row, k]
            im = Image.open(ROOT / "data/mimo" / r.file).convert("RGB")
            im = ImageOps.pad(im, (200, 200), color=(252, 252, 251))
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_edgecolor(HSC[r.hs1]); s.set_linewidth(3); s.set_visible(True)
            ax.set_title(f"{n}\n{r.hs1.lower()}", fontsize=7.5, color=INK2, pad=3)
        for k in range(len(names), ncol):
            axes[row, k].axis("off")
        axes[row, 0].text(-0.08, 0.5, f"cluster {row + 1}", transform=axes[row, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9, color=INK2)
        fig.text(0.5, 0.985 - row * 0.325, title, ha="center", fontsize=9.5, color=INK)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=v, label=k) for k, v in HSC.items()], loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               title="frame colour: Hornbostel-Sachs class of the object", title_fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.09, hspace=0.9, wspace=0.08)
    fig.savefig(OUT / "figA_clusters_vs_hs.png", dpi=160); plt.close(fig)
    return sub


def fig_b_depth():
    """AMI and probe against the number of classes for the 2 taxonomies and the name labels."""
    d = pd.read_csv(ROOT / "runs/mimo_wd_depth/depth_same_rows_corrected_majority.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    series = [("hs", "Hornbostel-Sachs, levels 1 to 6", C["blue"]), ("wd", "Wikidata tree, levels 1 to 7", C["orange"])]
    for pre, lab, col in series:
        s = d[d.label.str.startswith(pre)].sort_values("classes")
        ax.plot(s.classes, s.AMI, "-o", color=col, lw=2, ms=6, label=lab + ", k-means")
        ax.plot(s.classes, s.probe, "--o", color=col, lw=1.2, ms=6, mfc=SURF, label=lab + ", linear probe")
    n = d[d.label == "name"].iloc[0]
    ax.plot([n.classes], [n.AMI], "o", color=C["aqua"], ms=8, label="instrument name (123 classes), k-means")
    ax.plot([n.classes], [n.probe], "o", color=C["aqua"], ms=8, mfc=SURF, label="instrument name, linear probe")
    ax.annotate("name 0.57", (n.classes, n.AMI), xytext=(-8, -16), textcoords="offset points", fontsize=8.5, color=INK2, ha="right")
    ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 50, 100]); ax.set_xticklabels(["5", "10", "20", "50", "100"])
    ax.set_xlabel("number of classes in the label"); ax.set_ylabel("AMI (filled) or probe accuracy (hollow)"); ax.set_ylim(0, 1.0)
    ax.set_title("Finer labels fit the image clusters better, while the linear ceiling drops: the gap is the project", loc="left", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "figB_depth.png", dpi=170); plt.close(fig)


def fig_c_reference():
    """How good is AMI 0.5? Reference partitions scored against H-S level 3 on the same rows."""
    rows = [("random labels", 0.0, C["magenta"]),
            ("k-means on DINOv2, no training", 0.488, C["blue"]),
            ("instrument name (perfect fine labels)", 0.657, C["aqua"]),
            ("Wikidata taxonomy (second expert tree)", 0.875, C["orange"]),
            ("linear probe, supervised ceiling (accuracy)", 0.897, C["yellow"])]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, col) in zip(y, rows):
        ax.barh(yi, v, color=col, height=0.55)
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 1.0); ax.set_xlabel("agreement with Hornbostel-Sachs level 3 (16 classes), 25,790 objects")
    ax.set_title("Where 0.49 sits: between chance and a perfect name-level labeling", loc="left", fontsize=10.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(OUT / "figC_reference.png", dpi=170); plt.close(fig)


def fig_d_umap(df, Z):
    """UMAP of the common rows, coloured by H-S class and by Western family."""
    import umap
    m = (df.family != "") & (df.hs1 != "") & (df.name != "")
    idx = df.index[m].values
    rng = np.random.default_rng(0)
    idx = rng.choice(idx, size=min(14000, len(idx)), replace=False)
    emb = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=0).fit_transform(Z[idx])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, col, cmap, title in [(axes[0], "hs1", HSC, "coloured by Hornbostel-Sachs class (how the sound is made)"),
                                 (axes[1], "family", FAMC, "coloured by Western family (how it is played)")]:
        lab = df.loc[idx, col].values
        for k, cc in cmap.items():
            sel = lab == k
            ax.scatter(emb[sel, 0], emb[sel, 1], s=3, color=cc, alpha=0.55, linewidths=0, label=f"{k.replace(' instrument', '')} ({sel.sum():,})")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title(title, fontsize=10, loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8.5, frameon=False, markerscale=4)
    fig.suptitle("Same 14,000 MIMO photos, DINOv2 features, UMAP: the islands are instrument families; the top acoustic class merges brass with woodwind",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(OUT / "figD_umap.png", dpi=160); plt.close(fig)


def fig_e_museums():
    """Inside each museum (one photographer): H-S class recoverable, region not."""
    h = pd.read_csv(ROOT / "runs/mimo_full_origin/hs1_within_museum.csv")
    r = pd.read_csv(ROOT / "runs/mimo_full_origin/region_within_museum.csv")
    h["short"] = h.provider.str.replace("Staatliche Museen zu Berlin", "Berlin").str.replace("Museum of Musical Instruments, Leipzig University", "Leipzig University")
    h = h.sort_values("best_AMI", ascending=True)
    r = r.set_index("provider")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    y = np.arange(len(h))
    ax.hlines(y, 0, h.best_AMI, color=GRID, lw=1)
    ax.plot(h.best_AMI, y, "o", color=C["blue"], ms=8, label=f"Hornbostel-Sachs class ({h.classes.min()} to {h.classes.max()} classes)")
    rr = h.provider.map(r.best_AMI)
    ax.plot(rr, y, "o", color=C["orange"], ms=8, label=f"region of origin ({r.classes.min()} to {r.classes.max()} regions)")
    for yi, (a, b, n) in enumerate(zip(h.best_AMI, rr, h.n)):
        ax.text(a + 0.012, yi, f"{a:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([f"{s[:42]} ({n:,})" for s, n in zip(h.short, h.n)], fontsize=8.5)
    ax.set_xlim(0, 0.72); ax.set_xlabel("AMI of image clusters vs label, inside one museum")
    ax.set_title("Inside one museum (one photographer): acoustic class is in the object, origin mostly is not", loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(OUT / "figE_museums.png", dpi=170); plt.close(fig)


def main():
    df, X = load()
    Z = PCA(50, random_state=0).fit_transform(X)
    which = sys.argv[1:] or ["a", "b", "c", "d", "e"]
    if "a" in which:
        fig_a_disagreements(df, Z); print("A", flush=True)
    if "b" in which:
        fig_b_depth(); print("B", flush=True)
    if "c" in which:
        fig_c_reference(); print("C", flush=True)
    if "e" in which:
        fig_e_museums(); print("E", flush=True)
    if "d" in which:
        fig_d_umap(df, Z); print("D", flush=True)


if __name__ == "__main__":
    main()
