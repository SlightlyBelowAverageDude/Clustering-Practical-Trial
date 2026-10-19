"""Slide 2 picture: photos -> vision model -> map with blind groups -> compared with the labels.

Uses the 12,000-object subset from the method comparison; the map is UMAP, coloured by 6 blind k-means groups.
Output: runs/figures_pitch_v2/figP_pipeline.png, aspect 2.165:1 to replace the picture on slide 2.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import umap  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/figures_pitch_v2"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5cc9"]
INK, INK2, LINE, SURF = "#1d1b17", "#52514e", "#c9c5bb", "#ffffff"
PHOTOS = ["violin", "clarinet", "bugle", "gong", "frame drum", "square piano"]


def main():
    sub = pd.read_csv(ROOT / "runs/method_compare/subset.csv", dtype=str).fillna("")
    full = pd.read_csv(ROOT / "runs/mimo_full/sample_full.csv", dtype=str).fillna("")
    row = dict(zip(full.id, range(len(full))))
    X = np.load(ROOT / "runs/mimo_full/emb/dinov2_b.npy")[[row[i] for i in sub.id]]
    Z = PCA(50, random_state=0).fit_transform(X)
    E = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=0).fit_transform(Z)
    km = KMeans(6, n_init=10, random_state=0).fit_predict(Z)

    fig = plt.figure(figsize=(13, 6), facecolor=SURF)
    # column 1: photos (2 x 3)
    for i, n in enumerate(PHOTOS):
        g = sub[sub.name == n]
        if len(g) == 0:
            continue
        r = g.iloc[0]
        ax = fig.add_axes([0.03 + (i % 2) * 0.095, 0.56 - (i // 2) * 0.235, 0.085, 0.2])
        im = ImageOps.pad(Image.open(ROOT / "data/mimo" / r.file).convert("RGB"), (200, 200), color=(255, 255, 255))
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(LINE); s.set_linewidth(1)
    fig.text(0.125, 0.86, "63,000 photos", ha="center", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.125, 0.80, "with catalogue cards", ha="center", fontsize=11, color=INK2)

    # column 2: the map
    ax = fig.add_axes([0.30, 0.08, 0.40, 0.68])
    for c in range(6):
        sel = km == c
        ax.scatter(E[sel, 0], E[sel, 1], s=3, color=PAL[c], alpha=0.6, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(np.percentile(E[:, 0], 0.3), np.percentile(E[:, 0], 99.7)); ax.set_ylim(np.percentile(E[:, 1], 0.3), np.percentile(E[:, 1], 99.7))
    for s in ax.spines.values():
        s.set_edgecolor(LINE)
    fig.text(0.50, 0.86, "clustering", ha="center", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.50, 0.80, "each dot is a photo, each colour a group found without labels", ha="center", fontsize=11, color=INK2)

    # column 3: what the groups are compared with
    items = [("expert code", "Hornbostel-Sachs, 6 levels"), ("instrument names", "123 names via Wikipedia"), ("origin", "country and region"),
             ("recordings", "1,000 objects, same instrument")]
    for i, (t, s) in enumerate(items):
        y = 0.63 - i * 0.165
        box = FancyBboxPatch((0.765, y), 0.215, 0.12, boxstyle="round,pad=0.008,rounding_size=0.012", transform=fig.transFigure,
                             facecolor="#f3f1ec", edgecolor=LINE, linewidth=1)
        fig.add_artist(box)
        fig.text(0.78, y + 0.075, t, fontsize=13, fontweight="bold", color=INK, va="center")
        fig.text(0.78, y + 0.035, s, fontsize=10.5, color=INK2, va="center")
    fig.text(0.8725, 0.86, "compared with", ha="center", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.8725, 0.80, "do the groups match any of these?", ha="center", fontsize=11, color=INK2)

    # arrows
    for x0, x1 in ((0.225, 0.29), (0.71, 0.755)):
        fig.add_artist(FancyArrowPatch((x0, 0.45), (x1, 0.45), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22, color=INK2, linewidth=2))
    fig.savefig(OUT / "figP_pipeline.png", dpi=170, facecolor=SURF); plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
