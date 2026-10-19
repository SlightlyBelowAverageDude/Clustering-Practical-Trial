"""Slide-ready version of the 'blind groups cross the expert classes' figure: 3 rows, 6 photos each.

Same clustering as figures_pitch.fig_a (k-means, 20 groups, MIMO objects with a name of 50+ and an H-S code).
Output: runs/figures_pitch_v2/figA2_rows.png (16:9).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from pathlib import Path  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/figures_pitch_v2"
HS1 = {"1": "Idiophone", "2": "Membranophone", "3": "Chordophone", "4": "Aerophone", "5": "Electrophone"}
HSC = {"Aerophone": "#2a78d6", "Chordophone": "#eb6834", "Idiophone": "#1baf7a", "Membranophone": "#eda100", "Electrophone": "#e87ba4"}
INK, INK2, SURF = "#0b0b0b", "#52514e", "#ffffff"
ROWS = [  # cluster, title, subtitle, names in display order
    (13, "Rows of tubes and bars", "pan flutes (air)\ngrouped with xylophones\nand lamellophones (struck body)",
     ["pan flute", "syrinx", "mouth organ", "xylophone", "lamellophone", "ratchet"]),
    (18, "A neck and a body", "electric guitars (electric)\ngrouped with guitars,\nlutes and banjos (strings)",
     ["electric guitar", "guitar", "banjo", "cittern", "theorbo", "viola"]),
    (17, "Long and thin", "tube zithers (strings) and\nJew's harps (struck body)\ngrouped with flutes and pipes (air)",
     ["tube zither", "Jew's harp", "flute", "duct flute", "end-blown flute", "trumpet"]),
]


def main():
    df = pd.read_csv(ROOT / "runs/mimo_full/sample_full.csv", dtype={"notation": str}).fillna("")
    lab = pd.read_csv(ROOT / "runs/mimo_names/labels.csv", dtype=str).fillna("")
    df["name"] = lab.name.values
    df["hs1"] = df.notation.str.replace(".", "", regex=False).str[:1].map(HS1).fillna("")
    X = np.load(ROOT / "runs/mimo_full/emb/dinov2_b.npy")
    Z = PCA(50, random_state=0).fit_transform(X)
    m = (df.name != "") & (df.hs1 != "")
    vc = df.name[m].value_counts(); m &= df.name.map(vc) >= 50
    sub = df[m].copy()
    sub["c"] = KMeans(20, n_init=3, random_state=0).fit_predict(Z[sub.index.values])
    cen = {c: Z[sub.index.values][sub.c.values == c].mean(0) for c in range(20)}

    fig = plt.figure(figsize=(16, 9), facecolor=SURF)
    ncol = 6
    gs = fig.add_gridspec(3, ncol, left=0.215, right=0.985, top=0.93, bottom=0.11, hspace=0.55, wspace=0.06)
    for r, (c, title, sub_t, names) in enumerate(ROWS):
        g = sub[sub.c == c]
        fig.text(0.015, 0.93 - r * 0.2735 - 0.09, title, fontsize=16, fontweight="bold", color=INK, va="center")
        fig.text(0.015, 0.93 - r * 0.2735 - 0.09 - 0.075, sub_t, fontsize=11, color=INK2, va="center", linespacing=1.4)
        for k, n in enumerate(names):
            gg = g[g.name == n]
            ax = fig.add_subplot(gs[r, k])
            if len(gg) == 0:
                ax.axis("off"); continue
            d = np.linalg.norm(Z[gg.index.values] - cen[c], axis=1)  # the most typical object of that name in this group
            row = gg.iloc[int(np.argsort(d)[min(1, len(d) - 1)])]
            im = Image.open(ROOT / "data/mimo" / row.file).convert("RGB")
            im = ImageOps.pad(im, (260, 260), color=(255, 255, 255))
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(HSC[row.hs1]); s.set_linewidth(5); s.set_visible(True)
            ax.set_title(n, fontsize=13, color=INK, pad=6)
    fig.text(0.5, 0.975, "Three groups the computer found on its own. Frame colour: the expert class (how the sound is made).",
             ha="center", fontsize=15, color=INK)
    fig.legend(handles=[Patch(facecolor=v, label=k) for k, v in HSC.items() if k != "Membranophone"], loc="lower center", ncol=4, frameon=False, fontsize=13,
               bbox_to_anchor=(0.5, 0.015), handlelength=1.4, columnspacing=2.2)
    fig.savefig(OUT / "figA2_rows.png", dpi=160, facecolor=SURF); plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
