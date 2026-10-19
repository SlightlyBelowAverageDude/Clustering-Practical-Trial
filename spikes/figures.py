"""Figures for the instrument findings. Static PNGs via matplotlib.

Palette: fixed categorical order blue, orange, aqua, yellow, magenta (validated with the dataviz skill's
script). Thin marks, light grid, direct labels where they help, text in ink colours, never series colours.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/figures"
OUT.mkdir(parents=True, exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e2", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True})


def fig1_signal_by_dataset():
    """Which ordering do the clusters follow? Best AMI per label, 4 instrument datasets, DINOv2."""
    rows = [  # dataset, label, AMI
        ("Met pilot\n846, object-only", "acoustic class (4)", 0.38), ("Met pilot\n846, object-only", "acoustic level 2", 0.55),
        ("Met pilot\n846, object-only", "region", 0.29), ("Met pilot\n846, object-only", "era", 0.12),
        ("Met, all PD\n2,444", "acoustic class (4)", 0.44), ("Met, all PD\n2,444", "acoustic level 2", 0.51), ("Met, all PD\n2,444", "acoustic level 3", 0.56),
        ("Met, all PD\n2,444", "region", 0.24), ("Met, all PD\n2,444", "era", 0.12),
        ("MIMO, 173 museums\n4,000", "acoustic class (4)", 0.46), ("MIMO, 173 museums\n4,000", "acoustic level 2", 0.52), ("MIMO, 173 museums\n4,000", "acoustic level 3", 0.57),
        ("MIMO, 173 museums\n4,000", "region", 0.17), ("MIMO, 173 museums\n4,000", "museum", 0.21), ("MIMO, 173 museums\n4,000", "era", 0.08),
        ("Berlin only\n1,674, same photographer", "acoustic class (4)", 0.50), ("Berlin only\n1,674, same photographer", "acoustic level 2", 0.47),
        ("Berlin only\n1,674, same photographer", "region", 0.27), ("Berlin only\n1,674, same photographer", "country", 0.34),
    ]
    df = pd.DataFrame(rows, columns=["dataset", "label", "AMI"])
    labels = ["acoustic class (4)", "acoustic level 2", "acoustic level 3", "region", "country", "museum", "era"]
    colors = {"acoustic class (4)": C["blue"], "acoustic level 2": C["blue"], "acoustic level 3": C["blue"], "region": C["orange"],
              "country": C["orange"], "museum": C["yellow"], "era": C["magenta"]}
    hatch = {"acoustic level 2": "///", "acoustic level 3": "xxx", "country": "///"}
    datasets = list(dict.fromkeys(df.dataset))
    fig, ax = plt.subplots(figsize=(11, 5.4))
    w = 0.11
    for i, ds in enumerate(datasets):
        sub = df[df.dataset == ds].set_index("label").reindex(labels).dropna()
        offs = np.arange(len(sub)) - (len(sub) - 1) / 2
        for o, (lab, v) in zip(offs, sub.AMI.items()):
            ax.bar(i + o * w, v, width=w * 0.92, color=colors[lab], hatch=hatch.get(lab, ""), edgecolor=SURF, linewidth=0.8)
            ax.text(i + o * w, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2, rotation=90)
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylabel("best AMI of clusters vs label (DINOv2)"); ax.set_ylim(0, 0.68)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=C["blue"], label="acoustic class, Hornbostel-Sachs level 1"), Patch(facecolor=C["blue"], hatch="///", label="H-S level 2"),
               Patch(facecolor=C["blue"], hatch="xxx", label="H-S level 3"), Patch(facecolor=C["orange"], label="region of origin"),
               Patch(facecolor=C["orange"], hatch="///", label="country of origin"), Patch(facecolor=C["yellow"], label="museum"), Patch(facecolor=C["magenta"], label="era")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8, frameon=False, ncol=4)
    ax.set_title("Instrument photos cluster by how the sound is made first; origin is a weaker second signal", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig1_signal_by_dataset.png", dpi=170); plt.close(fig)


def fig2_tree_depth():
    """Agreement with the H-S tree rises with depth; one dendrogram cut per level."""
    data = {"Met, all PD (2,444)": [0.432, 0.510, 0.545], "MIMO (4,000), DINOv2-S": [0.435, 0.463, 0.486], "MIMO (4,000), DINOv2-B": [0.479, 0.556, 0.534]}
    fig, ax = plt.subplots(figsize=(6.2, 4))
    x = [1, 2, 3]
    for (name, ys), col, mk in zip(data.items(), [C["blue"], C["aqua"], C["orange"]], ["o", "s", "^"]):
        ax.plot(x, ys, color=col, linewidth=2, marker=mk, markersize=7, label=name)
        ax.text(3.08, ys[-1], f"{ys[-1]:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(["level 1\n4 or 5 classes", "level 2\n10 to 18 classes", "level 3\n14 to 31 classes"])
    ax.set_ylabel("AMI, dendrogram cut at the level's class count"); ax.set_ylim(0.3, 0.65); ax.set_xlim(0.8, 3.4)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("The cluster tree mirrors the Hornbostel-Sachs tree down 3 levels", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig2_tree_depth.png", dpi=170); plt.close(fig)


def fig3_encoders():
    """Encoder comparison with bootstrap intervals, object-only crops, Met pilot."""
    s = pd.read_csv(ROOT / "runs/instruments_encoders/scores.csv")
    enc_names = {"clip_b32": "CLIP B/32", "clip_l14": "CLIP L/14", "siglip_b16": "SigLIP B/16", "dinov2_s": "DINOv2-S", "dinov2_b": "DINOv2-B"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, lab, title, col in zip(axes, ["hs2", "region"], ["H-S level 2 (18 classes)", "region of origin (28 classes)"], [C["blue"], C["orange"]]):
        sub = s[s.label == lab].set_index("encoder").reindex(list(enc_names))
        m = sub.AMI_k_classes_boot.str.extract(r"([\d.]+) \[([\d.]+), ([\d.]+)\]").astype(float)
        y = np.arange(len(sub))
        ax.barh(y, m[0], color=col, height=0.55, edgecolor=SURF)
        ax.errorbar(m[0], y, xerr=[m[0] - m[1], m[2] - m[0]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
        ax.set_yticks(y); ax.set_yticklabels([enc_names[e] for e in sub.index]); ax.invert_yaxis()
        ax.set_xlim(0, 0.8); ax.set_xlabel("AMI at k = classes, 95 percent bootstrap"); ax.set_title(title, loc="left", fontsize=10)
        for yi, (v, hi, p) in enumerate(zip(m[0], m[2], sub.probe_acc)):
            ax.text(hi + 0.012, yi, f"{v:.2f}   probe {p:.2f}", va="center", fontsize=8, color=INK2)
    fig.suptitle("Self-supervised encoders see the acoustic tree best; origin sits at the same level for all", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig3_encoders.png", dpi=170); plt.close(fig)


def fig4_probe_vs_ami():
    """Origin is in the embedding (probe) but clustering captures it only partly (AMI)."""
    rows = [("Met pilot, region", 0.32, 0.22), ("Met all PD, region", 0.34, 0.24), ("MIMO, region", 0.82, 0.17), ("Berlin, region", 0.76, 0.27),
            ("Berlin, country", 0.55, 0.34), ("Brussels, region", 0.72, 0.15), ("Stockholm, region", 0.76, 0.17)]
    maj = [0.18, 0.14, 0.64, 0.30, 0.19, 0.28, 0.50]
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    y = np.arange(len(rows))
    ax.barh(y + 0.18, [r[2] for r in rows], height=0.34, color=C["orange"], label="AMI of clusters vs origin (what clustering finds)")
    ax.barh(y - 0.18, [r[1] for r in rows], height=0.34, color=C["blue"], label="balanced probe accuracy (what the embedding holds)")
    ax.scatter(maj, y - 0.18, marker="|", s=160, color=INK, zorder=3, label="majority baseline for the probe")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("score")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2)
    ax.set_title("Origin is in the embedding (probe) but clustering captures it only partly (AMI)", loc="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / "fig4_probe_vs_ami.png", dpi=170); plt.close(fig)


def fig5_berlin_confusion():
    """What AMI 0.30 looks like: Berlin, 3 regions, 3 clusters."""
    ct = pd.DataFrame([[464, 129, 399], [4, 341, 14], [32, 30, 87]], index=["cluster 0", "cluster 1", "cluster 2"], columns=["Africa", "Americas", "Asia"])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    im = ax.imshow(ct.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#2a78d6", "#0d366b"]), aspect="auto")
    for i in range(3):
        for j in range(3):
            v = ct.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11, color="white" if v > 250 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(ct.columns); ax.set_yticks(range(3)); ax.set_yticklabels(ct.index); ax.grid(False)
    ax.set_title("Berlin, 1,500 instruments, 3 clusters vs region\nAMI 0.30, purity 0.59: the Americas split off,\nAfrica and Asia share a cluster", loc="left", fontsize=9.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="instruments")
    fig.tight_layout(); fig.savefig(OUT / "fig5_berlin_confusion.png", dpi=170); plt.close(fig)


def fig6_umap():
    """The embedding space of the MIMO sample, coloured by acoustic class and by region."""
    import umap
    df = pd.read_csv(ROOT / "data/mimo/sample.csv", dtype={"hs1": str}).fillna("")
    HS1 = {"1": "Idiophone", "2": "Membranophone", "3": "Chordophone", "4": "Aerophone", "5": "Electrophone"}
    df["hs1"] = df.hs1.map(HS1).fillna("")
    X = np.load(ROOT / "runs/mimo/emb/dinov2_b.npy")
    E = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    order1 = ["Aerophone", "Chordophone", "Idiophone", "Membranophone", "Electrophone"]
    cols = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"]]
    for ax, col, order, title in ((axes[0], "hs1", order1, "coloured by acoustic class (Hornbostel-Sachs level 1)"),
                                  (axes[1], "region", ["Europe", "Africa", "Asia", "Americas"], "coloured by region of origin (grey: no place record)")):
        ax.scatter(E[:, 0], E[:, 1], s=4, color="#c9c8c3", linewidths=0, label="_")
        for name, c in zip(order, cols):
            m = (df[col] == name).values
            if m.any():
                ax.scatter(E[m, 0], E[m, 1], s=5, color=c, linewidths=0, label=f"{name} ({m.sum()})", alpha=0.85)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title(title, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8, markerscale=3, loc="lower left")
    fig.suptitle("MIMO, 4,000 instruments from 173 museums, DINOv2-B embeddings, UMAP", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig6_umap.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    for w in which:
        globals()[[n for n in globals() if n.startswith(f"fig{w}_")][0]]()
        print("fig", w, "done", flush=True)
