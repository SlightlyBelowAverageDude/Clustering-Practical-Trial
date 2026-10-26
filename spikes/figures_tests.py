"""Figures for the test round in runs/tests (probe gap, tree against tree, robustness, audio). Every number is read from
the tables in runs/tests. Same palette and style as spikes/figures_clean.py. Output: runs/figures_tests/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "runs/tests"
OUT = ROOT / "runs/figures_tests"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#9a978f", "#e6e5e2", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})
LEVEL = {f"hs{d}": f"level {d}" for d in range(1, 7)}


def title(fig, head, sub):
    fig.suptitle(head, x=0.01, y=0.995, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.95, sub, ha="left", va="top", fontsize=9.5, color=INK2)


def fig_gap():
    g = T / "gap"
    base = pd.read_csv(g / "base.csv").set_index("method")
    obj = pd.read_csv(g / "objonly.csv").set_index("method")
    scan = pd.read_csv(g / "scan.csv").set_index("label")
    temi = pd.read_csv(g / "temi_official_settings.csv").set_index("label")
    text = pd.read_csv(g / "text.csv")
    tx = {v: text[(text.variant == v) & (text.method == "k-means")].iloc[0] for v in text.variant.unique() if (text[text.variant == v].method == "k-means").any()}
    labels = [("hs3", "Hornbostel-Sachs level 3, 13 classes"), ("family", "Western family, 6 classes"), ("name", "instrument name, 129 classes")]
    rows = [
        ("average-link, cosine", lambda c: base.loc["average-link cosine", c]),
        ("k-means, background removed", lambda c: obj.loc["k-means", c]),
        ("average-link, background removed", lambda c: obj.loc["average-link cosine", c]),
        ("SCAN, after self-labelling", lambda c: scan.loc[c, "AMI_after_self_label"]),
        ("TEMI, official settings", lambda c: temi.loc[c, "TEMI_AMI_selected_head"]),
        ("k-means fit like TEMI, all photos", lambda c: temi.loc[c, "kmeans_fit_on_all_photos_AMI"]),
        ("k-means, SigLIP image embedding", lambda c: tx["SigLIP B/16 image embedding"][c]),
        ("k-means, text profile, 945 names", lambda c: tx["text profile (similarity to 945 names)"][c]),
        ("k-means, DINOv2 plus text profile", lambda c: tx["DINOv2-B + text profile"][c]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (c, lab) in zip(axes, labels):
        km, pr = base.loc["k-means", c], base.loc["probe", c]
        ax.axvspan(km, pr, color=GRID, alpha=0.55, linewidth=0)
        ax.axvline(km, color=INK2, linewidth=1.2, linestyle="--")
        ax.axvline(pr, color=INK, linewidth=1.6)
        ax.text(km - 0.008, len(rows) - 0.1, f"k-means {km:.2f}", ha="right", va="center", fontsize=8.5, color=INK2)
        ax.text(pr + 0.008, len(rows) - 0.1, f"probe {pr:.2f}", ha="left", va="center", fontsize=8.5, color=INK)
        vals = [f(c) for _, f in rows]
        ax.scatter(vals, y, s=64, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
        for v, yy in zip(vals, y):
            left = v < km
            ax.text(v - 0.014 if left else v + 0.014, yy, f"{v:.2f}", va="center", ha="right" if left else "left", fontsize=8.5, color=INK2)
        ax.set_xlim(0.15, 0.92)
        ax.set_ylim(-0.7, len(rows) + 0.4)
        ax.set_title(lab, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("AMI against the label (0 = chance, 1 = same sorting)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r for r, _ in rows])
    title(fig, "No blind method closes the gap to the probe",
          "12,000 objects, padded DINOv2-B unless stated, k = number of classes. Shaded: the gap between k-means and the supervised probe. "
          "Text variants use a name list that contains the label names.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.subplots_adjust(top=0.85)
    fig.savefig(OUT / "figT1_probe_gap.png", dpi=170)
    plt.close(fig)


def fig_tree():
    d = T / "tree"
    pur = pd.read_csv(d / "purity.csv")
    rank = pd.read_csv(d / "rank.csv")
    st = pd.read_csv(d / "stability.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw={"width_ratios": [1.1, 1.15, 1.0]})
    ax = axes[0]
    x = np.arange(1, 7)
    for tree, col, ls, name in (("average-link cosine", BLUE, "-", "average-link tree"), ("Ward", ORANGE, "-", "Ward tree")):
        p = pur[(pur.tree == tree) & pur.label.isin(LEVEL)].set_index("label").loc[list(LEVEL)]
        ax.plot(x, p.purity, color=col, linewidth=2, marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=2, label=name)
        ax.text(6.12, p.purity.iloc[-1], f"{name} {p.purity.iloc[-1]:.2f}", va="center", fontsize=8.5, color=INK2)
    sh = pur[(pur.tree == "Ward") & pur.label.isin(LEVEL)].set_index("label").loc[list(LEVEL)]
    ax.plot(x, sh.purity_shuffled_labels, color=MUTED, linewidth=1.6, linestyle="--", marker="o", markersize=5, label="shuffled labels (chance)")
    ax.text(6.12, sh.purity_shuffled_labels.iloc[-1], f"chance {sh.purity_shuffled_labels.iloc[-1]:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n({c})" for i, c in zip(x, sh.classes)])
    ax.set_xlabel("Hornbostel-Sachs level (classes)")
    ax.set_ylabel("dendrogram purity")
    ax.set_ylim(0, 0.85)
    ax.set_xlim(0.7, 7.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("The blind trees keep expert classes together", loc="left", fontsize=10)

    ax = axes[1]
    order = ["cosine distance, no tree", "average-link cosine tree height", "Ward tree height", "different instrument name (0 or 1), reference"]
    names = ["embedding distance, no tree", "average-link tree", "Ward tree", "reference: name differs or not"]
    r = rank.set_index("distance").loc[order]
    yy = np.arange(len(order))[::-1]
    cols = [BLUE, BLUE, BLUE, MUTED]
    ax.barh(yy, r.spearman_vs_HS_distance, color=cols, height=0.55)
    for v, y0 in zip(r.spearman_vs_HS_distance, yy):
        ax.text(v + 0.01, y0, f"{v:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("Spearman correlation with the Hornbostel-Sachs distance")
    ax.grid(axis="y", visible=False)
    ax.text(0.59, -0.55, f"permuted codes give {abs(rank.spearman_vs_permuted_codes).max():.2f} for every row", ha="right", fontsize=8.5, color=INK2)
    ax.set_ylim(-0.75, len(order) - 0.4)
    ax.set_title("Building the tree loses part of the order", loc="left", fontsize=10)

    ax = axes[2]
    lv = list(st.level)
    x = np.arange(1, len(lv) + 1)
    for pre, col, name, dx in (("Ward_cut_agreement", ORANGE, "Ward cuts", -0.08), ("kmeans_agreement", AQUA, "k-means", 0.08)):
        m, lo = st[f"{pre}_mean"], st[f"{pre}_min"]
        ax.vlines(x + dx, lo, m, color=col, linewidth=2)
        ax.plot(x + dx, m, color=col, linewidth=0, marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=2, label=name)
        ax.text(len(lv) + 0.3, m.iloc[-1], f"{name} {m.iloc[-1]:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_xlabel("level (k = " + ", ".join(str(k) for k in st.k) + ")")
    ax.set_ylabel("AMI between 2 subsamples")
    ax.set_ylim(0.6, 1.02)
    ax.set_xlim(0.6, len(lv) + 1.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 0.08))
    ax.set_title("Tree cuts vary more than k-means", loc="left", fontsize=10)
    title(fig, "A blind tree against the Hornbostel-Sachs tree",
          "30,050 objects with a code of 6 or more digits. Purity: share of an object's class under the point where the tree joins it to a classmate." + "\n"
          "Stability: 10 subsamples of 80 percent; dot = mean over 45 pairs of subsamples, line down to the lowest pair.")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.subplots_adjust(top=0.8)
    fig.savefig(OUT / "figT2_tree.png", dpi=170)
    plt.close(fig)


def fig_robust():
    d = T / "robust"
    sub = pd.read_csv(d / "subsample.csv")
    loo = pd.read_csv(d / "leave_museum_out.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})
    ax = axes[0]
    names = [LEVEL.get(l, l) for l in sub.label]
    yy = np.arange(len(sub))[::-1]
    ax.hlines(yy, sub["p2.5"], sub["p97.5"], color=BLUE, linewidth=5, alpha=0.55, label="95 percent range, 20 subsamples of 80 percent")
    ax.scatter(sub.full_set, yy, s=40, color=INK, zorder=3, label="full set")
    for v, y0 in zip(sub.full_set, yy):
        ax.text(v + 0.02, y0, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels(names)
    ax.set_xlim(0.3, 0.7)
    ax.set_xlabel("k-means AMI")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="y", visible=False)
    ax.set_title("Scores move by about 0.02 between subsamples", loc="left", fontsize=10)

    ax = axes[1]
    labs = [("hs1", "level 1"), ("hs3", "level 3"), ("family", "family"), ("name", "name")]
    for i, (c, n) in enumerate(labs):
        v = loo[f"kmeans_AMI_without_{c}"].values
        full = sub.set_index("label").loc[c, "full_set"]
        ax.scatter(np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), v, s=30, color=BLUE, edgecolor=SURF, linewidth=1, zorder=3)
        ax.hlines(full, i - 0.3, i + 0.3, color=INK, linewidth=2, zorder=4)
        ax.text(i + 0.33, full, f"all {full:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels([n for _, n in labs])
    ax.set_xlim(-0.5, len(labs) - 0.1)
    ax.set_ylim(0.3, 0.65)
    ax.set_ylabel("k-means AMI")
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", linewidth=0, label="one museum left out (9 museums)"),
                       Line2D([], [], color=INK, linewidth=2, label="all museums")], frameon=False, fontsize=8, loc="center right")
    ax.set_title("No single museum carries the result", loc="left", fontsize=10)

    ax = axes[2]
    short = {"Musical instrument museum": "Brussels", "Ethnological Museum, Staatliche Museen zu Berlin": "Berlin", "Philharmonie de Paris": "Paris",
             "Swedish Museum of Performing Arts": "Stockholm", "University of Edinburgh": "Edinburgh", "Museum of Musical Instruments, Leipzig University": "Leipzig",
             "Germanisches National Museum": "Nuremberg", "The Bate Collection": "Oxford, Bate", "Royal Museum for Central Africa": "Tervuren"}
    p = loo.assign(drop=loo.probe_acc_hs3_pooled_folds - loo.probe_acc_hs3_trained_without_museum).sort_values("drop")
    yy = np.arange(len(p))
    ax.hlines(yy, p.probe_acc_hs3_trained_without_museum, p.probe_acc_hs3_pooled_folds, color=GRID, linewidth=3)
    ax.scatter(p.probe_acc_hs3_pooled_folds, yy, s=50, color=MUTED, zorder=3, label="museum's photos in training")
    ax.scatter(p.probe_acc_hs3_trained_without_museum, yy, s=60, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=4, label="museum never seen in training")
    for y0, (a, b) in enumerate(zip(p.probe_acc_hs3_trained_without_museum, p.probe_acc_hs3_pooled_folds)):
        ax.text(min(a, b) - 0.012, y0, f"{a:.2f}", va="center", ha="right", fontsize=8.5, color=INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels([short.get(m, m) for m in p.museum_left_out])
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("probe accuracy, Hornbostel-Sachs level 3")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="y", visible=False)
    ax.set_title("The probe transfers to unseen Western museums", loc="left", fontsize=10)
    title(fig, "The main numbers are stable",
          "27,348 objects with code, family and name; classes with 30 or more objects. Museums with 900 or more objects are left out one at a time.")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.subplots_adjust(top=0.84)
    fig.savefig(OUT / "figT3_robustness.png", dpi=170)
    plt.close(fig)


def fig_audio():
    d = T / "audio"
    gr = pd.read_csv(d / "pair_auc_granularity.csv")
    mg = pd.read_csv(d / "matched_groups.csv").set_index("space")
    spaces = [("clap", "CLAP\ntrained with text that names instruments"), ("mfcc", "timbre statistics\nfrom the spectrum, no training"),
              ("hubert", "HuBERT\nself-supervised on unlabelled speech")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.9), gridspec_kw={"width_ratios": [1, 1, 1, 0.85]})
    for ax, (s, name) in zip(axes[:3], spaces):
        g = gr[gr.space == s]
        x = np.arange(len(g))
        ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
        ax.plot(x, g.AUC_label, color=ORANGE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=2, label="expert code or name")
        ax.plot(x, g.AUC_photo_groups_same_k_same_objects, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=2,
                label="photo groups, same number of groups")
        for xi, (a, b) in enumerate(zip(g.AUC_label, g.AUC_photo_groups_same_k_same_objects)):
            hi, lo = (a, b) if a >= b else (b, a)
            ax.text(xi, hi + 0.015, f"{hi:.2f}", ha="center", fontsize=8, color=INK2)
            ax.text(xi, lo - 0.035, f"{lo:.2f}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{LEVEL.get(l, l)}\n{k} groups" for l, k in zip(g.label, g.classes)], fontsize=8.5)
        ax.set_ylim(0.45, 1.02)
        ax.set_title(name, loc="left", fontsize=10)
        ax.text(len(g) - 0.6, 0.505, "chance", fontsize=8, color=INK2, ha="right")
    axes[0].set_ylabel("pair AUC: same group sounds more alike")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    ax = axes[3]
    order = ["clap", "mfcc", "hubert"]
    vals = [mg.loc[s, "objects_on_matched_pairs"] * 100 for s in order]
    x = np.arange(len(order))
    ax.bar(x, vals, color=BLUE, width=0.55)
    perm_mean, perm_max = mg.permutation_mean.mean() * 100, mg.permutation_max.max() * 100
    ax.bar(len(order), perm_mean, color=MUTED, width=0.55)
    ax.vlines(len(order), perm_mean, perm_max, color=INK2, linewidth=1.5)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=9, color=INK2)
    ax.text(len(order), perm_max + 1.5, f"{perm_mean:.0f}, max {perm_max:.0f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(range(len(order) + 1))
    ax.set_xticklabels(["CLAP", "timbre\nstatistics", "HuBERT", "random\nmatching"], fontsize=8.5)
    ax.set_ylim(0, 75)
    ax.set_ylabel("percent of objects")
    ax.grid(axis="x", visible=False)
    ax.set_title("10 photo groups matched to\n10 sound groups", loc="left", fontsize=10)
    title(fig, "Instruments that look alike sound alike, also with sound features that never saw a label",
          "868 objects with their own recording. Pair AUC: how often 2 objects of the same group sound more alike than 2 of different groups; 0.5 is chance. "
          "Random matching: mean and maximum of 1,000 permutations.")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.subplots_adjust(top=0.8)
    fig.savefig(OUT / "figT4_audio.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_gap, fig_tree, fig_robust, fig_audio):
        f()
        print(f.__name__, "done", flush=True)
