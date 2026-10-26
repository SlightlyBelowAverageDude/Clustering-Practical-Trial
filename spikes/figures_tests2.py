"""Figures for the second test round (naming the groups, non-Western split, level-by-level splitting, input side and
encoder fine-tuning). Every number is read from runs/tests. Style from figures_tests.py. Output: runs/figures_tests/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_tests import AQUA, BLUE, GRID, INK, INK2, MUTED, ORANGE, SURF, T, OUT, LEVEL, title  # noqa: E402


def fig_naming_nonwestern():
    nm = pd.read_csv(T / "naming/naming.csv")
    rf = pd.read_csv(T / "nonwestern/refit.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), gridspec_kw={"width_ratios": [1.1, 0.9, 1.1]})

    ax = axes[0]
    ks = sorted(nm.k.unique())
    x = np.arange(len(ks))
    a = nm[nm.vocabulary == "all 945 Wikidata names"].set_index("k").loc[ks]
    b = nm[nm.vocabulary == "the 157 label names"].set_index("k").loc[ks]
    series = [("majority label of the group", a.objects_right_hs3_majority_labelling, INK, "-"),
              ("zero-shot name, 157 label names", b.objects_right_hs3_group_name_code, BLUE, "-"),
              ("zero-shot name, 945 Wikidata names", a.objects_right_hs3_group_name_code, BLUE, "--")]
    for name, v, col, ls in series:
        ax.plot(x, v * 100, color=col, linestyle=ls, linewidth=2, marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=2, label=name)
        ax.text(len(ks) - 0.85, v.iloc[-1] * 100 + (2.5 if ls == "-" else -4), f"{v.iloc[-1] * 100:.0f}", fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("number of blind groups")
    ax.set_ylabel("objects given the right level-3 code, percent")
    ax.set_ylim(30, 100)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Naming a group by its mean embedding gets the type right", loc="left", fontsize=10)

    ax = axes[1]
    r100a, r100b = a.loc[100], b.loc[100]
    cats = ["one name covers\nhalf of the group", "zero-shot name is\nthe majority name", "objects with\nthe right name"]
    vals945 = [r100a.groups_nameable_without_labels, r100a.groups_name_equals_majority_name, r100a.objects_right_name_group_name * 100]
    vals157 = [r100b.groups_nameable_without_labels, r100b.groups_name_equals_majority_name, r100b.objects_right_name_group_name * 100]
    ref = [r100a.groups_nameable_by_labels, None, r100a.objects_right_name_majority_labelling * 100]
    xx = np.arange(len(cats))
    ax.bar(xx - 0.2, vals945, width=0.36, color="#9ec5f4", label="945 Wikidata names")
    ax.bar(xx + 0.2, vals157, width=0.36, color=BLUE, label="157 label names")
    for i, (v1, v2) in enumerate(zip(vals945, vals157)):
        ax.text(i - 0.2, v1 + 1, f"{v1:.0f}", ha="center", fontsize=8.5, color=INK2)
        ax.text(i + 0.2, v2 + 1, f"{v2:.0f}", ha="center", fontsize=8.5, color=INK2)
    for i, v in enumerate(ref):
        if v is not None:
            ax.hlines(v, i - 0.42, i + 0.42, color=INK, linewidth=2)
            ax.text(i + 0.44, v, f"{v:.0f} by labels", va="center", fontsize=8, color=INK2)
    ax.set_xticks(xx)
    ax.set_xticklabels(cats, fontsize=8.5)
    ax.set_ylim(0, 60)
    ax.set_ylabel("groups of 100, or percent of objects")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="x", visible=False)
    ax.set_title("The exact name is right 1 time in 5", loc="left", fontsize=10)

    ax = axes[2]
    labs = [("hs1", "level 1"), ("hs3", "level 3"), ("hs6", "level 6"), ("family", "family"), ("name", "name")]
    ne = rf[rf.set == "non-European"].set_index("label")
    eu = rf[rf.set == "Europe"].set_index("label")
    yy = np.arange(len(labs))[::-1]
    for y0, (c, n) in zip(yy, labs):
        ax.hlines(y0, ne.loc[c, "AMI"], eu.loc[c, "AMI"], color=GRID, linewidth=3)
        ax.scatter(ne.loc[c, "AMI"], y0, s=60, color=ORANGE, edgecolor=SURF, linewidth=1.5, zorder=4)
        ax.scatter(eu.loc[c, "AMI"], y0, s=60, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=4)
        ax.scatter(ne.loc[c, "probe_AMI"], y0, s=55, facecolor="none", edgecolor=ORANGE, linewidth=1.5, zorder=3)
        ax.scatter(eu.loc[c, "probe_AMI"], y0, s=55, facecolor="none", edgecolor=BLUE, linewidth=1.5, zorder=3)
        lo, hi = sorted([ne.loc[c, "AMI"], eu.loc[c, "AMI"]])
        ax.text(lo - 0.015, y0, f"{lo:.2f}", ha="right", va="center", fontsize=8, color=INK2)
        ax.text(hi + 0.015, y0, f"{hi:.2f}", ha="left", va="center", fontsize=8, color=INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels([n for _, n in labs])
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(-1.3, len(labs) - 0.4)
    ax.set_xlabel("AMI, k-means refit on the set alone (filled) and probe (hollow)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=ORANGE, marker="o", linewidth=0, label=f"non-European ({int(ne.loc['hs1', 'n']):,} coded)"),
                       Line2D([], [], color=BLUE, marker="o", linewidth=0, label=f"Europe ({int(eu.loc['hs1', 'n']):,} coded)")],
              frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Non-Western objects: blind groups and probe both lower", loc="left", fontsize=10)
    title(fig, "Naming the blind groups, and how they fit non-Western instruments",
          "Left and middle: 31,402 objects with a name and a code, names from SigLIP text embeddings. Right: region of origin from the museum record, each set clustered on its own.")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.subplots_adjust(top=0.84)
    fig.savefig(OUT / "figT5_naming_nonwestern.png", dpi=170)
    plt.close(fig)


def fig_levels_input():
    lv = pd.read_csv(T / "hier/levels.csv")
    base = pd.read_csv(T / "gap/base.csv").set_index("method")
    inp = pd.read_csv(T / "gap/input.csv")
    scores = pd.read_csv(T / "scan_ft/scores.csv") if (T / "scan_ft/scores.csv").exists() else None
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.0, 1, 1, 1]})

    ax = axes[0]
    x = np.arange(1, len(lv) + 1)
    for col, name, c, ls in (("AMI_flat_kmeans", "flat k-means", INK, "-"), ("AMI_divisive_global_pca", "level-by-level splitting", ORANGE, "-"),
                             ("AMI_average_link_cut", "average-link tree cut", BLUE, "-"), ("AMI_Ward_cut", "Ward tree cut", MUTED, "--")):
        ax.plot(x, lv[col], color=c, linestyle=ls, linewidth=2, marker="o", markersize=6, markeredgecolor=SURF, markeredgewidth=1.5, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_xlabel("level (k = " + ", ".join(str(k) for k in lv.k) + ")")
    ax.set_ylabel("AMI")
    ax.set_ylim(0.3, 0.65)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Splitting level by level does not help", loc="left", fontsize=10)

    rows = [("DINOv3-B/16, 256 px", lambda c: inp[(inp.variant == "DINOv3-B/16, 256 px") & (inp.method == "k-means")][c].iloc[0]),
            ("DINOv3-L/16, 256 px", lambda c: inp[(inp.variant == "DINOv3-L/16, 256 px") & (inp.method == "k-means")][c].iloc[0]),
            ("DINOv3-B/16, 512 px", lambda c: inp[(inp.variant == "DINOv3-B/16, 512 px") & (inp.method == "k-means")][c].iloc[0]),
            ("DINOv2-B, mirror averaged", lambda c: inp[(inp.variant == "DINOv2-B, photo and mirror image averaged") & (inp.method == "k-means")][c].iloc[0])]
    if scores is not None:
        ft = [v for v in scores.features.unique() if v.startswith("SCAN")]
        rows += [("DINOv2-B frozen, 224 px", lambda c: scores[(scores.features == "frozen, 224") & (scores.method == "k-means")][c].iloc[0])]
        if ft:
            rows += [("SCAN fine-tuned, k-means", lambda c: scores[(scores.features == ft[0]) & (scores.method == "k-means")][c].iloc[0]),
                     ("SCAN fine-tuned, cluster head", lambda c: scores[(scores.features == ft[0]) & (scores.method == "head")][c].iloc[0])]
    rows = [r for r in rows if _has(r[1])]
    y = np.arange(len(rows))[::-1]
    for ax, (c, lab) in zip(axes[1:], [("hs3", "level 3, 13 classes"), ("family", "family, 6 classes"), ("name", "name, 129 classes")]):
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
        ax.set_xlim(0.15, 0.95)
        ax.set_ylim(-0.7, len(rows) + 0.4)
        ax.set_title(lab, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_yticks(y)
        ax.set_yticklabels([r for r, _ in rows] if ax is axes[1] else [""] * len(rows))
    axes[2].set_xlabel("AMI, 12,000 objects; shaded: the gap between k-means on the published embedding and the probe")
    title(fig, "Hierarchy, input and fine-tuning: nothing moves blind clustering into the gap",
          "Left: 30,050 objects with 6-digit codes. Right: other encoders, mirror averaging and SCAN fine-tuning of the last 4 blocks of DINOv2-B on all 59,661 photos, without labels.")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.subplots_adjust(top=0.84)
    fig.savefig(OUT / "figT6_levels_input_finetune.png", dpi=170)
    plt.close(fig)


def _has(f):
    try:
        f("hs3")
        return True
    except (IndexError, KeyError):
        return False


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_naming_nonwestern()
    print("fig_naming_nonwestern done", flush=True)
    fig_levels_input()
    print("fig_levels_input done", flush=True)
