"""Figures for the third test round (per-node map of the expert tree, Met transfer, within-name structure).
Every number is read from runs/tests. Style from figures_tests.py. Output: runs/figures_tests/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_tests import BLUE, GRID, INK, INK2, MUTED, ORANGE, SURF, T, OUT, title  # noqa: E402

RAMP = LinearSegmentedColormap.from_list("blue_ramp", ["#e8f0fb", "#9ec5f4", "#2a78d6", "#143c6b"])
SHORT = {"interruptive free aerophones (free reeds)": "free reeds", "in the form of a frame (jaw harps)": "jaw harps",
         "in board or comb form (lamellophones)": "lamellophones", "simple chordophones (zithers)": "zithers", "composite chordophones": "composite",
         "wind instruments proper": "wind instruments", "edge-blown (flutes)": "flutes", "trumpets (lip-vibrated)": "trumpets",
         "displacement free aerophones": "displacement", "directly struck": "struck", "indirectly struck": "indir. struck", "struck idiophones": "struck", "struck drums": "struck"}


def fig_nodes():
    n = pd.read_csv(T / "nodes/nodes.csv", dtype={"code": str, "level": str})
    a = n[n.method == "average-link cosine"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})

    ax = axes[0]
    lv = {d: a[a.level == str(d)].set_index("code") for d in (1, 2, 3)}
    total = lv[1].objects.sum()
    x0 = {}
    x = 0.0
    for code, r in lv[1].sort_values("code").iterrows():
        x0[code] = x
        x += r.objects / total
    for d in (1, 2, 3):
        y = 3 - d
        starts = dict(x0) if d == 1 else {}
        if d > 1:
            # children start where their parent starts and are laid out in code order
            cursor = {}
            for code, r in lv[d].sort_values("code").iterrows():
                parent = code[:-1]
                while parent and parent not in x0:
                    parent = parent[:-1]
                cursor.setdefault(parent, x0.get(parent, 0.0))
                starts[code] = cursor[parent]
                cursor[parent] += r.objects / total
            x0.update(starts)
        for code, r in lv[d].iterrows():
            w = r.objects / total
            ax.add_patch(Rectangle((starts[code], y), w, 0.92, facecolor=RAMP(r.F1), edgecolor=SURF, linewidth=1.5))
            name = SHORT.get(r.hs_name, r.hs_name)
            col = SURF if r.F1 > 0.62 else INK
            if w > 0.07:
                ax.text(starts[code] + w / 2, y + 0.62, f"{code} {name}" if w > 0.1 else f"{code}\n{name}", ha="center", va="center", fontsize=8.5 if w > 0.12 else 7.5, color=col)
                ax.text(starts[code] + w / 2, y + 0.28, f"F1 {r.F1:.2f}", ha="center", va="center", fontsize=8.5, color=col)
            elif w > 0.025:
                ax.text(starts[code] + w / 2, y + 0.46, f"{code}\n{r.F1:.2f}", ha="center", va="center", fontsize=7, color=col)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3)
    ax.set_yticks([2.46, 1.46, 0.46])
    ax.set_yticklabels(["level 1", "level 2", "level 3"])
    ax.set_xticks([])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("How well one subtree of the blind tree reproduces each expert node (width: objects, colour: F1)", loc="left", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, aspect=40)
    cb.set_label("F1 of the best matching subtree (1 = the node is exactly one subtree)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    ax = axes[1]
    order = ["1", "2", "3", "4", "5", "6", "name"]
    rng = np.random.default_rng(0)
    for i, l in enumerate(order):
        g = a[a.level == l]
        k = n[(n.method == "k-means, k = classes") & (n.level == l)]
        jit = rng.uniform(-0.18, 0.18, len(g))
        ax.scatter(i + jit, g.F1, s=np.clip(g.objects / 60, 8, 90), color=BLUE, alpha=0.55, linewidths=0, zorder=3)
        wm = (g.F1 * g.objects).sum() / g.objects.sum()
        km = (k.F1 * k.objects).sum() / k.objects.sum()
        ax.hlines(wm, i - 0.3, i + 0.3, color=INK, linewidth=2.2, zorder=4)
        ax.hlines(km, i - 0.3, i + 0.3, color=ORANGE, linewidth=2.2, linestyle="--", zorder=4)
        ax.text(i + 0.33, wm, f"{wm:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"level {l}\n({len(a[a.level == l])})" if l != "name" else f"names\n({len(a[a.level == l])})" for l in order], fontsize=8.5)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("F1 of the best matching subtree")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", linewidth=0, alpha=0.6, label="one expert class (size: objects)"),
                       Line2D([], [], color=INK, linewidth=2.2, label="mean weighted by objects, blind tree"),
                       Line2D([], [], color=ORANGE, linewidth=2.2, linestyle="--", label="same for the best flat k-means cluster")],
              frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Deeper nodes and names: half are one subtree, half are not", loc="left", fontsize=10)
    title(fig, "Which expert nodes does the blind tree reproduce as one branch?",
          "30,050 objects with 6-digit codes, average-link tree on DINOv2-B. For each expert class, the tree node whose leaves match it best "
          "(F1 of precision and recall). Classes with 30 or more objects.")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.subplots_adjust(top=0.84)
    fig.savefig(OUT / "figT7_node_map.png", dpi=170)
    plt.close(fig)




def fig_met_within():
    tr = pd.read_csv(T / "met/transfer.csv")
    nt = pd.read_csv(T / "met/name_transfer.csv")
    wn = pd.read_csv(T / "within_name/scores.csv")
    ref = tr[tr.partition == "reference"].iloc[0]
    parts = tr[tr.partition != "reference"].reset_index(drop=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1.2, 0.9, 0.9, 1.2]})

    ax = axes[0]
    labs = ["level 1", "level 2", "full"]
    cols = [BLUE, ORANGE, "#1baf7a"]
    for i, lab in enumerate(labs):
        for j, (_, p) in enumerate(parts.iterrows()):
            x = i + (j - 1) * 0.22
            ax.scatter(x, p[f"AMI_{lab}_transferred"], s=70, color=cols[j], edgecolor=SURF, linewidth=1.5, zorder=4)
            ax.scatter(x, p[f"AMI_{lab}_refit_on_Met"], s=65, facecolor="none", edgecolor=cols[j], linewidth=1.5, zorder=3)
            ax.vlines(x, p[f"AMI_{lab}_refit_on_Met"], p[f"AMI_{lab}_transferred"], color=cols[j], linewidth=1, alpha=0.6)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(["Met class\n(level 1)", "Met level 2", "Met full\nclassification"], fontsize=8.5)
    ax.set_ylabel("AMI on the Met objects")
    ax.set_ylim(0, 0.7)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, marker="o", linewidth=0, label=p.partition.replace("MIMO ", "MIMO groups, ")) for c, p in zip(cols, parts.itertuples())]
    h += [Line2D([], [], color=INK2, marker="o", markerfacecolor="none", linewidth=0, label="hollow: k-means refit on the Met, same k")]
    ax.legend(handles=h, frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("MIMO groups carried to the Met (filled)", loc="left", fontsize=10)

    ax = axes[1]
    names = [p.partition.replace("MIMO ", "").split(",")[0] for p in parts.itertuples()] + ["MIMO probe"]
    vals = [p.level1_accuracy_from_MIMO_majority * 100 for p in parts.itertuples()] + [ref.level1_accuracy_MIMO_probe_on_Met * 100]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=[BLUE] * 3 + [INK2], width=0.6)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=8.5, color=INK2, bbox=dict(facecolor=SURF, edgecolor="none", pad=1), zorder=5)
    ax.axhline(ref.level1_majority_class_share * 100, color=MUTED, linewidth=1.2, linestyle="--")
    ax.text(-0.45, ref.level1_majority_class_share * 100 + 1.5, f"largest Met class {ref.level1_majority_class_share * 100:.0f}", ha="left", fontsize=8, color=INK2, bbox=dict(facecolor=SURF, edgecolor="none", pad=1), zorder=5)
    ax.axhline(ref["Met_probe_acc_level 1"] * 100, color=INK, linewidth=1.2)
    ax.text(-0.45, ref["Met_probe_acc_level 1"] * 100 + 1.5, f"Met's own probe {ref['Met_probe_acc_level 1'] * 100:.0f}", ha="left", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"groups\n{n}" if n != "MIMO probe" else "MIMO\nprobe" for n in names], fontsize=8.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Met objects with the right class, percent")
    ax.grid(axis="x", visible=False)
    ax.set_title("Class label carried from MIMO", loc="left", fontsize=10)

    ax = axes[2]
    rows = nt[~nt.source.str.startswith("Met majority")].reset_index(drop=True)
    y = np.arange(len(rows))[::-1]
    ax.barh(y + 0.18, rows.exact * 100, height=0.34, color=BLUE, label="exact name")
    ax.barh(y - 0.18, rows.substring * 100, height=0.34, color="#9ec5f4", label="one name contains the other")
    for yi, r in zip(y, rows.itertuples()):
        ax.text(r.exact * 100 + 0.8, yi + 0.18, f"{r.exact * 100:.0f}", va="center", fontsize=8, color=INK2)
        ax.text(r.substring * 100 + 0.8, yi - 0.18, f"{r.substring * 100:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("MIMO ", "").replace(" (the most frequent transferred name)", "") for s in rows.source], fontsize=7.5)
    ax.set_xlabel("Met objects whose name matches, percent")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_title("Instrument name carried from MIMO", loc="left", fontsize=10)

    ax = axes[3]
    names4 = list(dict.fromkeys(wn.name))
    labs = [("maker", BLUE), ("decade", ORANGE), ("museum", "#1baf7a")]
    for i, nm in enumerate(names4):
        for j, (lab, c) in enumerate(labs):
            r = wn[(wn.name == nm) & (wn.label == lab)]
            if r.empty:
                continue
            r = r.iloc[0]
            x = i + (j - 1) * 0.25
            ax.scatter(x, r.AMI_k10, s=70, color=c, edgecolor=SURF, linewidth=1.5, zorder=4)
            if pd.notna(r.get("probe_AMI", np.nan)):
                ax.scatter(x, r.probe_AMI, s=65, facecolor="none", edgecolor=c, linewidth=1.5, zorder=3)
                ax.vlines(x, r.AMI_k10, r.probe_AMI, color=c, linewidth=1, alpha=0.6)
    ax.set_xticks(range(len(names4)))
    ax.set_xticklabels([n.replace("Western concert ", "") for n in names4], fontsize=8.5)
    ax.set_ylabel("AMI, 10 blind sub-groups (filled) and probe (hollow)")
    ax.set_ylim(0, 0.8)
    ax.legend(handles=[Line2D([], [], color=c, marker="o", linewidth=0, label=lab) for lab, c in labs], frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Inside one name: museum beats period", loc="left", fontsize=10)
    title(fig, "Transfer to the Met, and the structure inside one instrument name",
          "Left 3 panels: public-domain Met instruments assigned to the MIMO blind groups by nearest centre. Right: objects of the 4 largest MIMO names, "
          "sub-grouped on their own; makers with 5 or more objects, decades and museums with 10 or more.")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.subplots_adjust(top=0.84)
    fig.savefig(OUT / "figT8_met_within.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["nodes", "met"]
    if "nodes" in which:
        fig_nodes()
        print("fig_nodes done", flush=True)
    if "met" in which:
        fig_met_within()
        print("fig_met_within done", flush=True)
