"""Figures from the cleaned set (runs/clean). Same designs as figures_pitch / figures_sheet / figures_clusters /
figure_pipeline, every number read from the clean tables. Output: runs/figures_clean/.

figA rows sheet, figB depth curve, figC reference bars, figD UMAP by H-S class and family, figE within museums,
figJ cluster map (blind groups vs family), figK sheet of 20 blind groups, figP pipeline picture.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import umap  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import os
RUN = ROOT / "runs/clean"
OUT = ROOT / os.environ.get("FIG_OUT", "runs/figures_clean")
ASSIGN = ROOT / "runs/clean_recheck"  # the partitions the tables were computed on (spikes/recheck_clean.py)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4", "purple": "#7a5cc9"}
PAL = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["purple"]]
INK, INK2, GRID, SURF, LINE = "#0b0b0b", "#52514e", "#e6e5e2", "#fcfcfb", "#c9c5bb"
HS1 = {"1": "Idiophone", "2": "Membranophone", "3": "Chordophone", "4": "Aerophone", "5": "Electrophone"}
HSC = {"Aerophone": C["blue"], "Chordophone": C["orange"], "Idiophone": C["aqua"], "Membranophone": C["yellow"], "Electrophone": C["magenta"]}
FAM = ["woodwind instrument", "string instrument", "percussion instrument", "brass instrument", "keyboard instrument", "electronic musical instrument"]
FAMC = dict(zip(FAM, PAL))
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def short(f):
    return f.replace(" instrument", "").replace(" musical", "")


def load():
    df = pd.read_csv(RUN / "records.csv", dtype=str).fillna("")
    df["hs1n"] = df.hs1.map(HS1).fillna("")
    X = np.load(RUN / "emb/dinov2_b_pad.npy")
    return df, X


def thumb(df, i, size=200):
    return ImageOps.pad(Image.open(ROOT / "data/mimo" / df.file.iloc[i]).convert("RGB"), (size, size), color=(255, 255, 255))


def fig_a(df, X, Z, sub, km20):
    """3 blind groups that hold a named instrument of another expert class. The crossing names are chosen by hand
    (shape-based cases); for each, the group where most of its objects sit is shown, with that name's share printed."""
    PREFER = ["xylophone", "electric guitar", "tube zither", "pipe organ", "ocarina", "ratchet"]
    # the 20 blind groups of the analysis itself (clean_analysis block 5: named, coded objects, names with 30+, PCA on those rows)
    a = pd.read_csv(ASSIGN / "k20_assignments.csv", dtype={"id": str})
    pos = dict(zip(df.id, range(len(df))))
    rows_idx = np.array([pos[i] for i in a.id])
    sub = df.iloc[rows_idx]
    km20 = a.cluster.values
    Z = np.zeros((len(df), 50)); Z[rows_idx] = PCA(50, random_state=0).fit_transform(X[rows_idx])
    total = sub.name.value_counts()
    rows, used = [], set()
    for name in PREFER:
        g_name = sub[sub.name == name]
        if len(g_name) < 10:
            continue
        counts = pd.Series(km20[sub.index.get_indexer(g_name.index)]).value_counts()
        c = int(counts.index[0]); share = counts.iloc[0] / len(g_name)
        g = sub[km20 == c]
        dom = g.hs1n.value_counts().index[0]
        if c in used or g_name.hs1n.value_counts().index[0] == dom or share < 0.3:
            continue
        per = g.groupby("name").agg(n=("hs1n", "size"), cls=("hs1n", lambda s: s.value_counts().index[0]))
        cross = per[(per.cls != dom) & (per.n >= 5)].sort_values("n", ascending=False)
        cross = [name] + [n for n in cross.index if n != name and (sub.name[km20 == c] == n).sum() / total[n] >= 0.2][:1]
        dom_names = per[per.cls == dom].sort_values("n", ascending=False).index.tolist()
        rows.append((c, cross + [n for n in dom_names if n not in cross][: 6 - len(cross)], dom, {n: counts.iloc[0] / len(g_name) if n == name else
                     (sub.name[km20 == c] == n).sum() / total[n] for n in cross}))
        used.add(c)
        if len(rows) == 3:
            break
    fig = plt.figure(figsize=(16, 9), facecolor="#ffffff")
    gs = fig.add_gridspec(3, 6, left=0.215, right=0.985, top=0.93, bottom=0.11, hspace=0.55, wspace=0.06)
    cen = {c: Z[sub.index.values][km20 == c].mean(0) for c in set(km20)}
    for r, (c, names, dom, shares) in enumerate(rows):
        g = sub[km20 == c]
        title = f"group {c}: mostly {dom.lower()}s"
        sub_t = "also holds\n" + "\n".join(f"{n}s ({g[g.name == n].hs1n.iloc[0].lower()}, {shares[n]:.0%} of them)" for n in shares)
        fig.text(0.015, 0.93 - r * 0.2735 - 0.08, title, fontsize=14, fontweight="bold", color=INK, va="center")
        fig.text(0.015, 0.93 - r * 0.2735 - 0.08 - 0.085, sub_t, fontsize=11, color=INK2, va="center", linespacing=1.4)
        for k, n in enumerate(names[:6]):
            gg = g[g.name == n]
            ax = fig.add_subplot(gs[r, k])
            if len(gg) == 0:
                ax.axis("off"); continue
            d = np.linalg.norm(Z[gg.index.values] - cen[c], axis=1)
            j = gg.index.values[int(np.argsort(d)[min(1, len(d) - 1)])]
            ax.imshow(thumb(df, j, 260)); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_edgecolor(HSC[df.hs1n.iloc[j]]); s.set_linewidth(5); s.set_visible(True)
            ax.set_title(n, fontsize=13, color=INK, pad=6)
    fig.text(0.5, 0.975, "Three groups the computer found on its own. Frame colour: the expert class (how the sound is made).", ha="center", fontsize=15, color=INK)
    fig.legend(handles=[Patch(facecolor=v, label=k) for k, v in HSC.items() if k != "Membranophone"], loc="lower center", ncol=4, frameon=False, fontsize=13,
               bbox_to_anchor=(0.5, 0.015), handlelength=1.4, columnspacing=2.2)
    fig.savefig(OUT / "figA_rows.png", dpi=160, facecolor="#ffffff"); plt.close(fig)


def fig_b():
    s = pd.read_csv(RUN / "same_rows.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    h = s[s.label.str.startswith("hs")].sort_values("classes")
    ax.plot(h.classes, h.AMI_at_classes, "-o", color=C["blue"], lw=2, ms=6, label="Hornbostel-Sachs, levels 1 to 6, k-means")
    ax.plot(h.classes, h.probe_AMI, "--o", color=C["blue"], lw=1.2, ms=6, mfc=SURF, label="Hornbostel-Sachs, linear probe (AMI of its guesses)")
    for lab, col, name in (("family", C["orange"], "Western family"), ("name", C["aqua"], "instrument name")):
        n = s[s.label == lab].iloc[0]
        ax.plot([n.classes], [n.AMI_at_classes], "o", color=col, ms=8, label=f"{name} ({int(n.classes)} classes), k-means")
        ax.plot([n.classes], [n.probe_AMI], "o", color=col, ms=8, mfc=SURF, label=f"{name}, linear probe")
    ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 50, 100]); ax.set_xticklabels(["5", "10", "20", "50", "100"])
    ax.set_xlabel("number of classes in the label"); ax.set_ylabel("AMI against the label"); ax.set_ylim(0, 1.0)
    ax.set_title(f"Blind k-means (filled) and the supervised probe (hollow), {int(s.n.min()):,} to {int(s.n.max()):,} objects", loc="left", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "figB_depth.png", dpi=170); plt.close(fig)


def fig_c():
    s = pd.read_csv(RUN / "same_rows.csv").set_index("label")
    t = pd.read_csv(RUN / "taxonomy_vs_taxonomy.csv")
    hs3 = s.loc["hs3"]
    name_vs = t[(t.a == "name") & (t.b == "hs3")].AMI.iloc[0]
    rows = [("random labels", 0.0, C["magenta"]), ("k-means on DINOv2, no training", hs3.AMI_at_classes, C["blue"]),
            ("instrument name (perfect fine labels)", name_vs, C["aqua"]), ("linear probe, supervised ceiling (AMI of its guesses)", hs3.probe_AMI, C["yellow"])]
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, col) in zip(y, rows):
        ax.barh(yi, v, color=col, height=0.55); ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9); ax.set_xlim(0, 1.0)
    ax.set_xlabel(f"AMI against Hornbostel-Sachs level 3 ({int(hs3.classes)} classes, {int(hs3.n):,} objects)")
    ax.set_title(f"Where {hs3.AMI_at_classes:.2f} sits: chance, blind groups, perfect names, ceiling", loc="left", fontsize=10.5)
    ax.grid(axis="y", visible=False); fig.tight_layout(); fig.savefig(OUT / "figC_reference.png", dpi=170); plt.close(fig)


def fig_d(df, Z, sub, E):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, col, cmap, title in [(axes[0], "hs1n", HSC, "coloured by Hornbostel-Sachs class (how the sound is made)"),
                                 (axes[1], "family", FAMC, "coloured by Western family (how it is played)")]:
        lab = sub[col].values
        for k, cc in cmap.items():
            sel = lab == k
            if sel.sum():
                ax.scatter(E[sel, 0], E[sel, 1], s=3, color=cc, alpha=0.55, linewidths=0, label=f"{short(k)} ({sel.sum():,})")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title(title, fontsize=10, loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8.5, frameon=False, markerscale=4)
    fig.suptitle(f"The same {len(sub):,} MIMO photos, DINOv2 features, UMAP: the islands are instrument families; the top acoustic class merges brass with woodwind",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(OUT / "figD_umap.png", dpi=160); plt.close(fig)


def fig_e():
    w = pd.read_csv(RUN / "within_museum.csv")
    h = w[w.label == "hs1n"].sort_values("AMI_at_classes"); r = w[w.label == "region"].set_index("provider")
    h["short"] = h.provider.str.replace("Staatliche Museen zu Berlin", "Berlin").str.replace("Museum of Musical Instruments, Leipzig University", "Leipzig University")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    y = np.arange(len(h)); ax.hlines(y, 0, h.AMI_at_classes, color=GRID, lw=1)
    ax.plot(h.AMI_at_classes, y, "o", color=C["blue"], ms=8, label=f"Hornbostel-Sachs class ({h.classes.min()} to {h.classes.max()} classes)")
    rr = h.provider.map(r.AMI_at_classes)
    ax.plot(rr, y, "o", color=C["orange"], ms=8, label=f"region of origin ({int(r.classes.min())} to {int(r.classes.max())} regions)")
    for yi, a in enumerate(h.AMI_at_classes):
        ax.text(a + 0.012, yi, f"{a:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([f"{s[:42]} ({n:,})" for s, n in zip(h.short, h.n)], fontsize=8.5)
    ax.set_xlim(0, 0.72); ax.set_xlabel("AMI of image clusters vs label, inside one museum")
    ax.set_title("Inside one museum: the acoustic class is in the object, origin mostly is not", loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(OUT / "figE_museums.png", dpi=170); plt.close(fig)


def match_colours(pred, y):
    ct = pd.crosstab(pred, pd.Categorical(y, categories=FAM))
    r, c = linear_sum_assignment(-ct.values)
    return {ct.index[i]: PAL[j] for i, j in zip(r, c)}


def fig_j(df, X):
    """The 6-group partitions of the methods comparison itself (same 12,000 rows, same PCA), so the AMI in the titles
    equals runs/clean/methods.csv."""
    a = pd.read_csv(ASSIGN / "methods_k6_assignments.csv", dtype={"id": str})
    pos = dict(zip(df.id, range(len(df))))
    rows_idx = np.array([pos[i] for i in a.id])
    sub = df.iloc[rows_idx]
    Zm = PCA(50, random_state=0).fit_transform(X[rows_idx])
    E = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=0).fit_transform(Zm)
    fam = sub.family.values
    km6, al6 = a.km6.values, a.al6.values
    from sklearn.metrics import adjusted_mutual_info_score as ami
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    panels = [(f"blind: k-means, 6 groups (AMI vs family {ami(fam, km6):.2f})", km6, match_colours(km6, fam)),
              (f"blind: average-link, 6 groups (AMI vs family {ami(fam, al6):.2f})", al6, match_colours(al6, fam)), ("truth: Western family", fam, FAMC)]
    for ax, (title, lab, cmap) in zip(axes, panels):
        for k, col in cmap.items():
            sel = lab == k
            if sel.sum():
                ax.scatter(E[sel, 0], E[sel, 1], s=2.5, color=col, alpha=0.6, linewidths=0, label=f"{short(k) if isinstance(k, str) else f'group {k}'} ({sel.sum():,})")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, loc="left", fontsize=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False, markerscale=5)
    fig.suptitle(f"The same {len(sub):,} instrument photos, coloured by blind groups and by the true family (colours matched by overlap)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "figJ_cluster_map.png", dpi=160); plt.close(fig)


def fig_k(df, sub, Zs, km20):
    cen = np.stack([Zs[km20 == c].mean(0) for c in range(20)])
    order = np.argsort(-np.bincount(km20))
    fig, axes = plt.subplots(5, 4, figsize=(16, 15.5))
    for ax, c in zip(axes.ravel(), order):
        ii = np.where(km20 == c)[0]
        d = np.linalg.norm(Zs[ii] - cen[c], axis=1); pick = ii[np.argsort(d)[:8]]
        sheet = Image.new("RGB", (4 * 150 + 12, 2 * 150 + 4), (252, 252, 251))
        for t, j in enumerate(pick):
            sheet.paste(thumb(df, sub.index.values[j], 150), ((t % 4) * 154, (t // 4) * 154))
        ax.imshow(sheet); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        vc = sub.name.iloc[ii].value_counts(); fv = sub.family.iloc[ii].value_counts()
        for s in ax.spines.values():
            s.set_edgecolor(FAMC[fv.index[0]]); s.set_linewidth(3); s.set_visible(True)
        second = f", {vc.index[1][:18]} {vc.iloc[1] / len(ii):.0%}" if len(vc) > 1 else ""
        ax.set_title(f"group {c}, {len(ii)} objects, {short(fv.index[0])} {fv.iloc[0] / len(ii):.0%}\n{vc.index[0][:18]} {vc.iloc[0] / len(ii):.0%}{second}", fontsize=8.5, loc="left", color=INK)
    fig.suptitle("The 20 blind k-means groups, 8 photos nearest each centre. Caption: size, dominant family, top 2 names with their share. Frame: the family colour.",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "figK_cluster_sheet.png", dpi=130); plt.close(fig)


def fig_p(df, sub, Zs, E):
    km = KMeans(6, n_init=10, random_state=0).fit_predict(Zs)
    fig = plt.figure(figsize=(13, 6), facecolor="#ffffff")
    for i, n in enumerate(["violin", "clarinet", "bugle", "gong", "frame drum", "square piano"]):
        g = sub[sub.name == n]
        if len(g) == 0:
            continue
        ax = fig.add_axes([0.03 + (i % 2) * 0.095, 0.56 - (i // 2) * 0.235, 0.085, 0.2])
        ax.imshow(thumb(df, g.index.values[0])); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_edgecolor(LINE); s.set_linewidth(1)
    fig.text(0.125, 0.86, f"{round(len(df) / 1000)},000 photos", ha="center", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.125, 0.80, "with catalogue cards", ha="center", fontsize=11, color=INK2)
    ax = fig.add_axes([0.30, 0.08, 0.40, 0.68])
    for c in range(6):
        sel = km == c; ax.scatter(E[sel, 0], E[sel, 1], s=3, color=PAL[c], alpha=0.6, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xlim(np.percentile(E[:, 0], 0.3), np.percentile(E[:, 0], 99.7)); ax.set_ylim(np.percentile(E[:, 1], 0.3), np.percentile(E[:, 1], 99.7))
    for s in ax.spines.values():
        s.set_edgecolor(LINE)
    fig.text(0.50, 0.86, "clustering", ha="center", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.50, 0.80, "each dot is a photo, each colour a group found without labels", ha="center", fontsize=11, color=INK2)
    n_rec = len(pd.read_csv(RUN / "audio_dedup/objects.csv"))
    items = [("expert code", "Hornbostel-Sachs, 6 levels"), ("instrument names", "via Wikipedia"), ("origin", "country and region"),
             ("recordings", f"about {round(n_rec, -2):,} objects, same instrument")]
    for i, (t, s) in enumerate(items):
        y = 0.63 - i * 0.165
        fig.add_artist(FancyBboxPatch((0.765, y), 0.215, 0.12, boxstyle="round,pad=0.008,rounding_size=0.012", transform=fig.transFigure, facecolor="#f3f1ec", edgecolor=LINE, linewidth=1))
        fig.text(0.78, y + 0.075, t, fontsize=13, fontweight="bold", color=INK, va="center"); fig.text(0.78, y + 0.035, s, fontsize=10.5, color=INK2, va="center")
    fig.text(0.8725, 0.86, "compared with", ha="center", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.8725, 0.80, "do the groups match any of these?", ha="center", fontsize=11, color=INK2)
    for x0, x1 in ((0.225, 0.29), (0.71, 0.755)):
        fig.add_artist(FancyArrowPatch((x0, 0.45), (x1, 0.45), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22, color=INK2, linewidth=2))
    fig.savefig(OUT / "figP_pipeline.png", dpi=170, facecolor="#ffffff"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["a", "b", "c", "d", "e", "j", "k", "p"]
    df, X = load()
    Z = PCA(50, random_state=0).fit_transform(X)
    sub = df[(df.name != "") & (df.family != "") & (df.hs3 != "")].copy()
    vc = sub.name.value_counts(); sub = sub[sub.name.map(vc) >= 50]
    rng = np.random.default_rng(0)
    if len(sub) > 14000:
        sub = sub.loc[np.sort(rng.choice(sub.index.values, 14000, replace=False))]
    Zs = Z[sub.index.values]
    km20 = KMeans(20, n_init=3, random_state=0).fit_predict(Zs)
    E = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=0).fit_transform(Zs) if any(c in which for c in "djp") else None
    if "a" in which:
        fig_a(df, X, Z, sub, km20); print("A", flush=True)
    if "b" in which:
        fig_b(); print("B", flush=True)
    if "c" in which:
        fig_c(); print("C", flush=True)
    if "d" in which:
        fig_d(df, Z, sub, E); print("D", flush=True)
    if "e" in which:
        fig_e(); print("E", flush=True)
    if "j" in which:
        fig_j(df, X); print("J", flush=True)
    if "k" in which:
        fig_k(df, sub, Zs, km20); print("K", flush=True)
    if "p" in which:
        fig_p(df, sub, Zs, E); print("P", flush=True)


if __name__ == "__main__":
    main()
