"""Cluster pictures on the 12,000-object method-comparison subset.

figJ: one UMAP map coloured 3 ways: blind k-means (6 groups), blind average-link (6 groups), true Western family.
      Cluster colours are matched to the family each cluster mostly holds, so a colour mismatch is a disagreement.
figK: thumbnail sheet of the 20 blind k-means groups (8 photos each, nearest the centre), with majority name and purity.
Outputs: runs/figures_pitch_v2/figJ_cluster_map.png, figK_cluster_sheet.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import umap  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/figures_pitch_v2"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5cc9"]
FAM = ["woodwind instrument", "string instrument", "percussion instrument", "brass instrument", "keyboard instrument", "electronic musical instrument"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e2", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def match_colours(pred, y):
    """Assign each cluster the colour of the family it overlaps most (one family per cluster, Hungarian on the contingency)."""
    ct = pd.crosstab(pred, pd.Categorical(y, categories=FAM))
    r, c = linear_sum_assignment(-ct.values)
    return {ct.index[i]: PAL[j] for i, j in zip(r, c)}


def main():
    sub = pd.read_csv(ROOT / "runs/method_compare/subset.csv", dtype=str).fillna("")
    full = pd.read_csv(ROOT / "runs/mimo_full/sample_full.csv", dtype=str).fillna("")
    row = dict(zip(full.id, range(len(full))))
    X = np.load(ROOT / "runs/mimo_full/emb/dinov2_b.npy")[[row[i] for i in sub.id]]
    Z = PCA(50, random_state=0).fit_transform(X)
    E = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=0).fit_transform(Z)
    fam = sub.family.values
    km6 = KMeans(6, n_init=10, random_state=0).fit_predict(Z)
    al6 = AgglomerativeClustering(6, linkage="average", metric="cosine").fit_predict(Z)
    km20 = KMeans(20, n_init=10, random_state=0).fit_predict(Z)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    panels = [("blind: k-means, 6 groups (AMI vs family 0.53)", km6, match_colours(km6, fam)),
              ("blind: average-link, 6 groups (AMI vs family 0.66)", al6, match_colours(al6, fam)),
              ("truth: Western family", fam, dict(zip(FAM, PAL)))]
    for ax, (title, lab, cmap) in zip(axes, panels):
        for k, col in cmap.items():
            sel = lab == k
            name = k.replace(" instrument", "").replace(" musical", "") if isinstance(k, str) else f"group {k}"
            ax.scatter(E[sel, 0], E[sel, 1], s=2.5, color=col, alpha=0.6, linewidths=0, label=f"{name} ({sel.sum():,})")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, loc="left", fontsize=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False, markerscale=5)
    fig.suptitle("The same 12,000 instrument photos, coloured by blind groups and by the true family (colours matched by overlap)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "figJ_cluster_map.png", dpi=160); plt.close(fig)

    # thumbnail sheet of the 20 k-means groups
    name = sub.name.values
    cen = np.stack([Z[km20 == c].mean(0) for c in range(20)])
    order = np.argsort(-np.bincount(km20))
    fig, axes = plt.subplots(5, 4, figsize=(16, 15.5))
    for ax, c in zip(axes.ravel(), order):
        ii = np.where(km20 == c)[0]
        d = np.linalg.norm(Z[ii] - cen[c], axis=1); pick = ii[np.argsort(d)[:8]]
        tiles = []
        for j in pick:
            im = Image.open(ROOT / "data/mimo" / sub.file.iloc[j]).convert("RGB")
            tiles.append(ImageOps.pad(im, (150, 150), color=(252, 252, 251)))
        sheet = Image.new("RGB", (4 * 150 + 3 * 4, 2 * 150 + 4), (252, 252, 251))
        for t, im in enumerate(tiles):
            sheet.paste(im, ((t % 4) * 154, (t // 4) * 154))
        ax.imshow(sheet); ax.set_xticks([]); ax.set_yticks([])
        vc = pd.Series(name[ii]).value_counts()
        fv = pd.Series(fam[ii]).value_counts()
        col = PAL[FAM.index(fv.index[0])]
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(3); s.set_visible(True)
        second = f", {vc.index[1][:18]} {vc.iloc[1] / len(ii):.0%}" if len(vc) > 1 else ""
        ax.set_title(f"group {c}, {len(ii)} objects, {fv.index[0].replace(' instrument', '').replace(' musical', '')} {fv.iloc[0] / len(ii):.0%}\n"
                     f"{vc.index[0][:18]} {vc.iloc[0] / len(ii):.0%}{second}", fontsize=8.5, loc="left", color=INK)
    fig.suptitle("The 20 blind k-means groups, 8 photos nearest each centre. Caption: top 2 names with their share, dominant family, size. Frame: the family colour matched to the group.",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "figK_cluster_sheet.png", dpi=130); plt.close(fig)
    print("J, K done")


if __name__ == "__main__":
    main()
