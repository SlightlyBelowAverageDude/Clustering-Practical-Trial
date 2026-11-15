"""Week 6 / P3 figures from saved CSV/NPY outputs only."""
import os, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data/phase2/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

W6 = ROOT / 'reports/week6'; P3 = W6 / 'p3_experiments'; FIG = W6 / 'figures'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
MC = {'kmeans': ('k-Means', '#2a78d6', 'o'), 'agglomerative_average_cosine': ('Agglomerative (avg, cosine)', '#eb6834', 's'),
      'spectral_knn': ('Spectral (15-NN)', '#eda100', 'D')}
ENC = {'clip': 'CLIP ViT-B/32', 'siglip': 'SigLIP ViT-B/16'}
HS3 = {'111': 'struck idiophones', '112': 'rattles/scrapers', '121': 'plucked idioph. (frame)', '122': 'lamellophones',
       '211': 'struck drums', '242': 'friction drums', '311': 'bar zithers', '312': 'tube zithers', '313': 'raft zithers',
       '314': 'board zithers (pianos…)', '315': 'trough zithers', '321': 'lutes', '322': 'harps', '323': 'harp lutes',
       '412': 'free reeds (accordions…)', '421': 'flutes (edge)', '422': 'reedpipes', '423': 'lip-reed (brass)',
       '133': '133', '513': 'electrophones 513', '531': 'electrophones 531', '533': 'electrophones 533'}
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'font.size': 11,
                     'axes.titlesize': 12.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True, 'legend.frameon': False})
FOOT = 'MIMO pilot (2,000 images, 216 museums); HS3 = Hornbostel–Sachs level 3; development pilot, not held-out; no significance test.'


def finish(fig, title, sub, name, foot=FOOT, top=0.84, bottom=0.12):
    fig.suptitle(title, x=0.01, ha='left', fontsize=15, fontweight='bold')
    fig.text(0.01, 0.935, sub, ha='left', va='top', fontsize=10.5, color=INK2)
    fig.text(0.01, 0.01, foot, ha='left', fontsize=8.5, color=INK2)
    fig.subplots_adjust(top=top, bottom=bottom); fig.savefig(FIG / name, dpi=220); plt.close(fig); print('wrote', name)


rob = pd.read_csv(W6 / 'P3_ROBUSTNESS_RESULTS.csv')
full = pd.read_csv(W6 / 'P2_METHOD_COMPARISON.csv')
full = full[(full.scoring == 'all_rows') & (full.representation == 'centered_l2') & (full.requested_k == 10)]

# ---------------- 1. Spectral robustness
fig, axes = plt.subplots(2, 2, figsize=(13.33, 8.2), sharex=True)
xs = np.arange(6)
for j, enc in enumerate(['clip', 'siglip']):
    for i, metric in enumerate(['AMI', 'ARI']):
        ax = axes[i, j]
        for off, (m, (lab, col, mk)) in zip([-0.18, 0, 0.18], MC.items()):
            g = rob[(rob.encoder == enc) & (rob.method == m)].sort_values('subset_seed')
            f = full[(full.encoder == enc) & (full.method == m)].iloc[0]
            y = list(g[f'HS3_{metric}_mean']) + [f[f'HS3_{metric}_mean']]
            e = list(g.HS3_AMI_sd.fillna(0)) + [0 if np.isnan(f.HS3_AMI_sd) else f.HS3_AMI_sd] if metric == 'AMI' else None
            ax.errorbar(xs + off, y, yerr=e, fmt=mk, color=col, ms=8, capsize=3, mec=SURF, mew=1.2, label=lab, elinewidth=1.4, ecolor=col)
        ax.axvline(4.5, color=INK2, lw=1, ls=':')
        ax.set_title(f'{ENC[enc]} · HS3 {metric}', loc='left')
        if j == 0: ax.set_ylabel(f'HS3 {metric}')
    axes[1, j].set_xticks(xs); axes[1, j].set_xticklabels(['subset\n2026', '2027', '2028', '2029', '2030', 'full\npilot (P2)'])
for ax in axes[0]: ax.set_ylim(0.40, 0.57)
for ax in axes[1]: ax.set_ylim(0.18, 0.42)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper right', bbox_to_anchor=(0.99, 0.905), fontsize=9.5, ncol=3)
r = rob[rob.method == 'spectral_knn']
win = {e: int((r[r.encoder == e].delta_HS3_AMI_vs_kmeans > 0).sum()) for e in ['clip', 'siglip']}
winari = {e: int((r[r.encoder == e].delta_HS3_ARI_vs_kmeans > 0).sum()) for e in ['clip', 'siglip']}
finish(fig, 'Spectral robustness on five 80% subsets (k = 10, centered + L2 refit per subset)',
       f'Spectral AMI > k-Means in {win["clip"]}/5 CLIP and {win["siglip"]}/5 SigLIP subsets; Spectral ARI > k-Means in {winari["clip"]}/5 and {winari["siglip"]}/5. '
       'Error bars: SD over 5 seeds (agglomerative deterministic).',
       'P3_SPECTRAL_ROBUSTNESS.png', top=0.82,
       foot='1,600 of 2,000 images per subset (phase-2 seeds 2026–2030; overlapping, not independent); HS3 scored on each subset’s labelled rows. Not held-out.')

# ---------------- 2. Blind k robustness (Spectral)
bk = pd.read_csv(W6 / 'P3_BLIND_K_ROBUSTNESS.csv'); s = bk[bk.method == 'spectral_knn']
fig, ax = plt.subplots(figsize=(13.33, 6.6))
rows = [(e, ss) for e in ['clip', 'siglip'] for ss in [2026, 2027, 2028, 2029, 2030]]
for yi, (e, ss) in enumerate(rows):
    g = s[(s.encoder == e) & (s.subset_seed == ss)].set_index('criterion')
    sil, eg = g.loc['silhouette_cosine'], g.loc['eigengap']
    ax.plot([min(sil.HS3_AMI_selected, eg.HS3_AMI_selected, sil.HS3_AMI_fixed_k10), sil.oracle_HS3_AMI_NOT_BLIND], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter(sil.oracle_HS3_AMI_NOT_BLIND, yi, s=150, facecolors='none', edgecolors=INK2, lw=1.5, zorder=2, label='best k chosen with labels (not blind)' if yi == 0 else None)
    ax.scatter(sil.HS3_AMI_fixed_k10, yi, marker='|', s=300, color=INK, lw=2, zorder=3, label='fixed k = 10' if yi == 0 else None)
    ax.scatter(sil.HS3_AMI_selected, yi + 0.12, marker='o', s=80, color='#2a78d6', edgecolors=SURF, zorder=4, label='silhouette-selected k' if yi == 0 else None)
    ax.scatter(eg.HS3_AMI_selected, yi - 0.12, marker='s', s=80, color='#eb6834', edgecolors=SURF, zorder=4, label='eigengap-selected k' if yi == 0 else None)
    ax.annotate(f'k={int(sil.selected_k)}', (sil.HS3_AMI_selected, yi + 0.12), xytext=(-8, 7), textcoords='offset points', fontsize=8.5, color='#2a78d6', ha='right')
    ax.annotate(f'k={int(eg.selected_k)}', (eg.HS3_AMI_selected, yi - 0.12), xytext=(-8, -13), textcoords='offset points', fontsize=8.5, color='#b5461c', ha='right')
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'{ENC[e].split()[0]} · subset {ss}' for e, ss in rows]); ax.invert_yaxis()
ax.axhline(4.5, color=INK2, lw=1, ls=':'); ax.set_xlabel('HS3 AMI of the Spectral partition at the chosen k (mean over 5 seeds)')
ax.legend(loc='lower left', fontsize=9.5, ncol=4, bbox_to_anchor=(0, -0.22)); ax.grid(axis='y', visible=False)
sw = {c: (s[s.criterion == c].gap_to_oracle.mean(), s[s.criterion == c].selected_k.tolist()) for c in ['silhouette_cosine', 'eigengap']}
finish(fig, 'Blind choice of k for Spectral clustering across subsets',
       f'Eigengap picked k ∈ {{{", ".join(map(str, sorted(set(sw["eigengap"][1]))))}}}, silhouette k ∈ {{{", ".join(map(str, sorted(set(sw["silhouette_cosine"][1]))))}}}. '
       f'Mean AMI gap to the label-chosen k: eigengap {sw["eigengap"][0]:+.3f}, silhouette {sw["silhouette_cosine"][0]:+.3f}. Neither is stable.',
       'P3_BLIND_K_ROBUSTNESS.png', bottom=0.2,
       foot='k grid {2,3,5,8,10,15,20}; selections written before any HS label was read. 1,600-image overlapping subsets of the pilot; not held-out.')

# ---------------- 3. AMI vs ARI heatmaps
A = P3 / 'ami_vs_ari'; summ = pd.read_csv(A / 'summary.csv')
fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.4), gridspec_kw={'wspace': 0.42})
for ax, meth in zip(axes, ['spectral_knn', 'agglomerative_average_cosine']):
    C = np.load(A / f'contingency_siglip_{meth}.npy'); cls = pd.read_csv(A / f'contingency_siglip_{meth}_classes.csv').iloc[:, 0].astype(str).tolist()
    tot = C.sum(1); keep = np.argsort(-tot)[:12]
    order = np.argsort(-C.sum(0))
    Cs = C[:, order]; share = Cs / Cs.sum(0)
    M = share[keep]
    im = ax.imshow(M.T, cmap='Greens', vmin=0, vmax=1, aspect='auto')
    for (ci, cj), v in np.ndenumerate(M.T):
        if v >= 0.15: ax.text(cj, ci, f'{v:.0%}'.replace('%', ''), ha='center', va='center', fontsize=8, color='white' if v > 0.55 else INK)
    ax.set_xticks(range(len(keep))); ax.set_xticklabels([f'{cls[i]} {HS3.get(cls[i], "")}' for i in keep], rotation=60, ha='right', fontsize=8.5)
    sz = Cs.sum(0)
    ax.set_yticks(range(Cs.shape[1])); ax.set_yticklabels([f'C{j + 1} · n={int(n)}' for j, n in enumerate(sz)], fontsize=9)
    sm = summ[(summ.encoder == 'siglip') & (summ.method == meth)].iloc[0]
    ax.set_title(f'{MC[meth][0]}\nAMI {sm.HS3_AMI:.3f} · ARI {sm.HS3_ARI:.3f} · pair precision {sm.pair_precision:.2f} · recall {sm.pair_recall:.2f}', loc='left', fontsize=11)
    ax.grid(False)
fig.subplots_adjust(right=0.9); cax = fig.add_axes([0.925, 0.3, 0.012, 0.5]); cb = fig.colorbar(im, cax=cax); cb.set_label('share of the cluster’s labelled images')
sp = summ[(summ.encoder == 'siglip') & (summ.method == 'spectral_knn')].iloc[0]; ag = summ[(summ.encoder == 'siglip') & (summ.method == 'agglomerative_average_cosine')].iloc[0]
finish(fig, 'Why AMI and ARI disagree: one large mixed cluster (SigLIP, centered + L2, k = 10, seed 0)',
       f'Spectral’s largest cluster holds {sp.largest_cluster_labelled_pct:.0f}% of labelled images ({sp.largest_cluster_n_classes_ge5pct} classes ≥5%, purity {sp.largest_cluster_purity:.2f}) and {sp.share_of_same_cluster_pairs_in_largest:.0%} of all\nsame-cluster pairs → low pair precision → lower ARI. '
       f'Its other clusters are purer → more mutual information → higher AMI.',
       'P3_AMI_VS_ARI.png', bottom=0.26, top=0.8,
       foot='Rows: clusters by labelled size (n = labelled HS3 images); columns: 12 most frequent HS3 classes (1,549 labelled rows). Purity/precision descriptive. Pilot, not held-out.')

# ---------------- 4. Preprocessing ablation
pp = pd.read_csv(W6 / 'P3_PREPROCESSING_RESULTS.csv'); p10 = pp[pp.requested_k == 10]
REPS = ['l2', 'centered_l2', 'pca50', 'pca50_l2', 'pca100_l2']
RLAB = ['raw L2', 'centered\n+ L2', 'PCA50', 'PCA50\n+ L2', 'PCA100\n+ L2']
fig, axes = plt.subplots(2, 2, figsize=(13.33, 8.2), sharex=True)
x = np.arange(len(REPS)); w = 0.26
for j, enc in enumerate(['clip', 'siglip']):
    for i, metric in enumerate(['AMI', 'ARI']):
        ax = axes[i, j]
        for o, (m, (lab, col, mk)) in zip([-1, 0, 1], MC.items()):
            g = p10[(p10.encoder == enc) & (p10.method == m)].set_index('representation').loc[REPS]
            sd = g[f'HS3_{metric}_sd'].fillna(0)
            ax.bar(x + o * (w + 0.02), g[f'HS3_{metric}_mean'], w, yerr=sd, capsize=2.5, color=col, label=lab, error_kw={'ecolor': INK2, 'lw': 1})
            for xi, v in zip(x + o * (w + 0.02), g[f'HS3_{metric}_mean']):
                ax.text(xi, v + 0.008, f'{v:.2f}', ha='center', fontsize=7.5, color=INK2)
        ax.set_title(f'{ENC[enc]} · HS3 {metric} · k = 10', loc='left'); ax.set_ylim(0, 0.62 if metric == 'AMI' else 0.45)
        if j == 0: ax.set_ylabel(f'HS3 {metric}')
    axes[1, j].set_xticks(x); axes[1, j].set_xticklabels(RLAB)
axes[0, 0].legend(loc='upper left', fontsize=9.5, ncol=3, bbox_to_anchor=(0, 1.0))
finish(fig, 'Preprocessing ablation: the effect of centering depends on the algorithm',
       'Raw L2: agglomerative collapses (AMI ≈ 0.03). Centering rescues it and helps Spectral, but slightly lowers k-Means. PCA50 without re-normalisation hurts Spectral.',
       'P3_PREPROCESSING_ABLATION.png',
       foot='Same 2,000 rows and masks (HS3 n = 1,549), k = 10, seeds 0–4 (mean ± SD); phase-2 representations. Global centering only (per-museum not run). Pilot, not held-out.')

# ---------------- 5. Cluster diagnostics
fig, axes = plt.subplots(1, 3, figsize=(13.33, 6.4), gridspec_kw={'width_ratios': [1, 1, 1.1], 'wspace': 0.3})
for ax, enc in zip(axes[:2], ['clip', 'siglip']):
    for o, (m, (lab, col, mk)) in zip([-1, 0, 1], MC.items()):
        g = p10[(p10.encoder == enc) & (p10.method == m)].set_index('representation').loc[REPS]
        ax.bar(x + o * (w + 0.02), g.largest_cluster_pct_max, w, color=col, label=lab)
    ax.axhline(10, color=INK2, ls='--', lw=1.2); ax.text(-0.45, 6.5, 'balanced = 10%', fontsize=8.5, color=INK2, ha='left')
    ax.set_xticks(x); ax.set_xticklabels(RLAB, fontsize=9); ax.set_ylim(0, 105)
    ax.set_title(f'{ENC[enc]} · largest cluster, k = 10', loc='left')
axes[0].set_ylabel('largest cluster, % of 2,000 images (worst seed)'); axes[0].legend(fontsize=9, loc='upper right')
ax = axes[2]
for m, (lab, col, mk) in MC.items():
    p = np.load(W6 / f'experiments/assignments/siglip__centered_l2__{m}__k10__s0.npy')
    s_ = np.sort(np.bincount(p))[::-1] / len(p) * 100
    ax.plot(np.arange(1, len(s_) + 1), s_, marker=mk, color=col, lw=2, ms=7, mec=SURF, label=lab)
ax.set_xlabel('cluster rank (largest first)'); ax.set_ylabel('% of images'); ax.set_xticks(range(1, 11))
ax.set_title('SigLIP · centered + L2 · cluster sizes', loc='left'); ax.legend(fontsize=9)
finish(fig, 'Cluster-size diagnostics: collapse and imbalance',
       'Raw-L2 agglomerative puts ~99% of images in one cluster. Spectral keeps one dominant cluster (28–52%) under every representation; k-Means stays the most balanced.',
       'P3_CLUSTER_DIAGNOSTICS.png',
       foot='k = 10; worst of 5 seeds for k-Means/Spectral; right panel seed 0. Same 2,000 rows. Development pilot, not held-out.')

# ---------------- 6. Mixed-cluster examples (local thumbnails, CC BY-NC-SA 4.0)
ex = pd.read_csv(A / 'mixed_cluster_examples.csv', dtype=str)
clusters = ex.cluster.unique()
fig = plt.figure(figsize=(13.33, 5.4))
n_per_row = 12
for ri, c in enumerate(clusters):
    g = ex[ex.cluster == c].reset_index(drop=True)
    for i, r in g.iterrows():
        ax = fig.add_axes([0.01 + i * (0.98 / n_per_row), 0.475 - ri * 0.405, 0.98 / n_per_row - 0.006, 0.255])
        im = ImageOps.exif_transpose(Image.open(ROOT / r.image_file)).convert('RGB'); im.thumbnail((300, 300))
        canvas = Image.new('RGB', (300, 300), 'white'); canvas.paste(im, ((300 - im.width) // 2, (300 - im.height) // 2))
        ax.imshow(canvas); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(GRID)
        t = r.title if len(r.title) <= 20 else r.title[:19] + '…'
        ax.set_title(f'{r.hs3} · {HS3.get(r.hs3, "")[:14]}', fontsize=7.5, color=INK, pad=3)
        ax.set_xlabel(t, fontsize=7.2, color=INK2, labelpad=2)
    first = g.iloc[0]
    comp = pd.read_csv(A / 'cluster_profiles.csv', dtype={'top1_class': str, 'top2_class': str, 'top3_class': str}); cp = comp[(comp.encoder == 'siglip') & (comp.method == 'spectral_knn')]
    prof = cp.iloc[[k for k in range(len(cp)) if True]]
    lab_ = cp.sort_values('cluster').reset_index(drop=True)
    row = lab_.iloc[int(np.where(np.unique(np.load(W6 / 'experiments/assignments/siglip__centered_l2__spectral_knn__k10__s0.npy')) == int(c))[0][0])]
    fig.text(0.01, 0.80 - ri * 0.405, f'Heatmap C{ri + 1} (cluster id {int(c)}): {int(row.size_all_rows)} images ({int(row.labelled)} labelled), {int(row.n_classes_present)} HS3 classes — '
             f'top {row.top1_class} {row.top1_share:.0%}, {row.top2_class} {row.top2_share:.0%}, {row.top3_class} {row.top3_share:.0%}',
             fontsize=11, fontweight='bold')
fig.suptitle('Mixed clusters: visual form over acoustic function (SigLIP + Spectral, k = 10, seed 0)', x=0.01, ha='left', fontsize=15, fontweight='bold')
fig.text(0.01, 0.905, 'Images sampled at random (seed 20261116) within the top-3 HS3 classes of the two largest clusters whose top class is < 60%. Titles and HS3 codes as catalogued.', fontsize=10.5, color=INK2)
fig.text(0.01, 0.01, 'Photos: MIMO / Europeana partner museums, CC BY-NC-SA 4.0 (record IDs and museums in p3_experiments/ami_vs_ari/mixed_cluster_examples.csv). Pilot, not held-out.', fontsize=8.5, color=INK2)
fig.savefig(FIG / 'P3_MIXED_CLUSTER_EXAMPLES.png', dpi=220); plt.close(fig); print('wrote P3_MIXED_CLUSTER_EXAMPLES.png')
