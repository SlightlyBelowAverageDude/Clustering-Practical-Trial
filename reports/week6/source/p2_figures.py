"""Week 6 / P2 figures, drawn only from saved CSVs under reports/week6/."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data/phase2/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W6 = ROOT / 'reports/week6'; EXP = W6 / 'experiments'; FIG = W6 / 'figures'; FIG.mkdir(exist_ok=True)
mc = pd.read_csv(W6 / 'P2_METHOD_COMPARISON.csv')
curves = pd.read_csv(EXP / 'blind_k_curves.csv')
sel = pd.read_csv(EXP / 'blind_k_selection.csv')
con = pd.read_csv(EXP / 'paired_contrasts.csv')

# validated reference palette (light mode), fixed slot order
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
METHODS = [('kmeans', 'k-Means', '#2a78d6', 'o'), ('agglomerative_average_cosine', 'Agglomerative (avg, cosine)', '#eb6834', 's'),
           ('gmm_diag', 'GMM (diagonal)', '#1baf7a', '^'), ('spectral_knn', 'Spectral (15-NN)', '#eda100', 'D')]
HDB = ('#e87ba4', 'P')
ENC = {'clip': ('CLIP ViT-B/32 (512-d)', '#2a78d6'), 'siglip': ('SigLIP ViT-B/16 (768-d)', '#eb6834')}
KS = [2, 3, 5, 8, 10, 15, 20]
FOOT = ('MIMO pilot: 2,000 images clustered (216 museums); HS3 scored on 1,549 labeled rows, 22 classes. '
        'Mean ± SD over 5 seeds (agglomerative and HDBSCAN deterministic). Development-set result, not held-out.')
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'font.size': 11,
                     'axes.titlesize': 12.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True, 'legend.frameon': False})
full = mc[mc.scoring.eq('all_rows')]
hdb = mc[mc.scoring.eq('noise_as_cluster')]


def finish(fig, title, sub, name):
    fig.suptitle(title, x=0.01, ha='left', fontsize=15, fontweight='bold')
    fig.text(0.01, 0.925, sub, ha='left', fontsize=10.5, color=INK2)
    fig.text(0.01, 0.01, FOOT, ha='left', fontsize=8.5, color=INK2)
    fig.subplots_adjust(top=0.84, bottom=0.14); fig.savefig(FIG / name, dpi=220); plt.close(fig); print('wrote', name)


def ami_vs_k(ax, enc, rep, mark_sel=False):
    for m, lab, col, mk in METHODS:
        g = full[(full.encoder == enc) & (full.representation == rep) & (full.method == m)].sort_values('requested_k')
        ax.errorbar(g.requested_k, g.HS3_AMI_mean, yerr=g.HS3_AMI_sd.fillna(0), color=col, marker=mk, ms=7, lw=2, capsize=3,
                    label=lab, mec=SURF, mew=1.2)
        if mark_sel:
            k = int(sel[(sel.encoder == enc) & (sel.representation == rep) & (sel.method == m) & (sel.criterion == 'silhouette_cosine')].selected_k.iloc[0])
            y = g.set_index('requested_k').HS3_AMI_mean[k]
            ax.scatter([k], [y], s=260, facecolors='none', edgecolors=col, linewidths=2.2, zorder=5)
    ax.axvline(10, color=INK2, ls=':', lw=1.2); ax.set_xscale('log'); ax.set_xticks(KS); ax.set_xticklabels(KS)
    ax.minorticks_off(); ax.set_xlabel('number of clusters k'); ax.set_title(f'{ENC[enc][0]} · {rep}', loc='left')


# ---------- Figure 1: method comparison ----------
fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.4), sharey=True)
for ax, enc in zip(axes, ['clip', 'siglip']):
    ami_vs_k(ax, enc, 'centered_l2')
    h = hdb[(hdb.encoder == enc) & (hdb.representation == 'centered_l2')]
    for r in h.itertuples():
        ax.scatter(r.actual_k_mean + 1, r.HS3_AMI_mean, marker=HDB[1], s=110, color=HDB[0], edgecolors=SURF, zorder=6,
                   label='HDBSCAN (noise = own cluster)' if r.min_cluster_size == 20 else None)
        ax.annotate(f'mcs={int(r.min_cluster_size)}\n{r.noise_pct_mean:.0f}% noise', (r.actual_k_mean + 1, r.HS3_AMI_mean),
                    xytext=(8, 10 if r.min_cluster_size == 20 else -22), textcoords='offset points', fontsize=8.5, color=INK2)
    ax.text(10.3, 0.02, 'fixed-k anchor k=10', fontsize=8.5, color=INK2)
axes[0].set_ylabel('HS3 AMI (higher = closer to expert classes)'); axes[0].set_ylim(0, 0.6)
axes[1].legend(loc='upper left', fontsize=9.5)
finish(fig, 'Clustering method comparison at HS level 3',
       'Representation: centered + L2 (phase-2 anchor). Spectral (15-NN) is highest at k = 10 and 15 on both backbones; HDBSCAN plotted at its own k (+1 for noise).',
       'P2_METHOD_COMPARISON.png')

# ---------- Figure 2: backbone comparison ----------
fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.4), sharey=True)
for ax, rep in zip(axes, ['centered_l2', 'pca50_l2']):
    labels = ['k-Means', 'Agglomerative\n(avg, cosine)', 'GMM\n(diagonal)', 'Spectral\n(15-NN)', 'HDBSCAN\nmcs=20, noise\n= own cluster']
    x = np.arange(len(labels)); w = 0.38
    for i, enc in enumerate(['clip', 'siglip']):
        vals, sds = [], []
        for m, *_ in METHODS:
            r = full[(full.encoder == enc) & (full.representation == rep) & (full.method == m) & (full.requested_k == 10)].iloc[0]
            vals.append(r.HS3_AMI_mean); sds.append(0 if np.isnan(r.HS3_AMI_sd) else r.HS3_AMI_sd)
        r = hdb[(hdb.encoder == enc) & (hdb.representation == rep) & (hdb.min_cluster_size == 20)].iloc[0]
        vals.append(r.HS3_AMI_mean); sds.append(0)
        b = ax.bar(x + (i - 0.5) * (w + 0.02), vals, w, yerr=sds, capsize=3, color=ENC[enc][1], label=ENC[enc][0],
                   error_kw={'ecolor': INK2, 'lw': 1})
        for xi, v in zip(x + (i - 0.5) * (w + 0.02), vals):
            ax.text(xi, v + 0.012, f'{v:.3f}', ha='center', fontsize=8, color=INK2)
    c = con[(con.contrast == 'siglip - clip') & (con.representation == rep)]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_title(f'{rep} · k=10 · SigLIP > CLIP in {int((c.delta_HS3_AMI_mean > 0).sum())}/{len(c)} method×k cells', loc='left')
axes[0].set_ylabel('HS3 AMI'); axes[0].set_ylim(0, 0.65); axes[0].legend(loc='upper left')
finish(fig, 'Backbone comparison: CLIP vs SigLIP on identical rows and masks',
       'Bars: fixed predeclared k=10 (HDBSCAN picks its own k), SD over 5 seeds. Cell counts: k ∈ {2,…,20} × 4 methods. DINOv2 not evaluated (no local weights).',
       'P2_BACKBONE_COMPARISON.png')

# ---------- Figure 3: blind k ----------
fig, axes = plt.subplots(2, 2, figsize=(13.33, 8.6), sharex=True)
for j, enc in enumerate(['clip', 'siglip']):
    ax = axes[0, j]
    for m, lab, col, mk in METHODS:
        g = curves[(curves.encoder == enc) & (curves.representation == 'centered_l2') & (curves.method == m)].sort_values('requested_k')
        ax.errorbar(g.requested_k, g.silhouette_cosine_mean, yerr=g.silhouette_cosine_sd.fillna(0), color=col, marker=mk, ms=7, lw=2,
                    capsize=3, label=lab, mec=SURF, mew=1.2)
        k = int(sel[(sel.encoder == enc) & (sel.representation == 'centered_l2') & (sel.method == m) & (sel.criterion == 'silhouette_cosine')].selected_k.iloc[0])
        ax.scatter([k], [g.set_index('requested_k').silhouette_cosine_mean[k]], s=260, facecolors='none', edgecolors=col, linewidths=2.2, zorder=5)
    ax.set_xscale('log'); ax.set_xticks(KS); ax.set_xticklabels(KS); ax.minorticks_off()
    ax.set_title(f'{ENC[enc][0]} · centered_l2 — label-free criterion', loc='left'); ax.axvline(10, color=INK2, ls=':', lw=1.2)
    ami_vs_k(axes[1, j], enc, 'centered_l2', mark_sel=True)
    axes[1, j].set_title(f'{ENC[enc][0]} — HS3 AMI evaluated after selection', loc='left')
axes[0, 0].set_ylabel('cosine silhouette (mean ± SD)'); axes[1, 0].set_ylabel('HS3 AMI'); axes[1, 0].set_ylim(0.08, 0.6); axes[1, 1].set_ylim(0.08, 0.6)
axes[0, 1].legend(loc='lower right', fontsize=9)
for ax in axes[1]: ax.axhline(np.nan)
finish(fig, 'Blind selection of k with silhouette',
       'Ring = k chosen by maximum silhouette without labels (selected k = 5–10 here). HS3 AMI usually peaks later, at k = 15–20. HS3 has 22 classes (13 with ≥10 images).',
       'P2_BLIND_K.png')

# ---------- Figure 4: diagnostics ----------
fig, axes = plt.subplots(1, 3, figsize=(13.33, 6.4), gridspec_kw={'width_ratios': [1, 1, 1.05], 'wspace': 0.45})
for ax, enc in zip(axes[:2], ['clip', 'siglip']):
    for m, lab, col, mk in METHODS:
        g = full[(full.encoder == enc) & (full.representation == 'centered_l2') & (full.method == m)].sort_values('requested_k')
        ax.plot(g.requested_k, g.largest_cluster_pct_max, color=col, marker=mk, ms=7, lw=2, label=lab, mec=SURF, mew=1.2)
    ax.plot(KS, [100 / k for k in KS], color=INK2, ls='--', lw=1.2, label='perfectly balanced (100/k)')
    ax.axhline(18.46, color=INK2, ls=':', lw=1.2); ax.text(2.05, 19.5, 'largest HS3 class: 18.5%', fontsize=8.5, color=INK2)
    ax.set_xscale('log'); ax.set_xticks(KS); ax.set_xticklabels(KS); ax.minorticks_off(); ax.set_ylim(0, 100)
    ax.set_xlabel('number of clusters k'); ax.set_title(f'{ENC[enc][0]} · centered_l2', loc='left')
axes[0].set_ylabel('largest cluster, % of 2,000 images (worst seed)'); axes[1].legend(fontsize=8.5, loc='upper right')
ax = axes[2]
h = hdb.sort_values(['encoder', 'representation', 'min_cluster_size'])
lbl = [f"{e.upper()} {r.replace('_l2', '')}\nmcs={int(s)}" for e, r, s in zip(h.encoder, h.representation, h.min_cluster_size)]
y = np.arange(len(h))
ax.barh(y, h.noise_pct_mean, color=HDB[0], height=0.62, label='noise (unassigned)')
ax.barh(y, h.largest_cluster_pct_mean, left=h.noise_pct_mean, color='#6250d6', height=0.62, label='largest cluster')
for yi, n, l, k in zip(y, h.noise_pct_mean, h.largest_cluster_pct_mean, h.actual_k_mean):
    ax.text(n + l + 1, yi, f'{n:.0f}% noise · k={int(k)}', va='center', fontsize=8.5, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=8.5); ax.set_xlim(0, 135); ax.invert_yaxis()
ax.set_title('HDBSCAN: noise and largest cluster', loc='left'); ax.legend(fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2); ax.grid(axis='y', visible=False)
finish(fig, 'Cluster diagnostics: imbalance and noise',
       'centered_l2: at k ≤ 3 Spectral puts 76–89% of images in one cluster; at k = 10 all k-requiring methods stay ≤ 39%. HDBSCAN leaves 59–70% as noise.',
       'P2_CLUSTER_DIAGNOSTICS.png')
