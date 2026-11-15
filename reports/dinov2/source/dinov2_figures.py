"""DINOv2 / three-backbone figures from reports/dinov2/{pilot,dev}/summary.csv and pilot/linear_probe.csv."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data/.matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = ROOT / 'reports/dinov2'; FIG = D / 'figures'; FIG.mkdir(exist_ok=True)
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': INK2, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'axes.axisbelow': True, 'legend.frameon': False})
BACK = {'clip': ('CLIP ViT-B/32', '#2a78d6'), 'siglip': ('SigLIP ViT-B/16', '#eb6834'), 'dinov2_s_518_L1': ('DINOv2-S/14 (518)', '#1baf7a'),
        'dinov2_b_518_L1': ('DINOv2-B/14 (518)', '#eda100'), 'dinov2_l_224_L1': ('DINOv2-L/14 (224)', '#e87ba4'), 'dinov2_b_224_L1': ('DINOv2-B/14 (224)', '#eda100')}
METH = [('kmeans', 'k-Means'), ('agglomerative_average_cosine', 'Agglom.'), ('spectral_knn', 'Spectral'), ('gmm_diag', 'GMM'), ('hdbscan', 'HDBSCAN*')]
S = pd.read_csv(D / 'pilot/summary.csv')
S = S[(S.scoring != 'assigned_only')]
FOOT = 'MIMO pilot: 2,000 images, HS3 n = 1,549 (22 classes); k = 10; mean of 5 seeds (agglomerative/HDBSCAN deterministic). *HDBSCAN noise = own cluster. Development pilot, not held-out.'


def finish(fig, title, sub, name, foot=FOOT, bottom=0.16):
    fig.suptitle(title, x=0.01, ha='left', fontsize=15, fontweight='bold'); fig.text(0.01, 0.925, sub, fontsize=10.5, color=INK2)
    fig.text(0.01, 0.01, foot, fontsize=8.5, color=INK2); fig.subplots_adjust(top=0.84, bottom=bottom); fig.savefig(FIG / name, dpi=200); plt.close(fig); print('wrote', name)


def get(var, rep, meth, metric='AMI_mean', label='hs3', df=None):
    df = S if df is None else df
    r = df[(df.variant == var) & (df.representation == rep) & (df.method == meth) & (df.label == label)]
    if meth != 'hdbscan': r = r[r.k == 10]
    return r[metric].iloc[0] if len(r) else np.nan


# 1 three backbones x methods (centered_l2), AMI and ARI
backs = ['clip', 'siglip', 'dinov2_s_518_L1', 'dinov2_b_518_L1', 'dinov2_l_224_L1']
fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.2), sharey=False)
x = np.arange(len(METH)); w = 0.16
for ax, metric in zip(axes, ['AMI', 'ARI']):
    for i, b in enumerate(backs):
        v = [get(b, 'centered_l2', m, f'{metric}_mean') for m, _ in METH]
        e = [get(b, 'centered_l2', m, f'{metric}_sd') for m, _ in METH]
        ax.bar(x + (i - 2) * w, v, w * 0.95, yerr=np.nan_to_num(e), capsize=2, color=BACK[b][1], label=BACK[b][0], error_kw={'ecolor': INK2, 'lw': 0.8})
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in METH]); ax.set_title(f'HS3 {metric} · centered + L2 · k = 10', loc='left'); ax.set_ylabel(f'HS3 {metric}')
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper right', bbox_to_anchor=(0.99, 0.93), ncol=5, fontsize=9)
best = {b: get(b, 'centered_l2', 'kmeans') for b in backs}
finish(fig, 'CLIP vs SigLIP vs DINOv2 on identical pilot rows', 'k-Means HS3 AMI: ' + ', '.join(f'{BACK[b][0].split(" (")[0]} {v:.3f}' for b, v in best.items()), 'DINOV2_BACKBONE_COMPARISON.png')

# 2 model scaling and layers (224 px), plus resolution
fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.0), gridspec_kw={'width_ratios': [1.4, 1]})
ax = axes[0]
sizes = ['s', 'b', 'l']; cols = {'kmeans': '#2a78d6', 'agglomerative_average_cosine': '#eb6834', 'spectral_knn': '#eda100'}
mk = {1: 'o', 2: 's', 4: '^'}
for meth, col in cols.items():
    for l in (1, 2, 4):
        v = [get(f'dinov2_{s}_224_L{l}', 'centered_l2', meth) for s in sizes]
        ax.plot([0, 1, 2], v, marker=mk[l], color=col, lw=2 if l == 1 else 1, ls='-' if l == 1 else ('--' if l == 2 else ':'), ms=7,
                label=f'{dict(METH)[meth]} · layer −{l}')
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['ViT-S/14\n(22 M, 384-d)', 'ViT-B/14\n(87 M, 768-d)', 'ViT-L/14\n(304 M, 1024-d)'])
ax.set_ylabel('HS3 AMI'); ax.set_title('Model size × layer (224 px, centered + L2, k = 10)', loc='left'); ax.legend(fontsize=7.5, ncol=3, loc='center', bbox_to_anchor=(0.5, 0.3))
ax = axes[1]
for i, s in enumerate(['s', 'b']):
    for j, (meth, col) in enumerate(cols.items()):
        a, b = get(f'dinov2_{s}_224_L1', 'centered_l2', meth), get(f'dinov2_{s}_518_L1', 'centered_l2', meth)
        ax.plot([i * 3 + j * 0.8, i * 3 + j * 0.8 + 0.5], [a, b], color=col, marker='o', lw=2, label=dict(METH)[meth] if i == 0 else None)
ax.set_xticks([1.05, 4.05]); ax.set_xticklabels(['ViT-S/14', 'ViT-B/14'])
ax.set_title('Input resolution, layer −1\n(each segment: 224 px → 518 px)', loc='left'); ax.set_ylabel('HS3 AMI'); ax.legend(fontsize=8.5)
finish(fig, 'RQ3: DINOv2 model size, layer depth and resolution', 'Same 2,000 pilot rows and masks; CLS token with final LayerNorm; layer −1 = pooled output.', 'DINOV2_SCALING.png')

# 3 preprocessing x backbone (k-Means, agglomerative, spectral)
REPS = ['l2', 'centered_l2', 'pca50_l2', 'pca100_l2']
fig, axes = plt.subplots(1, 3, figsize=(13.33, 5.6), sharey=True)
for ax, (meth, ml) in zip(axes, METH[:3]):
    for b in backs:
        ax.plot(range(4), [get(b, r, meth) for r in REPS], marker='o', color=BACK[b][1], lw=2, label=BACK[b][0])
    ax.set_xticks(range(4)); ax.set_xticklabels(['raw L2', 'centered\n+L2', 'PCA50\n+L2', 'PCA100\n+L2']); ax.set_title(ml, loc='left')
axes[0].set_ylabel('HS3 AMI'); axes[2].legend(fontsize=8.5, loc='lower right')
finish(fig, 'Preprocessing effect per backbone (k = 10)', 'Raw-L2 agglomerative collapse and its rescue by centering, checked for DINOv2.', 'DINOV2_PREPROCESSING.png')

# 4 linear probe vs clustering (supervised reference, separate)
P = pd.read_csv(D / 'pilot/linear_probe.csv')
order = ['clip', 'siglip'] + [f'dinov2_{s}_{r}_L{l}' for s in 'sbl' for r in (224, 518) for l in (1, 2, 4) if not (s == 'l' and r == 518)]
P = P.set_index('variant').loc[[o for o in order if o in set(P.variant)]].reset_index()
fig, ax = plt.subplots(figsize=(13.33, 5.6))
xx = np.arange(len(P))
ax.bar(xx - 0.2, P.probe_AMI, 0.4, color='#b9b8b2', label='supervised linear probe (5-fold, out-of-fold) — reference only')
ax.bar(xx + 0.2, P.kmeans_k10_AMI_same_rows_mean, 0.4, color='#2a78d6', label='unsupervised k-Means k = 10 (same rows)')
ax.set_xticks(xx); ax.set_xticklabels([v.replace('dinov2_', 'D2-').replace('_L', ' L') for v in P.variant], rotation=45, ha='right', fontsize=9)
ax.set_ylabel(f'HS3 AMI ({int(P.classes.iloc[0])} classes, {int(P.rows.iloc[0]):,} rows)'); ax.legend(fontsize=9.5, loc='upper left')
finish(fig, 'Probe gap: what the embedding contains vs what clustering recovers', 'Probe and clustering are reported separately; probe labels never enter clustering.', 'DINOV2_PROBE_GAP.png',
       foot='HS3 classes with ≥10 labelled pilot images; LogisticRegression(C=0.5) on centered + L2 features, StratifiedKFold(5). Pilot, not held-out.', bottom=0.26)

# 5 dev partition
if (D / 'dev/summary.csv').exists():
    V = pd.read_csv(D / 'dev/summary.csv'); V = V[V.scoring != 'assigned_only']
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.0))
    dbacks = ['clip', 'siglip', 'dinov2_b_224_L1']
    for ax, metric in zip(axes, ['AMI', 'ARI']):
        for i, b in enumerate(dbacks):
            v = [get(b, 'centered_l2', m, f'{metric}_mean', df=V) for m, _ in METH]
            ax.bar(x + (i - 1) * 0.26, v, 0.25, color=BACK[b][1], label=BACK[b][0])
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in METH]); ax.set_title(f'HS3 {metric} · centered + L2 · k = 10', loc='left')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper right', bbox_to_anchor=(0.99, 0.93), ncol=3, fontsize=9.5)
    finish(fig, 'Development partition (12,000 benchmark objects, 13 HS3 classes)', 'Larger, cleaner benchmark: same pipeline; scores are NOT comparable to pilot numbers (different rows and class sets).', 'DINOV2_DEV_PARTITION.png',
           foot='Development partition of the reconstructed 27,347-object benchmark; held-out never used. Silhouette on a 4,000-row sample. *HDBSCAN noise = own cluster.')
