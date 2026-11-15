"""Dev-benchmark-only figures for the final Week 6 deck. Values come from ../final_claims.json (recomputed from assignments)."""
import json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data/.matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIN = ROOT / 'reports/week6/presentation/final'; FIG = FIN / 'figures'; FIG.mkdir(exist_ok=True)
C = json.loads((FIN / 'final_claims.json').read_text()); S = C['scores']; D = C['dataset']
SURF, INK, INK2, GRID = '#ffffff', '#0b0b0b', '#52514e', '#e4e3df'
BK = [('clip', 'CLIP ViT-B/32', '#2a78d6'), ('siglip', 'SigLIP ViT-B/16', '#eb6834'), ('dinov2', 'DINOv2-B/14 (224 px)', '#1baf7a')]
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': INK2, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 13, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'axes.axisbelow': True, 'legend.frameon': False})
CAP = f"Development benchmark: {D['dev']:,} objects, {D['hs3_classes']} HS3 classes · HS3 score · k = 10 (predeclared)"


def save(fig, name):
    fig.savefig(FIG / name, dpi=220, bbox_inches='tight'); plt.close(fig); print('wrote', name)


def g(b, rep, m, k='k10'):
    return S[f'{b}|{rep}|{m}|{k}']


# 1 backbones: k-Means and agglomerative, AMI (left) and ARI (right)
fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
for ax, metric in zip(axes, ['AMI', 'ARI']):
    x = np.arange(2); w = 0.26
    for i, (b, lab, col) in enumerate(BK):
        vals = [g(b, 'centered_l2', m)[metric] for m in ('kmeans', 'agglo')]
        sds = [g(b, 'centered_l2', 'kmeans')['AMI_sd'], 0] if metric == 'AMI' else None
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.95, color=col, label=lab, yerr=sds, capsize=3, error_kw={'ecolor': INK2, 'lw': 1})
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 0.016, f'{v:.3f}', ha='center', fontsize=10.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(['k-Means', 'Agglomerative\n(average, cosine)']); ax.set_ylim(0, 0.65 if metric == 'AMI' else 0.5)
    ax.set_title(f'HS3 {metric} · centered + L2', loc='left', fontsize=14); ax.set_ylabel(f'HS3 {metric}')
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=3, fontsize=11)
fig.text(0.0, -0.07, CAP + '. k-Means: mean ± SD over 5 seeds; agglomerative deterministic.', fontsize=10, color=INK2)
save(fig, 'FINAL_RQ1_BACKBONES.png')

# 2 AMI vs ARI scatter, all five methods x three backbones
MK = {'kmeans': ('o', 'k-Means'), 'agglo': ('s', 'Agglomerative'), 'spectral': ('D', 'Spectral'), 'gmm': ('^', 'GMM'), 'hdbscan': ('P', 'HDBSCAN*')}
fig, ax = plt.subplots(figsize=(9.5, 6.2))
for b, lab, col in BK:
    for m, (mk, ml) in MK.items():
        r = g(b, 'centered_l2', m, 'kNone' if m == 'hdbscan' else 'k10')
        ax.scatter(r['AMI'], r['ARI'], marker=mk, s=150, color=col, edgecolors='white', linewidths=1.2, zorder=3)
for b, txt, dx, dy in [('dinov2', 'DINOv2-B + Agglomerative\nAMI 0.561 · ARI 0.417\nlargest cluster 21.6%', -0.075, 0.012),
                       ('siglip', 'SigLIP + Spectral\nAMI 0.559 · ARI 0.311\nlargest cluster 34.7%', -0.02, -0.105),
                       ('dinov2s', 'DINOv2-B + Spectral\nAMI 0.476 · ARI 0.165\nlargest cluster 50.1%', 0.012, -0.075)]:
    key = ('dinov2', 'agglo') if b == 'dinov2' else (('siglip', 'spectral') if b == 'siglip' else ('dinov2', 'spectral'))
    r = g(key[0], 'centered_l2', key[1])
    assert abs(r['AMI'] - float(txt.split('AMI ')[1][:5])) < 1e-9 and abs(r['ARI'] - float(txt.split('ARI ')[1][:5])) < 1e-9
    ax.annotate(txt, (r['AMI'], r['ARI']), xytext=(r['AMI'] + dx, r['ARI'] + dy), fontsize=10.5, color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
ax.set_xlabel('HS3 AMI (information shared with expert classes)'); ax.set_ylabel('HS3 ARI (pairwise agreement)')
ax.set_xlim(0.37, 0.6); ax.set_ylim(0.05, 0.47)
h1 = [plt.Line2D([], [], marker='o', ls='', color=col, ms=10, label=lab) for _, lab, col in BK]
h2 = [plt.Line2D([], [], marker=mk, ls='', color=INK2, ms=9, label=ml) for mk, ml in MK.values()]
l1 = ax.legend(handles=h1, loc='upper left', fontsize=10.5); ax.add_artist(l1); ax.legend(handles=h2, loc='lower right', fontsize=10.5, ncol=1)
fig.text(0.0, -0.02, CAP + '; centered + L2. *HDBSCAN chooses its own k; noise counted as one cluster.', fontsize=10, color=INK2)
save(fig, 'FINAL_RQ1_AMI_VS_ARI.png')

# 3 preprocessing, agglomerative
REPS = [('l2', 'raw L2'), ('centered_l2', 'centered + L2'), ('pca50_l2', 'PCA50 + L2'), ('pca100_l2', 'PCA100 + L2')]
fig, ax = plt.subplots(figsize=(11, 5.0))
x = np.arange(len(REPS)); w = 0.26
for i, (b, lab, col) in enumerate(BK):
    vals = [g(b, r, 'agglo')['AMI'] for r, _ in REPS]
    ax.bar(x + (i - 1) * w, vals, w * 0.95, color=col, label=lab)
    for j, (xi, v) in enumerate(zip(x + (i - 1) * w, vals)):
        ax.text(xi, v + 0.008, f'{v:.3f}', ha='center', fontsize=10.5)
    lg = g(b, 'l2', 'agglo')['largest_pct_max']
    ax.text(x[0] + (i - 1) * w, max(g(b, 'l2', 'agglo')['AMI'], 0) + 0.05, f'{lg:.1f}%\nin one\ncluster', ha='center', fontsize=9, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in REPS]); ax.set_ylim(0, 0.66); ax.set_ylabel('HS3 AMI')
ax.set_title('Agglomerative clustering (average linkage, cosine) · k = 10', loc='left', fontsize=14); ax.legend(loc='upper left', fontsize=11, ncol=3, bbox_to_anchor=(0.18, 1.0))
fig.text(0.0, -0.02, CAP + '. Global centering = subtract the mean embedding of the development objects; no labels used.', fontsize=10, color=INK2)
save(fig, 'FINAL_RQ2_PREPROCESSING.png')

# 4 HS3 class distribution of the development benchmark
NAMES = {'421': 'flutes (edge)', '422': 'reedpipes', '321': 'lutes', '423': 'lip-reed (brass)', '111': 'struck idiophones', '112': 'rattles, scrapers',
         '211': 'struck drums', '412': 'free reeds', '314': 'board zithers', '322': 'harps', '531': 'electrophones', '513': 'electrophones', '212': 'rattle drums'}
dv = sorted(D['dev_hs3'].items(), key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(6.2, 5.6))
ax.barh([f'{k} {NAMES.get(k, "")}' for k, _ in dv], [v for _, v in dv], color='#1F8A3B')
for i, (_, v) in enumerate(dv):
    ax.text(v + 30, i, f'{v:,}', va='center', fontsize=10)
ax.set_xlim(0, max(v for _, v in dv) * 1.22); ax.set_xlabel('objects'); ax.grid(axis='y', visible=False); ax.tick_params(axis='y', labelsize=10.5)
ax.set_title(f"HS3 classes, development benchmark (n = {D['dev']:,})", loc='left', fontsize=13)
save(fig, 'FINAL_DEV_HS3_CLASSES.png')
