"""Figures for the reconstructed full MIMO dataset, from reports/full_dataset/{stats.json,class_distributions.csv,museum_distribution.csv}."""
import json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'data/.matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = ROOT / 'reports/full_dataset'; FIG = D / 'figures'; FIG.mkdir(exist_ok=True)
S = json.loads((D / 'stats.json').read_text())
dist = pd.read_csv(D / 'class_distributions.csv', dtype={'class': str})
mus = pd.read_csv(D / 'museum_distribution.csv')
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': INK2, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'axes.axisbelow': True, 'legend.frameon': False})
HS1 = {'1': 'Idiophones', '2': 'Membranophones', '3': 'Chordophones', '4': 'Aerophones', '5': 'Electrophones', '(missing)': '(no HS code)'}


def finish(fig, title, sub, name):
    fig.suptitle(title, x=0.01, ha='left', fontsize=15, fontweight='bold'); fig.text(0.01, 0.925, sub, fontsize=10.5, color=INK2)
    fig.text(0.01, 0.01, f"MIMO via Europeana API (harvest snapshot verified {S['harvested_records']:,} records on 15.11.2026); built {S['built_utc']}. Counts recomputed, not copied.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(top=0.84, bottom=0.14); fig.savefig(FIG / name, dpi=200); plt.close(fig); print('wrote', name)


# 1 funnel vs reference
ref = S['reference_counts_from_plan']
stages = [('Europeana MIMO image records', S['harvested_records'], ref['original_records']), ('valid downloaded images', S['valid_images'], None),
          ('cleaned photographs', S['cleaned_photographs'], ref['cleaned_photographs']), ('cleaned, with HS code', S['clean_coded_hs1'], ref['objects_with_hs_code']),
          ('benchmark objects', S['benchmark_rows'], ref['benchmark_objects'])]
fig, ax = plt.subplots(figsize=(12, 5.6))
y = np.arange(len(stages))[::-1]
ax.barh(y + 0.18, [s[1] for s in stages], 0.34, color=BLUE, label='recomputed (P5)')
ax.barh(y - 0.18, [s[2] or 0 for s in stages], 0.34, color='#b9b8b2', label='reported in research plan')
for yi, (lab, n, r) in zip(y, stages):
    ax.text(n + 600, yi + 0.18, f'{n:,}', va='center', fontsize=9.5)
    if r: ax.text(r + 600, yi - 0.18, f'{r:,}', va='center', fontsize=9.5, color=INK2)
fig.subplots_adjust(left=0.22); ax.set_yticks(y); ax.set_yticklabels([s[0] for s in stages]); ax.set_xlim(0, 75000); ax.set_xlabel('records / objects'); ax.legend(loc='lower right'); ax.grid(axis='y', visible=False)
finish(fig, 'Full MIMO dataset: recomputed pipeline counts vs research plan',
       f"Benchmark: HS3, family and name present; classes with ≥30 objects → {S['benchmark_classes']['hs3']} HS3 classes (plan: 13).", 'FULL_DATASET_FUNNEL.png')

# 2 HS1 and HS3 distributions
fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.2), gridspec_kw={'width_ratios': [1, 1.5], 'wspace': 0.3})
h1 = dist[(dist.level == 'hs1') & (dist.scope == 'clean')].copy(); h1['name'] = h1['class'].map(HS1)
axes[0].barh(h1.name, h1.n, color=BLUE); axes[0].invert_yaxis(); axes[0].set_title(f"HS1, cleaned photographs (n = {S['cleaned_photographs']:,})", loc='left')
for i, n in enumerate(h1.n): axes[0].text(n + 200, i, f'{n:,}', va='center', fontsize=9)
axes[0].set_xlim(0, h1.n.max() * 1.2)
h3 = dist[(dist.level == 'hs3') & (dist.scope == 'benchmark')].sort_values('n', ascending=True)
d3 = dist[(dist.level == 'hs3') & (dist.scope == 'dev')].set_index('class').n
axes[1].barh(h3['class'], h3.n, color=ORANGE, label='benchmark'); axes[1].barh(h3['class'], h3['class'].map(d3).fillna(0), color=AQUA, label='development partition (12,000)')
for i, n in enumerate(h3.n): axes[1].text(n + 80, i, f'{n:,}', va='center', fontsize=9)
axes[1].set_title(f"HS3 classes in the benchmark (n = {S['benchmark_rows']:,})", loc='left'); axes[1].legend(loc='lower right'); axes[1].set_xlim(0, h3.n.max() * 1.15)
finish(fig, 'Class distributions', 'Aerophones and chordophones dominate; the benchmark keeps HS3 classes with ≥30 objects that also carry a family and a name.', 'FULL_DATASET_CLASSES.png')

# 3 museums
fig, ax = plt.subplots(figsize=(12, 6))
m = mus.sort_values('clean', ascending=False).head(20).iloc[::-1]
ax.barh(m.museum.str.slice(0, 45), m.clean, color=BLUE, label='cleaned photographs'); ax.barh(m.museum.str.slice(0, 45), m.benchmark, color=ORANGE, label='benchmark objects')
fig.subplots_adjust(left=0.36); ax.legend(loc='lower right'); ax.set_xlabel('objects'); ax.grid(axis='y', visible=False); ax.tick_params(axis='y', labelsize=8.5)
share = mus.benchmark.nlargest(5).sum() / mus.benchmark.sum()
finish(fig, 'Museum distribution (20 largest of the cleaned set)', f"{int((mus.clean > 0).sum())} museums in the cleaned set, {int((mus.benchmark > 0).sum())} in the benchmark; the 5 largest hold {share:.0%} of benchmark objects (museum confounding risk).",
       'FULL_DATASET_MUSEUMS.png')
