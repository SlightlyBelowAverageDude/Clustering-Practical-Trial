"""P1 figures + runtime/memory table + model manifest, built only from saved P1 artifacts.

Writes:
  artifacts/DINOV2_RUNTIME_MEMORY.csv     extraction (per model) and clustering (per model x method) cost
  artifacts/DINOV2_MODEL_MANIFEST.json    checkpoint, params, dims, layers, shapes, hashes, runtime, peak memory, failures
  artifacts/figures/P1_*.png              four figures
  artifacts/figures/figure_values.json    every number drawn, so p1_verify.py can check the figures against the CSVs
Usage (repo root): .venv/bin/python reports/roadmap/phases/P1/source/p1_figures.py
"""
import json, re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / 'reports/roadmap/phases/P1/artifacts'; FIG = ART / 'figures'; LOG = ART / 'logs'
EMB = ROOT / 'reports/dinov2/embeddings'
MODELS = ['dinov2_s', 'dinov2_b', 'dinov2_l']
LBL = {'dinov2_s': 'DINOv2-S', 'dinov2_b': 'DINOv2-B', 'dinov2_l': 'DINOv2-L'}
METH = {'kmeans': 'k-Means', 'agglomerative_average_cosine': 'Agglomerative (avg, cosine)', 'spectral_knn': 'Spectral (15-NN)'}
# reference categorical palette, light mode, slots 1-3 (validated all-pairs); neutral ink for context
COL = {'dinov2_s': '#2a78d6', 'dinov2_b': '#eb6834', 'dinov2_l': '#1baf7a'}
MCOL = {'kmeans': '#2a78d6', 'agglomerative_average_cosine': '#eb6834', 'spectral_knn': '#1baf7a'}
INK, INK2, GRID, SURF, CTX = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#8a8984'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'text.color': INK})
VALUES = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gnu_time(p):
    t = Path(p).read_text()
    rss = int(re.search(r'Maximum resident set size \(kbytes\): (\d+)', t).group(1))
    wall = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)', t).group(1).split(':')
    wall = sum(float(x) * 60 ** i for i, x in enumerate(reversed(wall)))
    code = int(re.search(r'Exit status: (\d+)', t).group(1))
    return rss / 1024 ** 2, wall, code


def stalls(log_text, total_seconds):
    """Wall-clock stalls in an extraction log: progress intervals slower than 10x the median s/img (e.g. system sleep)."""
    pts = [(int(a), int(b)) for a, b in re.findall(r'224px: (\d+)/12000 (\d+)s', log_text)]
    rates = [((b2 - b1) / (a2 - a1), b2 - b1, a2 - a1) for (a1, b1), (a2, b2) in zip(pts, pts[1:])]
    if not rates:
        return 0.0, total_seconds / 12000
    med = float(np.median([r[0] for r in rates]))
    stall = sum(sec - n * med for r, sec, n in rates if r > 10 * med)
    return float(stall), (total_seconds - stall) / 12000


def runtime_and_manifest():
    models = json.loads((ROOT / 'reports/dinov2/models.json').read_text())['models']
    runs = pd.read_csv(ART / 'p1_runs.csv')
    rows, man = [], {'manifest_id': 'MIMO-HS3-DEV12K-v1', 'protocol_id': 'FAIR-CMP-v1', 'models': {}}
    for m in MODELS:
        e = json.loads((EMB / f'dev_{m}_224.json').read_text()); info = models[m]
        tf = LOG / f'extract_{m}.time'
        if tf.exists():
            rss, wall, code = gnu_time(tf); prov = 'P1 new extraction'; logp = f'reports/roadmap/phases/P1/artifacts/logs/extract_{m}.log'
            log = (LOG / f'extract_{m}.log').read_text(); retries = log.count('== ' + m) - 1
            stall, active_rate = stalls(log, e['seconds'])
        else:
            rss, wall, code = np.nan, np.nan, 0; prov = 'reused validated Week 6 cache (not re-extracted in P1)'; logp = 'data/dev_embed.log (local, Week 6)'; retries = 0
            stall, active_rate = 0.0, e['seconds'] / e['n']
        rows.append({'stage': 'extraction', 'model': m, 'method': '', 'provenance': prov, 'images': e['n'], 'new_images': e['new_images'], 'cache_hits': e['cache_hits'],
                     'seconds': e['seconds'], 'stall_seconds': stall, 'active_seconds': e['seconds'] - stall, 'seconds_per_image_active': active_rate,
                     'wall_seconds_incl_load': wall, 'peak_rss_gib': rss,
                     'torch_threads': e['threads'], 'batch': 16, 'device': e['device'], 'exit_status': code, 'restarts': retries})
        man['models'][m] = {'provenance': prov, 'checkpoint': f"{info['repo']}@{info['revision']}", 'weight_sha256': info['weight_sha256'], 'weight_bytes': info['weight_bytes'],
                            'params_million': info['params_million'], 'blocks': info['depth'], 'embedding_dim': info['embed_dim'], 'input_resolution': e['descriptor']['size'],
                            'adaptation': e['descriptor']['adaptation'], 'transform': e['descriptor']['transform'], 'pooling': e['descriptor']['pooling'],
                            'layers': {k: {**v, 'block_index_0based': info['depth'] + int(k)} for k, v in e['arrays'].items()}, 'dtype': 'float32',
                            'ordered_inputs_sha256': e['ordered_inputs_sha256'], 'embedding_manifest': f'reports/dinov2/embeddings/dev_{m}_224.json',
                            'cache_fingerprint': e['fingerprint'], 'cache_dir': f"data/dinov2/cache/{e['fingerprint']}/", 'images_processed': e['n'], 'failures': 0 if e['n'] == 12000 else 12000 - e['n'],
                            'extraction_seconds': e['seconds'], 'stall_seconds': round(stall, 1), 'active_extraction_seconds': round(e['seconds'] - stall, 1), 'peak_rss_gib': None if np.isnan(rss) else round(rss, 3), 'threads': e['threads'], 'device': e['device'],
                            'torch': e['torch'], 'timm': e['timm'], 'extracted_utc': e['utc'], 'log': logp}
        for meth in METH:
            r = runs[(runs.model == m) & (runs.layer == -1) & (runs.method == meth)]
            rows.append({'stage': 'clustering (layer -1, centered_l2, k=10)', 'model': m, 'method': meth, 'provenance': 'P1 new execution', 'images': 12000,
                         'seconds': r.fit_seconds.sum(), 'seconds_per_image': np.nan, 'fit_seconds_mean': r.fit_seconds.mean(), 'fits': len(r), 'torch_threads': np.nan, 'blas_threads': 4, 'device': 'cpu'})
    ct = LOG / 'cluster_fit.time'
    if ct.exists():
        rss, wall, code = gnu_time(ct)
        rows.append({'stage': 'clustering, whole P1 grid (99 fits incl. silhouettes)', 'model': 'all', 'method': 'all', 'provenance': 'P1 new execution', 'images': 12000,
                     'wall_seconds_incl_load': wall, 'peak_rss_gib': rss, 'blas_threads': 4, 'exit_status': code, 'device': 'cpu'})
    rt = pd.DataFrame(rows); rt.to_csv(ART / 'DINOV2_RUNTIME_MEMORY.csv', index=False, float_format='%.4f')
    (ART / 'DINOV2_MODEL_MANIFEST.json').write_text(json.dumps(man, indent=2))
    return rt


def fig_scaling(main):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    VALUES['P1_MODEL_SCALING'] = []
    for ax, meth in zip(axes, METH):
        d = main[main.method == meth]
        for i, m in enumerate(MODELS):
            r = d[d.model == m].iloc[0]; sd = 0 if np.isnan(r.HS3_AMI_sd) else r.HS3_AMI_sd
            ax.bar(i, r.HS3_AMI_mean, width=0.6, color=COL[m], yerr=sd if sd else None, error_kw={'ecolor': INK2, 'elinewidth': 1, 'capsize': 3})
            ax.text(i, r.HS3_AMI_mean + sd + 0.008, f'{r.HS3_AMI_mean:.3f}', ha='center', va='bottom', fontsize=9, color=INK)
            VALUES['P1_MODEL_SCALING'].append({'method': meth, 'model': m, 'HS3_AMI_mean': r.HS3_AMI_mean, 'HS3_AMI_sd': sd})
        for c, ls in (('clip', ':'), ('siglip', '--')):
            r = d[d.model == c].iloc[0]
            ax.axhline(r.HS3_AMI_mean, color=CTX, ls=ls, lw=1.2)
            ax.text(2.45, r.HS3_AMI_mean, {'clip': 'CLIP', 'siglip': 'SigLIP'}[c], va='center', ha='left', fontsize=8, color=INK2)
            VALUES['P1_MODEL_SCALING'].append({'method': meth, 'model': c, 'HS3_AMI_mean': r.HS3_AMI_mean, 'HS3_AMI_sd': 0})
        ax.set_xticks(range(3), [LBL[m] for m in MODELS]); ax.set_title(METH[meth], fontsize=10, color=INK); ax.set_xlim(-0.6, 2.9)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('HS3 AMI (mean ± SD over seeds)'); axes[0].set_ylim(0.35, 0.65)
    fig.suptitle('DINOv2 model size, last layer: 12,000 dev objects, centered + L2, k = 10   (grey: CLIP/SigLIP, cached Week 6 results)', fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(FIG / 'P1_MODEL_SCALING.png', dpi=160); plt.close(fig)


def fig_layers(g):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    VALUES['P1_LAYER_DEPTH'] = []
    for ax, meth in zip(axes, METH):
        for m in MODELS:
            d = g[(g.method == meth) & (g.model == m)].set_index('layer').loc[[-4, -2, -1]]
            ax.plot([0, 1, 2], d.HS3_AMI_mean, color=COL[m], lw=2, marker='o', ms=6, mec=SURF, mew=1.5, label=LBL[m])
            for lay, v in d.HS3_AMI_mean.items():
                VALUES['P1_LAYER_DEPTH'].append({'method': meth, 'model': m, 'layer': int(lay), 'HS3_AMI_mean': v})
        ax.set_xticks([0, 1, 2], ['layer −4', 'layer −2', 'layer −1 (last)']); ax.set_title(METH[meth], fontsize=10, color=INK); ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('HS3 AMI (mean over seeds)'); axes[0].legend(loc='lower right', fontsize=9)
    fig.suptitle('Representation depth: CLS token of block −4 / −2 / −1, 12,000 dev objects, centered + L2, k = 10', fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(FIG / 'P1_LAYER_DEPTH.png', dpi=160); plt.close(fig)


def fig_ami_ari(main):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    mk = {'kmeans': 'o', 'agglomerative_average_cosine': 's', 'spectral_knn': '^'}
    VALUES['P1_AMI_ARI_COMPARISON'] = []
    for meth in METH:
        for m in MODELS + ['clip', 'siglip']:
            r = main[(main.method == meth) & (main.model == m)].iloc[0]
            c = COL.get(m, CTX)
            ax.scatter(r.HS3_AMI_mean, r.HS3_ARI_mean, s=60 + 1.5 * r.largest_cluster_pct_max, marker=mk[meth], color=c, edgecolor=SURF, linewidth=1.5, zorder=3)
            VALUES['P1_AMI_ARI_COMPARISON'].append({'method': meth, 'model': m, 'HS3_AMI_mean': r.HS3_AMI_mean, 'HS3_ARI_mean': r.HS3_ARI_mean, 'largest_cluster_pct_max': r.largest_cluster_pct_max})
            if m in MODELS and meth != 'kmeans' or (m == 'dinov2_l' and meth == 'kmeans'):
                ax.annotate(f'{LBL[m].split("-")[1]} {r.largest_cluster_pct_max:.0f}%', (r.HS3_AMI_mean, r.HS3_ARI_mean), xytext=(6, -3), textcoords='offset points', fontsize=7.5, color=INK2)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='', color=COL[m], label=LBL[m]) for m in MODELS] + [Line2D([], [], marker='o', ls='', color=CTX, label='CLIP / SigLIP (cached)')]
    h += [Line2D([], [], marker=mk[k], ls='', color=INK2, mfc='none', label=v) for k, v in METH.items()]
    ax.legend(handles=h, fontsize=8, loc='upper left', ncol=2)
    ax.set_xlabel('HS3 AMI (mean over seeds)'); ax.set_ylabel('HS3 ARI (mean over seeds)')
    ax.set_title('AMI vs ARI, last layer, k = 10 (marker size and label: largest cluster %, worst seed)', fontsize=9.5, color=INK, loc='left')
    fig.tight_layout(); fig.savefig(FIG / 'P1_AMI_ARI_COMPARISON.png', dpi=160); plt.close(fig)


def fig_runtime(main, rt):
    ex = rt[rt.stage == 'extraction'].set_index('model')
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    VALUES['P1_RUNTIME_TRADEOFF'] = []
    for meth in METH:
        xs, ys = [], []
        for m in MODELS:
            r = main[(main.method == meth) & (main.model == m)].iloc[0]
            xs.append(ex.loc[m, 'active_seconds'] / 60); ys.append(r.HS3_AMI_mean)
            VALUES['P1_RUNTIME_TRADEOFF'].append({'method': meth, 'model': m, 'extraction_minutes': ex.loc[m, 'active_seconds'] / 60, 'HS3_AMI_mean': r.HS3_AMI_mean})
        ax.plot(xs, ys, color=MCOL[meth], lw=2, marker='o', ms=7, mec=SURF, mew=1.5, label=METH[meth])
        for x, y, m in zip(xs, ys, MODELS):
            ax.annotate(LBL[m].split('-')[1], (x, y), xytext=(5, 4), textcoords='offset points', fontsize=8, color=INK2)
    ax.set_xscale('log'); ax.set_xlabel('Active embedding extraction time, 12,000 images, CPU (minutes, log scale; stalls removed)'); ax.set_ylabel('HS3 AMI (mean over seeds)')
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator([10, 15, 20, 30, 50, 75, 100])); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); ax.set_xlim(9.5, 125)
    ax.legend(fontsize=8.5, loc='lower right', bbox_to_anchor=(0.97, 0.2))
    ax.set_title('Cost vs clustering quality (last layer, centered + L2, k = 10)', fontsize=10, color=INK, loc='left')
    fig.tight_layout(); fig.savefig(FIG / 'P1_RUNTIME_TRADEOFF.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    FIG.mkdir(parents=True, exist_ok=True)
    rt = runtime_and_manifest()
    g = pd.read_csv(ART / 'DINOV2_LAYER_ABLATION.csv'); main = pd.read_csv(ART / 'DINOV2_SBL_12K.csv')
    fig_scaling(main); fig_layers(g); fig_ami_ari(main); fig_runtime(main, rt)
    VALUES['_png_sha256'] = {p.name: sha(p) for p in sorted(FIG.glob('*.png'))}
    (FIG / 'figure_values.json').write_text(json.dumps(VALUES, indent=1))
    print('figures:', sorted(VALUES['_png_sha256']))
