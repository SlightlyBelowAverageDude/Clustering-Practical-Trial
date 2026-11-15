"""Frozen OpenCLIP benchmark with content-verified, ordered embedding caches."""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'data/.matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from pilot_acquire import DATA, ROOT, atomic_csv, atomic_json, digest, now

MODEL_REPO = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'
PREPROCESS = 'EXIF transpose; RGB; white square pad preserving aspect ratio; official OpenCLIP eval resize/crop 224; official mean/std; L2 output; float32 CPU'


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def save_array(path, array):
    tmp = path.with_suffix('.part')
    with tmp.open('wb') as f:
        np.save(f, array, allow_pickle=False)
    tmp.replace(path)


def load_model_info():
    """Resolve once, then pin an immutable Hub revision and exact weight hash."""
    from huggingface_hub import HfApi, hf_hub_download
    model_dir = DATA / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)
    info_path = model_dir / 'model.json'
    if info_path.exists():
        info = json.loads(info_path.read_text())
        weight = Path(info['weight_path'])
        if weight.exists() and sha_file(weight) == info['weight_sha256']:
            # Runtime changes must invalidate inference caches even with identical weights.
            for field, package in [('open_clip_version', 'open_clip_torch'), ('pillow_version', 'pillow'), ('torchvision_version', 'torchvision')]:
                info[field] = importlib.metadata.version(package)
            info['torch_version'] = torch.__version__
            info['preprocessing'] = PREPROCESS
            return info
        revision = info['revision']
    else:
        revision = HfApi().model_info(MODEL_REPO).sha
    kwargs = {'repo_id': MODEL_REPO, 'revision': revision, 'cache_dir': str(model_dir / 'hub')}
    weight = hf_hub_download(filename='open_clip_model.safetensors', **kwargs)
    cfg_path = hf_hub_download(filename='open_clip_config.json', **kwargs)
    info = {'repo': MODEL_REPO, 'revision': revision, 'weight_path': weight, 'weight_sha256': sha_file(weight), 'config': json.loads(Path(cfg_path).read_text()), 'preprocessing': PREPROCESS, 'open_clip_version': importlib.metadata.version('open_clip_torch'), 'torch_version': torch.__version__, 'pillow_version': importlib.metadata.version('pillow'), 'torchvision_version': importlib.metadata.version('torchvision'), 'device': 'cpu', 'precision': 'float32'}
    atomic_json(info_path, info)
    return info


def valid_vector(path, meta_path, descriptor):
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta['descriptor'] != descriptor or meta['array_sha256'] != sha_file(path):
            return None
        x = np.load(path, allow_pickle=False)
        if x.shape != (512,) or not np.isfinite(x).all() or not np.isclose(np.linalg.norm(x), 1, atol=1e-4):
            return None
        return x
    except (OSError, ValueError, KeyError):
        return None


def embed(df, out, batch_size=16):
    model_info = load_model_info()
    fingerprint = digest(canonical(model_info).encode())
    paths = [DATA / f for f in df['file']]
    inputs = []
    for row, path in zip(df.to_dict('records'), paths):
        actual_hash = sha_file(path)
        if actual_hash != row['sha256']:
            raise RuntimeError('Image content changed since acquisition: ' + row['image_id'])
        inputs.append({'id': row['id'], 'image_id': row['image_id'], 'sha256': actual_hash})
    if len({r['image_id'] for r in inputs}) != len(inputs) or len(set(df.pixel_sha256)) != len(df):
        raise RuntimeError('Duplicate image identifiers or pixels in benchmark manifest')
    descriptor = {'inputs': inputs, 'model_fingerprint': fingerprint}
    input_key = digest(canonical(descriptor).encode())
    cache = DATA / 'embeddings'
    cache.mkdir(parents=True, exist_ok=True)
    vector_dir = cache / fingerprint
    vector_dir.mkdir(parents=True, exist_ok=True)
    xs, pending = [None] * len(df), []
    for i, row in enumerate(inputs):
        desc = {'image_id': row['image_id'], 'sha256': row['sha256'], 'model_fingerprint': fingerprint}
        key = digest(canonical(desc).encode())
        path, meta = vector_dir / (key + '.npy'), vector_dir / (key + '.json')
        x = valid_vector(path, meta, desc)
        if x is None:
            pending.append((i, path, meta, desc))
        else:
            xs[i] = x
    started = time.perf_counter()
    print('Embedding cache:', len(df) - len(pending), 'verified hits;', len(pending), 'new images', flush=True)
    if pending:
        import open_clip
        cfg = model_info['config'].get('preprocess_cfg', {})
        kwargs = {'image_' + k: cfg[k] for k in ('mean', 'std', 'interpolation', 'resize_mode') if k in cfg}
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=model_info['weight_path'], device='cpu', **kwargs)
        model.eval().requires_grad_(False)
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            tensors = []
            for i, _, _, _ in batch:
                with Image.open(paths[i]) as im:
                    im = ImageOps.exif_transpose(im).convert('RGB')
                    side = max(im.size)
                    im = ImageOps.pad(im, (side, side), color=(255, 255, 255))
                    tensors.append(preprocess(im))
            with torch.inference_mode():
                features = model.encode_image(torch.stack(tensors)).float()
                features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
            for (i, path, meta, desc), x in zip(batch, features):
                if not np.isfinite(x).all() or not np.isclose(np.linalg.norm(x), 1, atol=1e-4):
                    raise RuntimeError('Invalid model embedding')
                save_array(path, x)
                atomic_json(meta, {'descriptor': desc, 'array_sha256': sha_file(path)})
                xs[i] = x
            print('Embedded:', min(offset + batch_size, len(pending)), '/', len(pending), f'in {time.perf_counter() - started:.1f}s', flush=True)
        del model
    X = np.stack(xs).astype('float32')
    array_path = cache / (input_key + '.npy')
    save_array(array_path, X)
    evidence = {'input_key': input_key, 'descriptor': descriptor, 'array_sha256': sha_file(array_path), 'shape': list(X.shape), 'model': model_info, 'new_images': len(pending), 'cache_hits': len(df) - len(pending), 'embedding_seconds': time.perf_counter() - started, 'array_path': str(array_path)}
    atomic_json(cache / (input_key + '.json'), evidence)
    atomic_json(out / 'embedding_manifest.json', evidence)
    atomic_csv(out / 'image_order.csv', [{'row': i, **v} for i, v in enumerate(inputs)])
    return X, evidence


def run_clustering(X, df, out, evidence, ks, seeds):
    results, assignments = [], []
    gallery_choice = None
    for representation in ['l2', 'pca50_l2']:
        started = time.perf_counter()
        if representation == 'l2':
            Z, dimensions, variance = normalize(X), X.shape[1], 1.0
        else:
            dimensions = min(50, len(X) - 1, X.shape[1])
            pca = PCA(n_components=dimensions, svd_solver='full', whiten=False)
            Z = normalize(pca.fit_transform(X))
            variance = float(pca.explained_variance_ratio_.sum())
            np.savez(out / 'pca.npz', components=pca.components_, mean=pca.mean_, variance=pca.explained_variance_ratio_)
        prep_seconds = time.perf_counter() - started
        for k in ks:
            if k >= len(df):
                continue
            for method in ['kmeans', 'average_cosine']:
                for seed in seeds if method == 'kmeans' else [seeds[0]]:
                    started = time.perf_counter()
                    if method == 'kmeans':
                        pred = KMeans(n_clusters=k, n_init=10, max_iter=300, tol=1e-4, random_state=seed, algorithm='lloyd').fit_predict(Z)
                    else:
                        pred = AgglomerativeClustering(n_clusters=k, linkage='average', metric='cosine').fit_predict(Z)
                    elapsed = time.perf_counter() - started
                    run_id = f'{representation}_{method}_k{k}_s{seed}'
                    np.save(out / (run_id + '.npy'), pred, allow_pickle=False)
                    for i, label in enumerate(pred):
                        assignments.append({'run_id': run_id, 'row': i, 'image_id': df.iloc[i].image_id, 'cluster': int(label)})
                    for label in ['hs1', 'hs2', 'hs3']:
                        mask = df[label].ne('').to_numpy()
                        y, p = df.loc[mask, label].to_numpy(), pred[mask]
                        classes = len(set(y))
                        metrics = {'AMI': np.nan, 'ARI': np.nan, 'NMI': np.nan}
                        if len(y) >= 2 and classes >= 2:
                            metrics = {'AMI': adjusted_mutual_info_score(y, p), 'ARI': adjusted_rand_score(y, p), 'NMI': normalized_mutual_info_score(y, p)}
                        results.append({'run_id': run_id, 'representation': representation, 'method': method, 'requested_k': k, 'actual_k': len(set(pred)), 'seed': seed, 'deterministic': method == 'average_cosine', 'n_images': len(df), 'label': label, 'n_evaluated': len(y), 'label_classes': classes, 'dimensions': dimensions, 'pca_variance_retained': variance, 'preprocess_seconds': prep_seconds, 'cluster_seconds': elapsed, 'input_key': evidence['input_key'], **metrics})
                    print(run_id, f'{elapsed:.3f}s', flush=True)
                    # Gallery chosen in advance, independently of taxonomy scores.
                    if representation == 'l2' and method == 'kmeans' and seed == seeds[0] and k == min(ks):
                        gallery_choice = (run_id, pred, Z)
                    atomic_csv(out / 'scores.csv', results)
    atomic_csv(out / 'assignments.csv', assignments)
    return pd.DataFrame(results), gallery_choice


def plot_scores(scores, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, label in zip(axes, ['hs1', 'hs2', 'hs3']):
        for (rep, method), sub in scores[scores.label == label].groupby(['representation', 'method']):
            means = sub.groupby('requested_k').AMI.agg(['mean', 'std']).fillna(0)
            ax.errorbar(means.index, means['mean'], yerr=means['std'], marker='o', capsize=3, label=f'{rep} / {method}')
        ax.set(title=label.upper(), xlabel='Number of clusters')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Adjusted mutual information (AMI)')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Frozen OpenCLIP ViT-B/32 · identical images · bars: seed standard deviation')
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=180)
    plt.close(fig)


def gallery(df, choice, out):
    if choice is None:
        return
    run_id, pred, Z = choice
    thumbnails = DATA / 'gallery' / out.name
    thumbnails.mkdir(parents=True, exist_ok=True)
    sections = []
    for cluster in sorted(set(pred)):
        ids = np.flatnonzero(pred == cluster)
        center = Z[ids].mean(axis=0)
        ids = ids[np.argsort(np.linalg.norm(Z[ids] - center, axis=1), kind='stable')[:10]]
        cards = []
        for i in ids:
            row = df.iloc[i]
            with Image.open(DATA / row.file) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((180, 160))
                thumbnail = thumbnails / (row.image_id + '.jpg')
                im.save(thumbnail, format='JPEG', quality=85)
            # Keep all photo bytes inside ignored data/, including gallery thumbnails.
            src = os.path.relpath(thumbnail, out)
            e = html.escape
            cards.append(f'<figure><img src="{e(src)}"><figcaption>{e(row.title)}<br>HS: {e(row.notations)}<br>{e(row.museum)}<br>{e(row.id)}<br>Rights: {e(row.rights)}<br><a href="{e(row.image_url)}">Source image</a></figcaption></figure>')
        sections.append(f'<section><h2>Cluster {cluster} · {int((pred == cluster).sum())} images</h2><div>{"".join(cards)}</div></section>')
    (out / 'cluster_gallery.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>MIMO pilot cluster gallery</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px;background:#f6f5f1}section>div{display:flex;flex-wrap:wrap}figure{width:190px;margin:8px;padding:10px;background:white}img{width:180px;height:160px;object-fit:contain}figcaption{font-size:11px;overflow-wrap:anywhere}</style><h1>MIMO cluster gallery</h1><p>' + html.escape(run_id) + '. Ten images nearest each cluster mean; chosen without label scores. Labels shown only for interpretation.</p>' + ''.join(sections) + '</html>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='pilot')
    ap.add_argument('--ks', nargs='+', type=int, default=[5, 10, 20, 40])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    out = ROOT / 'results' / a.tag
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / f'sample_{a.tag}.csv', dtype=str, keep_default_na=False)
    if len(df) < 3:
        raise RuntimeError('At least three acquired images required')
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    source_names = ['pilot_acquire.py', 'pilot_benchmark.py', 'pilot_validate.py', 'pilot_report.py', 'mimo_harvest.py']
    sources = {name: sha_file(ROOT / 'spikes' / name) for name in source_names if (ROOT / 'spikes' / name).exists()}
    sampling_seed = json.loads((out / 'dataset_report.json').read_text())['seed']
    atomic_json(out / 'environment.json', {'recorded_utc': now(), 'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(), 'packages': packages, 'torch_threads': a.threads, 'device': 'cpu', 'deterministic_algorithms': True, 'embedding_seed': 42, 'sampling_seed': sampling_seed, 'clustering_seeds': a.seeds, 'command': sys.argv, 'parameters': vars(a), 'source_sha256': sources})
    with threadpool_limits(limits=a.threads):
        X, evidence = embed(df, out, a.batch_size)
        scores, choice = run_clustering(X, df, out, evidence, a.ks, a.seeds)
    plot_scores(scores, out)
    gallery(df, choice, out)
    atomic_json(out / 'completion.json', {'finished_utc': now(), 'images': len(df), 'score_rows': len(scores), 'clustering_runs': scores.run_id.nunique(), 'input_key': evidence['input_key'], 'success': True})
    print(scores.groupby(['representation', 'method', 'label']).AMI.max().to_string(), flush=True)


if __name__ == '__main__':
    main()
