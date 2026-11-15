"""Frozen pretrained DINOv2 embeddings (CLS token of layers -1, -2, -4) for a fixed, hash-verified image list.

Input adaptation is identical to the CLIP/SigLIP pilot runs (EXIF transpose, RGB, white square padding), followed by
the checkpoint's own timm evaluation transform at the requested size (518 px = timm default for these checkpoints and the
size used by the teammate's spikes; 224 px = cheaper variant). forward_intermediates(norm=True) applies the final
LayerNorm to every requested layer; layer -1's CLS token equals the model's pooled output (verified at run time).

Sets: pilot  -> data/mimo_pilot/sample_pilot.csv rows in results/phase2/inputs/image_order.csv order
      dev    -> reports/full_dataset/benchmark/dev_ids.csv rows (development partition only; held-out never embedded here)
Caches: data/dinov2/cache/<fingerprint>/<start>.npz (resumable). Arrays: data/dinov2/emb/<set>/<model>_<size>_L<layer>.npy
Manifest (tracked): reports/dinov2/embeddings/<set>_<model>_<size>.json
Usage: python reports/dinov2/source/dinov2_embed.py --set pilot --model dinov2_b --size 518 [--threads 6]
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, timm
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dinov2_weights import load, sha  # noqa: E402

LAYERS = [-1, -2, -4]


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'))


def inputs(name):
    if name == 'pilot':
        df = pd.read_csv(ROOT / 'data/mimo_pilot/sample_pilot.csv', dtype=str, keep_default_na=False)
        order = pd.read_csv(ROOT / 'results/phase2/inputs/image_order.csv', dtype=str, keep_default_na=False)
        assert df.image_id.tolist() == order.image_id.tolist()
        return [{'id': r.id, 'image_id': r.image_id, 'sha256': r.sha256, 'path': str(ROOT / 'data/mimo_pilot' / r.file)} for r in df.itertuples()]
    if name == 'dev':
        df = pd.read_csv(ROOT / 'reports/full_dataset/benchmark/dev_ids.csv', dtype=str, keep_default_na=False)
        assert (df.partition == 'dev').all()
        return [{'id': r.id, 'image_id': r.image_id, 'sha256': r.sha256, 'path': str(ROOT / 'data/mimo_pilot' / r.file)} for r in df.itertuples()]
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--set', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--size', type=int, default=518); ap.add_argument('--threads', type=int, default=6); ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args()
    torch.manual_seed(42); torch.set_num_threads(a.threads)
    info = json.loads((ROOT / 'reports/dinov2/models.json').read_text())['models'][a.model]
    model = load(a.model, info)
    cfg = timm.data.resolve_data_config({}, model=model); cfg['input_size'] = (3, a.size, a.size); cfg['crop_pct'] = 1.0
    tf = timm.data.create_transform(**cfg)
    rows = inputs(a.set)
    desc = {'model': a.model, 'repo': info['repo'], 'revision': info['revision'], 'weight_sha256': info['weight_sha256'], 'size': a.size,
            'layers': LAYERS, 'adaptation': 'exif_transpose,rgb,white_square_pad', 'transform': repr(tf), 'pooling': 'CLS token, final LayerNorm applied (norm=True)'}
    fp = hashlib.sha256(canonical(desc).encode()).hexdigest()
    cache = ROOT / 'data/dinov2/cache' / fp; cache.mkdir(parents=True, exist_ok=True)
    t0, out, new, hits = time.time(), {l: [] for l in LAYERS}, 0, 0
    for start in range(0, len(rows), a.batch):
        chunk = rows[start:start + a.batch]
        key = hashlib.sha256(canonical([(r['image_id'], r['sha256']) for r in chunk]).encode()).hexdigest()
        p = cache / f'{start:06d}.npz'
        if p.exists():
            z = np.load(p)
            if str(z['key']) == key:
                for i, l in enumerate(LAYERS): out[l].append(z['x'][i])
                hits += len(chunk); continue
        batch = []
        for r in chunk:
            raw = Path(r['path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == r['sha256'], 'image bytes changed: ' + r['id']
            with Image.open(r['path']) as im:
                im = ImageOps.exif_transpose(im).convert('RGB'); side = max(im.size)
                batch.append(tf(ImageOps.pad(im, (side, side), color=(255, 255, 255))))
        x = torch.stack(batch)
        with torch.inference_mode():
            inter = model.forward_intermediates(x, indices=LAYERS, norm=True, return_prefix_tokens=True, intermediates_only=True, output_fmt='NLC')
            # timm returns intermediates in ascending block order, not in the requested order -> map explicitly
            depth = len(model.blocks); blocks = sorted(depth + l for l in LAYERS)
            by_block = {b: t[1][:, 0].float().numpy() for b, t in zip(blocks, inter)}
            cls = np.stack([by_block[depth + l] for l in LAYERS])
            if start == 0:  # layer -1 CLS must equal the model's pooled output
                pooled = model(x).float().numpy()
                assert np.allclose(pooled, cls[0], atol=1e-4), 'layer -1 CLS != pooled output'
        assert np.isfinite(cls).all()
        tmp = p.with_suffix('.part.npz'); np.savez(tmp, x=cls, key=key); tmp.replace(p)
        for i, l in enumerate(LAYERS): out[l].append(cls[i])
        new += len(chunk)
        if (start // a.batch) % 20 == 0:
            print(f'{a.set} {a.model} {a.size}px: {start + len(chunk)}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
    edir = ROOT / 'data/dinov2/emb' / a.set; edir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for l in LAYERS:
        X = np.concatenate(out[l]).astype(np.float32)
        assert X.shape == (len(rows), info['embed_dim']) and np.isfinite(X).all()
        f = edir / f'{a.model}_{a.size}_L{abs(l)}.npy'; np.save(f, X)
        arrays[str(l)] = {'path': str(f.relative_to(ROOT)), 'sha256': sha(f), 'shape': list(X.shape), 'norm_min': float(np.linalg.norm(X, axis=1).min()), 'norm_max': float(np.linalg.norm(X, axis=1).max())}
    man = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'set': a.set, 'n': len(rows), 'descriptor': desc, 'fingerprint': fp,
           'ordered_inputs_sha256': hashlib.sha256(canonical([(r['image_id'], r['sha256']) for r in rows]).encode()).hexdigest(),
           'arrays': arrays, 'new_images': new, 'cache_hits': hits, 'seconds': round(time.time() - t0, 1), 'threads': a.threads,
           'device': 'cpu', 'torch': torch.__version__, 'timm': timm.__version__}
    md = ROOT / 'reports/dinov2/embeddings'; md.mkdir(parents=True, exist_ok=True)
    (md / f'{a.set}_{a.model}_{a.size}.json').write_text(json.dumps(man, indent=2))
    print('done', a.set, a.model, a.size, {l: v['shape'] for l, v in arrays.items()}, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
