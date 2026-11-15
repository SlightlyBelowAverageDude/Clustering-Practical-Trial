"""CLIP ViT-B/32 and SigLIP ViT-B/16 embeddings for the development partition, with the exact pilot pipeline.

Model loading copies spikes/pilot_benchmark.py (CLIP) and results/phase2/source/siglip_embed.py (SigLIP): same pinned
weights/hash, same transform kwargs, EXIF/RGB/white-square padding, L2-normalised float32, CPU.
Parity gate: the first 32 pilot images are re-embedded and must match the saved pilot caches (max |diff| < 1e-4)
before any development image is processed.
Usage: python reports/dinov2/source/openclip_embed.py --model clip|siglip --set dev
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch, open_clip
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dinov2_embed import inputs, canonical  # noqa: E402
from dinov2_weights import sha  # noqa: E402


def build(name):
    if name == 'clip':
        info = json.loads((ROOT / 'data/mimo_pilot/model/model.json').read_text())
        cfg = info['config'].get('preprocess_cfg', {}); arch = 'ViT-B-32'
    else:
        info = json.loads((ROOT / 'data/phase2/siglip_model/model.json').read_text())
        open_clip.add_model_config(ROOT / 'data/phase2/siglip_model/SiglipPhase2.json'); cfg = info['config']['preprocess_cfg']; arch = 'SiglipPhase2'
    assert sha(info['weight_path']) == info['weight_sha256']
    kw = {'image_' + k: cfg[k] for k in ('mean', 'std', 'interpolation', 'resize_mode') if k in cfg}
    model, _, pre = open_clip.create_model_and_transforms(arch, pretrained=info['weight_path'], device='cpu', **kw)
    return model.eval().requires_grad_(False), pre, info


def embed(model, pre, rows, batch=16, log=''):
    out, t0 = [], time.time()
    for s in range(0, len(rows), batch):
        ts = []
        for r in rows[s:s + batch]:
            assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() == r['sha256']
            with Image.open(r['path']) as im:
                im = ImageOps.exif_transpose(im).convert('RGB'); side = max(im.size)
                ts.append(pre(ImageOps.pad(im, (side, side), color=(255, 255, 255))))
        with torch.inference_mode():
            out.append(torch.nn.functional.normalize(model.encode_image(torch.stack(ts)).float(), dim=-1).numpy())
        if log and (s // batch) % 50 == 0:
            print(f'{log}: {s + len(ts)}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--set', default='dev'); ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args(); torch.manual_seed(42); torch.set_num_threads(a.threads)
    model, pre, info = build(a.model)
    pilot = inputs('pilot')[:32]
    ref = np.load(json.loads((ROOT / ('results/pilot/embedding_manifest.json' if a.model == 'clip' else 'results/phase2/siglip/embedding_manifest.json')).read_text())['array_path'].replace(
        '/home/hayvan/Programming/LMU/CMVAL/ClusteringmeetsVisionandLanguage/Clustering-Practical-Trial-main', str(ROOT)))[:32]
    diff = float(np.abs(embed(model, pre, pilot) - ref).max())
    print(a.model, 'pilot parity max |diff|', diff, flush=True)
    assert diff < 1e-4, 'pipeline does not reproduce the pilot cache'
    rows = inputs(a.set)
    t0 = time.time(); X = embed(model, pre, rows, log=f'{a.set} {a.model}')
    assert np.isfinite(X).all() and np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-4)
    edir = ROOT / 'data/dinov2/emb' / a.set; edir.mkdir(parents=True, exist_ok=True)
    f = edir / f'{a.model}.npy'; np.save(f, X)
    man = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'set': a.set, 'n': len(rows), 'model': {k: info[k] for k in ('repo', 'revision', 'weight_sha256')},
           'pilot_parity_max_abs_diff': diff, 'array': {'path': str(f.relative_to(ROOT)), 'sha256': sha(f), 'shape': list(X.shape)},
           'ordered_inputs_sha256': hashlib.sha256(canonical([(r['image_id'], r['sha256']) for r in rows]).encode()).hexdigest(), 'seconds': round(time.time() - t0, 1)}
    md = ROOT / 'reports/dinov2/embeddings'; md.mkdir(parents=True, exist_ok=True)
    (md / f'{a.set}_{a.model}.json').write_text(json.dumps(man, indent=2)); print('done', a.model, X.shape, flush=True)


if __name__ == '__main__':
    main()
