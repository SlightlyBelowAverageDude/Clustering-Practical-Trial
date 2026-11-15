"""Download official pretrained DINOv2 S/B/L weights (timm conversions of facebookresearch/dinov2, LVD-142M) into the
ignored data/dinov2_model cache, pin the Hugging Face revision, record sha256, and time a CPU forward pass.
Output: reports/dinov2/models.json (no weights are written outside data/)."""
import hashlib, json, os, time, platform
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / 'data/dinov2_model'
os.environ['HF_HUB_DISABLE_XET'] = '1'
import torch, timm
from huggingface_hub import HfApi, hf_hub_download

MODELS = {'dinov2_s': 'vit_small_patch14_dinov2.lvd142m', 'dinov2_b': 'vit_base_patch14_dinov2.lvd142m', 'dinov2_l': 'vit_large_patch14_dinov2.lvd142m'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def load(name, info=None):
    """Load a pinned pretrained model from the local cache (used by the extraction script)."""
    info = info or json.loads((ROOT / 'reports/dinov2/models.json').read_text())['models'][name]
    m = timm.create_model(MODELS[name], pretrained=True, num_classes=0, dynamic_img_size=True,
                          pretrained_cfg_overlay={'file': info['weight_path']}).eval().requires_grad_(False)
    assert sha(info['weight_path']) == info['weight_sha256']
    return m


def main():
    torch.set_num_threads(4)
    out = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
           'platform': platform.platform(), 'torch': torch.__version__, 'timm': timm.__version__, 'models': {}}
    for name, mid in MODELS.items():
        repo = 'timm/' + mid
        rev = HfApi().model_info(repo).sha
        w = hf_hub_download(repo, 'model.safetensors', revision=rev, cache_dir=str(CACHE / 'hub'))
        c = hf_hub_download(repo, 'config.json', revision=rev, cache_dir=str(CACHE / 'hub'))
        info = {'repo': repo, 'revision': rev, 'weight_path': w, 'weight_sha256': sha(w), 'weight_bytes': os.path.getsize(w), 'config_sha256': sha(c)}
        m = load(name, info)
        cfg = timm.data.resolve_data_config({}, model=m)
        info.update({'params_million': round(sum(p.numel() for p in m.parameters()) / 1e6, 1), 'embed_dim': m.num_features, 'depth': len(m.blocks),
                     'default_data_config': {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}})
        for size in (224, 518):
            x = torch.randn(8, 3, size, size)
            with torch.no_grad():
                m.forward_intermediates(x[:1], indices=[-1], intermediates_only=True)
                t = time.perf_counter(); m.forward_intermediates(x, indices=[-1, -2, -4], return_prefix_tokens=True, intermediates_only=True)
                info[f'cpu_seconds_per_image_{size}px'] = round((time.perf_counter() - t) / 8, 3)
        out['models'][name] = info
        print(name, {k: v for k, v in info.items() if k not in ('default_data_config', 'weight_path')}, flush=True)
    (ROOT / 'reports/dinov2').mkdir(parents=True, exist_ok=True)
    (ROOT / 'reports/dinov2/models.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
