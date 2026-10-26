"""Shared pieces for the viability spikes: image embedding and cluster scoring.

Every spike does the same thing: embed a sample of images with 1 or more frozen
encoders, cluster the embeddings with a few standard methods, and score the
clusters against a ground-truth label with AMI, NMI and ARI.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# name -> (library, model id, pretrained tag)
ENCODERS = {
    "clip_b32": ("open_clip", "ViT-B-32", "laion2b_s34b_b79k"),
    "clip_l14": ("open_clip", "ViT-L-14", "laion2b_s32b_b82k"),
    "dinov2_b": ("timm", "vit_base_patch14_dinov2.lvd142m", None),
    "dinov2_s": ("timm", "vit_small_patch14_dinov2.lvd142m", None),
}


class Encoder:
    def __init__(self, name: str, pad_square: bool = False):
        """pad_square: pad each image to a white square before the model's own transform, so the resize keeps the
        whole object. Without it the default transforms resize the short side and centre-crop, which cuts the ends
        off elongated objects (MIMO thumbnails have a median aspect ratio of 1.5)."""
        lib, model_id, tag = ENCODERS[name][:3]
        kw = ENCODERS[name][3] if len(ENCODERS[name]) > 3 else {}  # timm create_model keyword arguments, e.g. img_size
        self.name, self.lib, self.pad_square = name, lib, pad_square
        if lib == "open_clip":
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id, pretrained=tag)
            self.tokenizer = open_clip.get_tokenizer(model_id)
        elif lib == "timm":
            import timm
            self.model = timm.create_model(model_id, pretrained=True, num_classes=0, **kw)
            cfg = timm.data.resolve_data_config({}, model=self.model)
            if "img_size" in kw:
                cfg["input_size"] = (3, kw["img_size"], kw["img_size"])
            self.preprocess = timm.data.create_transform(**cfg)
        self.model = self.model.to(DEVICE).eval()

    @torch.no_grad()
    def embed_images(self, images, batch_size: int = 64) -> np.ndarray:
        """images: list of file paths or PIL images. Returns L2-normalised float32 array."""
        out = []
        for i in tqdm(range(0, len(images), batch_size), desc=f"embed {self.name}", leave=False):
            batch = []
            for im in images[i:i + batch_size]:
                if not isinstance(im, Image.Image):
                    im = Image.open(im)
                im = im.convert("RGB")
                if self.pad_square:
                    from PIL import ImageOps
                    im = ImageOps.pad(im, (max(im.size), max(im.size)), color=(255, 255, 255))
                batch.append(self.preprocess(im))
            x = torch.stack(batch).to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                f = self.model.encode_image(x) if self.lib == "open_clip" else self.model(x)
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def embed_texts(self, texts) -> np.ndarray:
        assert self.lib == "open_clip", "text embedding needs a CLIP-type encoder"
        tok = self.tokenizer(list(texts)).to(DEVICE)
        f = self.model.encode_text(tok)
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()


def cached_embeddings(cache_dir: str | Path, encoder_name: str, images, key_col=None, pad_square: bool = False) -> np.ndarray:
    """Embed once, save to <cache_dir>/<encoder>.npy (<encoder>_pad.npy with pad_square), reuse on the next run.

    A cache is reused only for the same image list in the same order: the list's hash is stored next to the array.
    A cache written before this check existed has no hash file; it is adopted when the length matches, with a note."""
    import hashlib
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{encoder_name}_pad" if pad_square else encoder_name
    f, kf = cache_dir / f"{stem}.npy", cache_dir / f"{stem}.key"
    paths = [str(i) for i in images] if all(not isinstance(i, Image.Image) for i in images) else None
    key = hashlib.md5("\n".join(paths).encode("utf-8")).hexdigest() if paths is not None else None
    if f.exists():
        X = np.load(f)
        if len(X) == len(images):
            if key is None or not kf.exists():
                if key is not None:
                    print(f"note: adopting {f} (no key file; same length, order not verifiable)", flush=True)
                    kf.write_text(key)
                return X
            if kf.read_text().strip() == key:
                return X
            print(f"note: {f} was built from a different image list; recomputing", flush=True)
    X = Encoder(encoder_name, pad_square=pad_square).embed_images(images)
    np.save(f, X)
    if key is not None:
        kf.write_text(key)
    return X


def hs_level(notation, d: int) -> np.ndarray:
    """Hornbostel-Sachs code cut to its first d digits, '' where the code has fewer than d digits.

    The suffix after a hyphen (321.322-71: the -71 means "with keyboard" and similar modifiers) is not part of the
    hierarchy and is dropped before slicing, so it can never form a class of its own. Compound codes joined with "+"
    (bagpipes: 422.112+422.22-62) are cut to their first code."""
    s = pd.Series(notation).fillna("").astype(str).str.split(r"[-+]", regex=True).str[0].str.replace(".", "", regex=False)
    return np.where(s.str.len() >= d, s.str[:d], "")


def score(labels_true, labels_pred) -> dict:
    return {
        "AMI": adjusted_mutual_info_score(labels_true, labels_pred),
        "NMI": normalized_mutual_info_score(labels_true, labels_pred),
        "ARI": adjusted_rand_score(labels_true, labels_pred),
    }


def cluster_and_score(X: np.ndarray, y, ks=(2, 4, 8, 16), methods=("kmeans", "agglo", "gmm"),
                      pca_dim: int | None = 50, seed: int = 0, hdbscan_min_cluster_size: int | None = 15) -> pd.DataFrame:
    """Run each clustering method for each k and return one row per (method, k) with AMI/NMI/ARI."""
    y = np.asarray(y)
    Z = X
    d = min(pca_dim or 0, X.shape[0] - 1, X.shape[1])
    if pca_dim and X.shape[1] > d > 1:
        Z = PCA(n_components=d, random_state=seed).fit_transform(X)
    rows = []
    for k in ks:
        for m in methods:
            if m == "kmeans":
                pred = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
            elif m == "agglo":
                pred = AgglomerativeClustering(k, linkage="ward").fit_predict(Z)
            elif m == "gmm":
                pred = GaussianMixture(k, covariance_type="diag", random_state=seed, n_init=2).fit_predict(Z)
            else:
                raise ValueError(m)
            rows.append({"method": m, "k": k, **score(y, pred)})
    if hdbscan_min_cluster_size:
        import hdbscan
        pred = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size).fit_predict(Z)
        n_cl = len(set(pred)) - (1 if -1 in pred else 0)
        rows.append({"method": "hdbscan", "k": n_cl, **score(y, pred), "noise_frac": float((pred == -1).mean())})
    return pd.DataFrame(rows)


def best_table(df: pd.DataFrame, by="AMI") -> pd.DataFrame:
    return df.sort_values(by, ascending=False).reset_index(drop=True)
