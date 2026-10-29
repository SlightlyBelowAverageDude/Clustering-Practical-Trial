"""Rock-bottom baseline: raw DINOv2-B embeddings, no preprocessing, standard k-means, AMI.

Sets (argument, default "same"):
  same   the 27,348 same rows (runs/clean/same_rows_ids.csv); labels hs1 to hs6, family, name as in same_rows.csv
  coded  all 40,697 objects with a Hornbostel-Sachs code; labels hs1 to hs6 as in depth.csv
Embeddings: a fresh pass with the model's own default transform (resize the short side to 518 px, centre crop, ImageNet
mean and std), float32, the pooled output as it comes out of the model; no padding, no background removal, no L2
normalisation, no PCA. Clustering: sklearn KMeans with its defaults (k-means++ init, 1 run, 300 iterations),
k = number of classes (classes with 30+ objects, deeper levels on the rows whose code is long enough), 5 seeds, plus a
20-seed sweep at level 3 with k-means++ and with random init.

For orientation the same k-means is also run on 3 other inputs: the raw embeddings L2-normalised; the published
embeddings (padded images, L2-normalised) without PCA; and the project pipeline (published embeddings, PCA 50,
k-means with 10 runs on the same rows, 5 on the coded rows), which must reproduce same_rows.csv and depth.csv.

Outputs in runs/tests/rock_bottom/: emb_dinov2_b_raw[_coded].npy (+ .key), ami[_coded].csv (one row per variant,
label, seed), summary[_coded].csv, sweep[_coded].csv
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.preprocessing import normalize
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, Encoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/tests/rock_bottom"
MIN = 30
SEEDS = (0, 1, 2, 3, 4)
SETS = {"same": {"suffix": "", "labels": ("hs1", "hs2", "hs3", "hs4", "hs5", "hs6", "family", "name"), "pipeline_n_init": 10},
        "coded": {"suffix": "_coded", "labels": ("hs1", "hs2", "hs3", "hs4", "hs5", "hs6"), "pipeline_n_init": 5}}


def kept(y):
    y = np.array(pd.Series(y).astype(str).tolist(), dtype=object)
    vc = pd.Series(y).value_counts()
    return (pd.Series(y).map(vc).values >= MIN) & (y != "") & (y != "nan"), y


@torch.no_grad()
def embed_raw(paths, batch_size=64):
    """DINOv2-B through timm's default transform, float32, no normalisation of the output."""
    from PIL import Image
    enc = Encoder("dinov2_b", pad_square=False)
    out = []
    for i in tqdm(range(0, len(paths), batch_size), desc="embed raw", leave=False):
        batch = [enc.preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]
        f = enc.model(torch.stack(batch).to(DEVICE))
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def cached_raw(paths, suffix):
    """Raw embeddings for these paths in this order; rows already embedded for another set are reused by path."""
    f, kf = OUT / f"emb_dinov2_b_raw{suffix}.npy", OUT / f"emb_dinov2_b_raw{suffix}.key"
    key = hashlib.md5("\n".join(paths).encode("utf-8")).hexdigest()
    if f.exists() and kf.exists() and kf.read_text().strip() == key:
        return np.load(f)
    have = {}
    for other in OUT.glob("emb_dinov2_b_raw*.npy"):
        okf = other.with_suffix(".key")
        plist = other.with_suffix(".paths")
        if okf.exists() and plist.exists():
            ps = plist.read_text(encoding="utf-8").split("\n")
            if hashlib.md5("\n".join(ps).encode("utf-8")).hexdigest() == okf.read_text().strip():
                E = np.load(other)
                have.update(zip(ps, E))
    todo = [p for p in paths if p not in have]
    print(f"{len(paths) - len(todo)} embeddings reused, {len(todo)} to compute", flush=True)
    if todo:
        have.update(zip(todo, embed_raw(todo)))
    X = np.stack([have[p] for p in paths]).astype(np.float32)
    np.save(f, X)
    kf.write_text(key)
    f.with_suffix(".paths").write_text("\n".join(paths), encoding="utf-8")
    return X


def main(set_name):
    cfg = SETS[set_name]
    sfx = cfg["suffix"]
    OUT.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    if set_name == "same":
        ids = pd.read_csv(ROOT / "runs/clean/same_rows_ids.csv", dtype=str).id
        pos = dict(zip(rec.id, range(len(rec))))
        idx = np.array([pos[i] for i in ids])
    else:
        idx = np.where((rec.hs1 != "").values)[0]
    sub = rec.iloc[idx].reset_index(drop=True)
    paths = [str(ROOT / "data/mimo" / f) for f in sub.file]
    print(set_name, len(sub), "objects", flush=True)

    X_raw = cached_raw(paths, sfx)
    X_pub = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")[idx]
    norms = np.linalg.norm(X_raw, axis=1)
    print(f"raw embeddings {X_raw.shape}, norm {norms.mean():.1f} ({norms.min():.1f} to {norms.max():.1f}), "
          f"mean {X_raw.mean():.3f}, std {X_raw.std():.3f}", flush=True)

    n_pipe = cfg["pipeline_n_init"]
    inputs = {
        "raw, no preprocessing": (X_raw, 1),
        "raw, L2 only": (normalize(X_raw), 1),
        "published (padded, L2), no PCA": (X_pub, 1),
        f"project pipeline (padded, L2, PCA 50, {n_pipe} runs)": (PCA(50, random_state=0).fit_transform(X_pub), n_pipe),
    }
    rows = []
    for variant, (Z, n_init) in inputs.items():
        for lab in cfg["labels"]:
            keep, y = kept(sub[lab].values)
            yk, Zk = y[keep], Z[keep]
            k = len(set(yk))
            for seed in SEEDS:
                if n_init == 1:
                    pred = KMeans(n_clusters=k, random_state=seed).fit_predict(Zk)
                else:
                    if seed != 0:
                        continue
                    pred = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(Zk)
                r = {"variant": variant, "label": lab, "n": int(keep.sum()), "classes": k, "seed": seed, "AMI": round(float(ami(yk, pred)), 4)}
                rows.append(r)
                print(r, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"ami{sfx}.csv", index=False)
    s = df.groupby(["variant", "label", "n", "classes"], sort=False).AMI.agg(["mean", "min", "max", "count"]).round(3).reset_index()
    s.to_csv(OUT / f"summary{sfx}.csv", index=False)
    print(s.to_string(), flush=True)

    keep, y = kept(sub["hs3"].values)
    yk, Xk = y[keep], X_raw[keep]
    k = len(set(yk))
    sw = []
    for init in ("k-means++", "random"):
        for seed in range(20):
            sw.append({"init": init, "seed": seed, "AMI": round(float(ami(yk, KMeans(n_clusters=k, init=init, random_state=seed).fit_predict(Xk))), 4)})
    sw = pd.DataFrame(sw)
    sw.to_csv(OUT / f"sweep{sfx}.csv", index=False)
    print(f"level 3 sweep, raw, k = {k}, 20 seeds:", flush=True)
    print(sw.groupby("init").AMI.agg(["mean", "min", "max"]).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "same")
