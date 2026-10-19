"""Galaxies spike: Galaxy10 DECaLS (17,736 images, 10 morphology classes) from Hugging Face.

Embeds a random sample with frozen encoders, clusters, scores against the 10 classes. Also prints
zero-shot CLIP accuracy with plain class prompts as a hint of whether text criteria can steer anything.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Encoder, cached_embeddings, cluster_and_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/galaxies"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
ENCODERS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["clip_b32", "dinov2_s"]
CLASSES = ["disturbed galaxy", "merging galaxy", "round smooth galaxy", "in-between round smooth galaxy",
           "cigar shaped smooth galaxy", "barred spiral galaxy", "unbarred tight spiral galaxy",
           "unbarred loose spiral galaxy", "edge-on galaxy without bulge", "edge-on galaxy with bulge"]


def main():
    ds = load_dataset("matthieulel/galaxy10_decals", split="train")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(N, len(ds)), replace=False)
    sub = ds.select(idx)
    images, y = [ex["image"] for ex in sub], np.array([ex["label"] for ex in sub])
    print("class counts:", np.bincount(y))
    out = []
    for enc in ENCODERS:
        X = cached_embeddings(RUNS / f"emb_{N}", enc, images)
        res = cluster_and_score(X, y, ks=(5, 10, 15, 20), hdbscan_min_cluster_size=20)
        res.insert(0, "encoder", enc)
        out.append(res)
        print(f"\n== {enc}, n={len(y)}")
        print(res.drop(columns=["encoder"]).round(3).to_string(index=False))
        if enc.startswith("clip"):
            e = Encoder(enc)
            T = e.embed_texts([f"a telescope image of a {c}" for c in CLASSES])
            zs = (X @ T.T).argmax(1)
            print(f"zero-shot accuracy with class prompts: {(zs == y).mean():.3f} (chance 0.1)")
    pd.concat(out).to_csv(RUNS / "scores.csv", index=False)


if __name__ == "__main__":
    main()
