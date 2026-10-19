"""Instruments: encoder comparison on object-only crops with bootstrap intervals, and zero-shot text criteria.

Encoders: CLIP B/32, CLIP L/14, SigLIP B/16, DINOv2-S, DINOv2-B. For each: H-S level 1 and 2 and region AMI
(best of k-means and agglomerative at k in {4, 10, 20, 40}) with a 95 percent bootstrap interval over
objects (200 resamples, k-means at k = classes). Zero-shot: CLIP and SigLIP text prompts for the 4 H-S
classes in plain words, and for the top regions, accuracy against the labels. That shows whether a text
criterion could steer the clustering (Topic 3) for instruments.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import Encoder, cached_embeddings, score  # noqa: E402
from labels import run_labels  # noqa: E402

common.ENCODERS["siglip_b16"] = ("open_clip", "ViT-B-16-SigLIP", "webli")
ROOT = Path(__file__).resolve().parents[1]
DATA, RUNS = ROOT / "data/instruments", ROOT / "runs/instruments_encoders"
ENCODERS = ["clip_b32", "clip_l14", "siglip_b16", "dinov2_s", "dinov2_b"]
HS_PROMPTS = {"Aerophone": "a photo of a wind instrument", "Chordophone": "a photo of a string instrument",
              "Idiophone": "a photo of a percussion instrument that is struck or shaken, such as a bell, rattle or xylophone",
              "Membranophone": "a photo of a drum"}


def boot_ami(X, y, k, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    Z = PCA(min(50, X.shape[1]), random_state=seed).fit_transform(X)
    vals = []
    for r in range(reps):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        vals.append(score(y[i], KMeans(k, n_init=3, random_state=r).fit_predict(Z[i]))["AMI"])
    return np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    df = pd.read_csv(DATA / "objects.csv").fillna("")
    df["culture1"] = df.culture.astype(str).str.split(",|;|\\(").str[0].str.strip()
    df["region"] = df.country.astype(str).where(df.country.astype(str) != "", df.culture1)
    paths = [str(DATA / "object_only" / Path(f).name) for f in df.file]
    RUNS.mkdir(parents=True, exist_ok=True)
    rows, zs = [], []
    for enc in ENCODERS:
        X = cached_embeddings(RUNS / "emb_object_only", enc, paths)
        r = run_labels(X, df, ["hs1", "hs2", "region"], ks=(4, 10, 20, 40), encoder_name=enc)
        for lab in ("hs1", "hs2", "region"):
            vc = df[lab].value_counts(); keep = df[lab].isin(vc[vc >= 5].index).values & (df[lab] != "").values
            y = np.array(df[lab][keep].tolist(), dtype=object)
            m, lo, hi = boot_ami(X[keep], y, len(set(y)))
            r.loc[r.label == lab, "AMI_k_classes_boot"] = f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(r)
        print(f"\n== {enc}"); print(r.drop(columns=["encoder"]).to_string(index=False), flush=True)
        if common.ENCODERS[enc][0] == "open_clip":
            e = Encoder(enc)
            T = e.embed_texts(list(HS_PROMPTS.values()))
            pred = np.array(list(HS_PROMPTS))[(X @ T.T).argmax(1)]
            acc = float((pred == df.hs1.values).mean())
            top = [c for c in df.region.value_counts().index if c][:8]
            Tr = e.embed_texts([f"a photo of a musical instrument from {c}" for c in top])
            m = df.region.isin(top).values
            predr = np.array(top)[(X[m] @ Tr.T).argmax(1)]
            accr = float((predr == df.region.values[m]).mean())
            zs.append({"encoder": enc, "zero-shot H-S level 1 acc": round(acc, 3), "majority": round(df.hs1.value_counts(normalize=True).iloc[0], 3),
                       "zero-shot region acc (top 8)": round(accr, 3), "majority region": round(df.region[m].value_counts(normalize=True).iloc[0], 3),
                       "AMI zero-shot H-S vs label": round(score(df.hs1.values, pred)["AMI"], 3)})
            print("   zero-shot:", zs[-1], flush=True)
            del e; torch.cuda.empty_cache()
    pd.concat(rows).to_csv(RUNS / "scores.csv", index=False)
    pd.DataFrame(zs).to_csv(RUNS / "zero_shot.csv", index=False)
    print("\n== zero-shot summary"); print(pd.DataFrame(zs).to_string(index=False))


if __name__ == "__main__":
    main()
