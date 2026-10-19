"""Dogs spike: do visual clusters of dog breeds follow kennel-club function groups or country of origin?

Stanford Dogs (120 breeds) with per-breed FCI group, AKC group and country from Wikipedia infoboxes.
Breed itself is a label too: the ceiling, since a group label can only be recovered if breeds are.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings  # noqa: E402
from labels import run_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/dogs"
PER_BREED = int(sys.argv[1]) if len(sys.argv) > 1 else 25
ENCODERS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["clip_b32", "dinov2_s"]


def main():
    ds = load_dataset("maurice-fp/stanford-dogs", split="train")
    lab = pd.read_csv(ROOT / "data/dogs_labels.csv").fillna("")
    labels = np.array(ds["label"])
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(labels == b)[0], size=min(PER_BREED, int((labels == b).sum())), replace=False)
                          for b in np.unique(labels)])
    sub = ds.select(idx)
    df = pd.DataFrame({"label": labels[idx]}).merge(lab, on="label", how="left")
    df["fcigroup"] = df.fcigroup.astype(str).str.extract(r"(\d+)")[0].fillna("")
    df["country"] = df.country.astype(str).str.split(",").str[0].str.strip()
    print("breeds per FCI group:", df.drop_duplicates("breed").groupby("fcigroup").size().to_dict())
    print("breeds per AKC group:", df.drop_duplicates("breed").groupby("akcgroup").size().to_dict())
    images = [ex["image"] for ex in sub]
    RUNS.mkdir(parents=True, exist_ok=True)
    out = []
    for enc in ENCODERS:
        X = cached_embeddings(RUNS / f"emb_{PER_BREED}", enc, images)
        r = run_labels(X, df, ["breed", "fcigroup", "akcgroup", "country"], ks=(8, 10, 20, 40, 120), encoder_name=enc)
        out.append(r)
        print(f"\n== {enc}, n={len(df)}")
        print(r.drop(columns=["encoder"]).to_string(index=False))
    pd.concat(out).to_csv(RUNS / "scores.csv", index=False)


if __name__ == "__main__":
    main()
