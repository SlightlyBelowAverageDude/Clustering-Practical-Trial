"""Cars spike: do car photos cluster by body type (function), by the maker's country (design tradition), or by model year?

Stanford Cars (196 classes "Make Model Body Year"). Body type is the second-to-last token of the class
name, country is mapped from the make, year is the last token.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings  # noqa: E402
from labels import run_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/cars"
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
ENCODERS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["clip_b32", "dinov2_s"]
COUNTRY = {"AM General": "US", "Acura": "JP", "Aston Martin": "UK", "Audi": "DE", "BMW": "DE", "Bentley": "UK", "Bugatti": "FR",
           "Buick": "US", "Cadillac": "US", "Chevrolet": "US", "Chrysler": "US", "Daewoo": "KR", "Dodge": "US", "Eagle": "US",
           "FIAT": "IT", "Ferrari": "IT", "Fisker": "US", "Ford": "US", "GMC": "US", "Geo": "US", "HUMMER": "US", "Honda": "JP",
           "Hyundai": "KR", "Infiniti": "JP", "Isuzu": "JP", "Jaguar": "UK", "Jeep": "US", "Lamborghini": "IT", "Land Rover": "UK",
           "Lincoln": "US", "MINI": "UK", "Maybach": "DE", "Mazda": "JP", "McLaren": "UK", "Mercedes-Benz": "DE", "Mitsubishi": "JP",
           "Nissan": "JP", "Plymouth": "US", "Porsche": "DE", "Ram": "US", "Rolls-Royce": "UK", "Scion": "JP", "Spyker": "NL",
           "Suzuki": "JP", "Tesla": "US", "Toyota": "JP", "Volkswagen": "DE", "Volvo": "SE", "smart": "DE"}
BODY = {"SUV", "Sedan", "Coupe", "Convertible", "Hatchback", "Wagon", "Van", "Minivan", "Cab"}


def parse(name):
    toks = name.split()
    year = toks[-1]
    body = toks[-2] if toks[-2] in BODY else ("Cab" if "Cab" in name else "other")
    make = next((m for m in sorted(COUNTRY, key=len, reverse=True) if name.startswith(m + " ")), toks[0])
    return make, COUNTRY.get(make, ""), body, year


def main():
    ds = load_dataset("tanganke/stanford_cars", split="train")
    names = ds.features["label"].names
    labels = np.array(ds["label"])
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(labels == c)[0], size=min(PER_CLASS, int((labels == c).sum())), replace=False)
                          for c in np.unique(labels)])
    sub = ds.select(idx)
    meta = pd.DataFrame([parse(names[c]) for c in labels[idx]], columns=["make", "country", "body", "year"])
    meta["model"] = [names[c] for c in labels[idx]]
    meta["era"] = pd.cut(meta.year.astype(int), [1990, 2005, 2010, 2013], labels=["1991-2005", "2006-2010", "2011-2012"]).astype(str)
    print("countries:", meta.drop_duplicates("model").country.value_counts().to_dict())
    print("bodies:", meta.drop_duplicates("model").body.value_counts().to_dict())
    print("eras:", meta.drop_duplicates("model").era.value_counts().to_dict())
    images = [ex["image"] for ex in sub]
    RUNS.mkdir(parents=True, exist_ok=True)
    out = []
    for enc in ENCODERS:
        X = cached_embeddings(RUNS / f"emb_{PER_CLASS}", enc, images)
        r = run_labels(X, meta, ["model", "body", "country", "make", "era"], ks=(3, 5, 7, 9, 20, 50, 196), encoder_name=enc)
        out.append(r)
        print(f"\n== {enc}, n={len(meta)}")
        print(r.drop(columns=["encoder"]).to_string(index=False))
    pd.concat(out).to_csv(RUNS / "scores.csv", index=False)


if __name__ == "__main__":
    main()
