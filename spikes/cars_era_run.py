"""Cars over decades (VMMRdb, 1950 to 2016): did car design converge across countries of origin?

VMMRdb folders are named make_model_year. Per decade, a balanced sample of images is embedded; then
(a) AMI of clusters against decade, body-agnostic, and against maker country; (b) per decade, how well a
linear probe separates maker countries (chance-corrected). A falling per-decade country separability is
the convergence signal. Within-decade AMI vs country is reported too.

Usage: python spikes/cars_era_run.py <per_decade> [encoders]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings  # noqa: E402
from labels import run_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dl/vmmrdb"
RUNS = ROOT / "runs/cars_era"
PER_DECADE = int(sys.argv[1]) if len(sys.argv) > 1 else 400
ENCODERS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["clip_b32", "dinov2_s"]
COUNTRY = {"acura": "JP", "honda": "JP", "toyota": "JP", "lexus": "JP", "nissan": "JP", "infiniti": "JP", "mazda": "JP", "subaru": "JP",
           "mitsubishi": "JP", "suzuki": "JP", "isuzu": "JP", "datsun": "JP", "scion": "JP",
           "hyundai": "KR", "kia": "KR", "daewoo": "KR",
           "audi": "DE", "bmw": "DE", "mercedes": "DE", "mercedes-benz": "DE", "porsche": "DE", "volkswagen": "DE", "vw": "DE", "opel": "DE", "smart": "DE",
           "ford": "US", "chevrolet": "US", "chevy": "US", "dodge": "US", "chrysler": "US", "jeep": "US", "gmc": "US", "cadillac": "US", "buick": "US",
           "pontiac": "US", "oldsmobile": "US", "lincoln": "US", "mercury": "US", "plymouth": "US", "saturn": "US", "hummer": "US", "ram": "US",
           "tesla": "US", "studebaker": "US", "packard": "US", "amc": "US", "desoto": "US", "nash": "US", "hudson": "US", "eagle": "US", "geo": "US",
           "jaguar": "UK", "land": "UK", "rover": "UK", "mini": "UK", "bentley": "UK", "rolls-royce": "UK", "aston": "UK", "lotus": "UK", "mg": "UK", "triumph": "UK",
           "fiat": "IT", "ferrari": "IT", "lamborghini": "IT", "maserati": "IT", "alfa": "IT", "lancia": "IT",
           "volvo": "SE", "saab": "SE", "peugeot": "FR", "renault": "FR", "citroen": "FR"}


def parse(folder):
    toks = folder.lower().split("_")
    m = re.match(r"^(\d{4})$", toks[-1])
    if not m:
        return None
    year = int(m.group(1))
    make = toks[0]
    return make, COUNTRY.get(make, ""), "_".join(toks[1:-1]), year


def main():
    root = next((p for p in SRC.rglob("*") if p.is_dir() and parse(p.name)), None)
    if root is None:
        raise SystemExit("no make_model_year folders found under dl/vmmrdb")
    base = root.parent
    by_decade = defaultdict(list)
    for d in base.iterdir():
        p = parse(d.name) if d.is_dir() else None
        if p and p[1] and 1950 <= p[3] <= 2016:
            by_decade[p[3] // 10 * 10].append((d, p))
    rng = np.random.default_rng(0)
    rows = []
    for dec, folders in sorted(by_decade.items()):
        # spread the per-decade budget over as many models as possible, 3 images per model max
        rng.shuffle(folders)
        for d, (make, country, model, year) in folders:
            if sum(r["decade"] == dec for r in rows) >= PER_DECADE:
                break
            imgs = [f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
            for f in imgs[:3]:
                rows.append({"file": str(f), "make": make, "country": country, "model": model, "year": year, "decade": dec})
    df = pd.DataFrame(rows)
    df["decade"] = df.decade.astype(str)
    print(df.groupby("decade").size().to_dict())
    print(df.groupby(["decade", "country"]).size().unstack(fill_value=0).to_string())
    RUNS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUNS / f"sample_{PER_DECADE}.csv", index=False)
    for enc in ENCODERS:
        X = cached_embeddings(RUNS / f"emb_{PER_DECADE}", enc, df.file.tolist())
        r = run_labels(X, df, ["decade", "country", "make"], ks=(5, 7, 10, 20, 50), encoder_name=enc)
        print(f"\n== {enc}, n={len(df)}")
        print(r.drop(columns=["encoder"]).to_string(index=False))
        print("   per decade: country probe accuracy minus majority baseline, and AMI of clusters vs country")
        for dec, g in df.groupby("decade"):
            vc = g.country.value_counts()
            keep = g.country.isin(vc[vc >= 15].index)
            gg = g[keep]
            if gg.country.nunique() < 3:
                print(f"   {dec}: fewer than 3 countries with 15+ images")
                continue
            Xg, y = X[gg.index.values], np.array(gg.country.tolist(), dtype=object)
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            probe = cross_val_score(LogisticRegression(max_iter=3000, C=0.5), Xg, y, cv=cv).mean()
            maj = vc[vc >= 15].max() / keep.sum()
            rr = run_labels(Xg, gg.reset_index(drop=True), ["country"], ks=(gg.country.nunique(),), encoder_name=enc, min_count=15)
            print(f"   {dec}s: n={len(gg)}, countries={gg.country.nunique()}, probe-majority={probe - maj:+.3f} (probe {probe:.3f}),"
                  f" AMI vs country={rr.best_AMI.iloc[0]:.3f}")


if __name__ == "__main__":
    main()
