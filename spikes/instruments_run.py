"""Musical instruments spike, step 2: do instrument photos cluster by Hornbostel-Sachs class, by culture, or by era?

Labels: hs1 (Aerophone, Chordophone, Idiophone, Membranophone, ...), hs2 (second level, e.g. Chordophone-Lute),
culture (Met free text, first token), country, era. Within each hs1 class, culture is scored again.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings  # noqa: E402
from labels import run_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RUNS = ROOT / "data/instruments", ROOT / "runs/instruments"
ENCODERS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["clip_b32", "dinov2_s"]


def main():
    df = pd.read_csv(DATA / "objects.csv").fillna("")
    df["culture1"] = df.culture.astype(str).str.split(",|;|\\(").str[0].str.strip()
    df["region"] = df.country.astype(str).where(df.country.astype(str) != "", df.culture1)
    df["era"] = pd.cut(pd.to_numeric(df.year, errors="coerce"), [-5000, 1600, 1800, 1900, 2100],
                       labels=["<1600", "1600-1799", "1800-1899", "1900+"]).astype(str)
    print(df.hs1.value_counts().to_dict())
    print(df.era.value_counts().to_dict())
    print(df.region.value_counts().head(12).to_dict())
    paths = [str(DATA / f) for f in df.file]
    RUNS.mkdir(parents=True, exist_ok=True)
    out = []
    for enc in ENCODERS:
        X = cached_embeddings(RUNS / "emb", enc, paths)
        r = run_labels(X, df, ["hs1", "hs2", "region", "era"], ks=(4, 6, 10, 20, 40), encoder_name=enc)
        out.append(r)
        print(f"\n== {enc}, n={len(df)}")
        print(r.drop(columns=["encoder"]).to_string(index=False))
        for cls, g in df.groupby("hs1"):
            if g.region.nunique() >= 3 and len(g) >= 60:
                rr = run_labels(X[g.index.values], g.reset_index(drop=True), ["region", "era", "hs2"], ks=(3, 5, 8), encoder_name=enc)
                print(f"   within {cls} (n={len(g)}): " + "; ".join(
                    f"{a.label} AMI={a.best_AMI} probe={a.probe_acc} majority={a.majority}" for a in rr.itertuples() if a.classes >= 2))
    pd.concat(out).to_csv(RUNS / "scores.csv", index=False)


if __name__ == "__main__":
    main()
