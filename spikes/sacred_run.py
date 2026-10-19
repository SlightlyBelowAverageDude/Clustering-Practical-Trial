"""Sacred buildings spike: do photos of religious buildings cluster by religion (class), country, or era?"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings  # noqa: E402
from labels import run_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RUNS = ROOT / "data/sacred", ROOT / "runs/sacred"
ENCODERS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["clip_b32", "dinov2_s"]


def main():
    df = pd.read_csv(DATA / "buildings.csv")
    df["era"] = pd.cut(df.year, [-10000, 1000, 1500, 1800, 1900, 2100],
                       labels=["<1000", "1000-1499", "1500-1799", "1800-1899", "1900+"]).astype(str)
    print(df.groupby("cls").size().to_dict())
    print(df.era.value_counts().to_dict())
    paths = [str(DATA / f) for f in df.file]
    RUNS.mkdir(parents=True, exist_ok=True)
    out = []
    for enc in ENCODERS:
        X = cached_embeddings(RUNS / "emb", enc, paths)
        r = run_labels(X, df, ["cls", "country", "era"], ks=(5, 6, 10, 20), encoder_name=enc)
        out.append(r)
        print(f"\n== {enc}, n={len(df)}")
        print(r.drop(columns=["encoder"]).to_string(index=False))
        for cls, g in df.groupby("cls"):
            if g.country.nunique() >= 3 and len(g) >= 60:
                rr = run_labels(X[g.index.values], g.reset_index(drop=True), ["country", "era"], ks=(3, 5, 8), encoder_name=enc)
                print(f"   within {cls} (n={len(g)}): " + "; ".join(
                    f"{a.label} AMI={a.best_AMI} probe={a.probe_acc} majority={a.majority}" for a in rr.itertuples()))
    pd.concat(out).to_csv(RUNS / "scores.csv", index=False)


if __name__ == "__main__":
    main()
