"""Insects spike: where do image clusters and DNA clusters (BINs) disagree?

BIOSCAN-30k slice (FiftyOne export of BIOSCAN-5M, CC BY 3.0). Labels: order, family, genus, DNA barcode
index number (BIN). Overall AMI of image clusters against each rank, then per family: how well image
clusters recover BINs (low = visually cryptic lineages, or convergent look-alikes).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings  # noqa: E402
from labels import run_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RUNS = ROOT / "data/bioscan30k", ROOT / "runs/insects"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
ENCODERS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["clip_b32", "dinov2_s"]


def label(sample, key):
    v = sample.get(key)
    if isinstance(v, dict):
        return v.get("label") or v.get("value") or ""  # Classification fields use "label", Attribute fields "value"
    return v if isinstance(v, str) else ""


def main():
    samples = json.load(open(DATA / "samples.json", encoding="utf-8"))["samples"]
    keys = [k for k in samples[0] if not k.startswith("_")]
    bin_key = next((k for k in keys if "bin" in k.lower()), None)
    print("fields:", keys, "| BIN field:", bin_key)
    rows = []
    for s in samples:
        f = DATA / s["filepath"]
        if not f.exists():
            continue
        rows.append({"file": str(f), "order": label(s, "order"), "family": label(s, "family"), "genus": label(s, "genus"),
                     "species": label(s, "species"), "country": label(s, "country"), "bin": label(s, bin_key) if bin_key else ""})
    df = pd.DataFrame(rows)
    df = df[(df.order != "") & (df.family != "")].sample(n=min(N, len(df)), random_state=0).reset_index(drop=True)
    print(f"{len(df)} images; orders {df.order.nunique()}, families {df.family.nunique()}, genera {(df.genus != '').sum()} labelled,"
          f" BINs {df.bin.nunique()}")
    print(df.order.value_counts().head(8).to_dict())
    RUNS.mkdir(parents=True, exist_ok=True)
    out = []
    for enc in ENCODERS:
        X = cached_embeddings(RUNS / f"emb_{N}", enc, df.file.tolist())
        r = run_labels(X, df, ["order", "family", "genus", "bin"], ks=(10, 20, 50, 100, 200), encoder_name=enc, min_count=5)
        out.append(r)
        print(f"\n== {enc}, n={len(df)}")
        print(r.drop(columns=["encoder"]).to_string(index=False))
        # per family: image clusters vs BINs (families with enough images and at least 3 BINs with 5+ images)
        fam_rows = []
        for fam, g in df.groupby("family"):
            vc = g.bin.value_counts()
            vc = vc[(vc >= 5) & (vc.index != "")]
            if len(g) < 40 or len(vc) < 3:
                continue
            gg = g[g.bin.isin(vc.index)]
            rr = run_labels(X[gg.index.values], gg.reset_index(drop=True), ["bin"], ks=(len(vc),), encoder_name=enc, min_count=5)
            fam_rows.append({"family": fam, "order": g.order.iloc[0], "n": len(gg), "bins": len(vc),
                             "AMI_img_vs_bin": rr.best_AMI.iloc[0], "probe_acc": rr.probe_acc.iloc[0], "majority": rr.majority.iloc[0]})
        fr = pd.DataFrame(fam_rows).sort_values("AMI_img_vs_bin")
        print(f"   per-family image-cluster vs BIN agreement ({len(fr)} families), lowest and highest:")
        print(pd.concat([fr.head(5), fr.tail(5)]).to_string(index=False))
        fr.to_csv(RUNS / f"per_family_{enc}.csv", index=False)
    pd.concat(out).to_csv(RUNS / "scores.csv", index=False)


if __name__ == "__main__":
    main()
