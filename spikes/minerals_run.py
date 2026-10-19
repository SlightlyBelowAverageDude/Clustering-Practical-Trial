"""Minerals spike: do photos of mineral specimens cluster by crystal system, by chemical class (Strunz), or by species?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings  # noqa: E402
from labels import run_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/minerals"
PER_SPECIES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
ENCODERS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["clip_b32", "dinov2_s"]
SYSTEMS = "cubic|isometric|hexagonal|trigonal|tetragonal|orthorhombic|monoclinic|triclinic|amorphous"
# Species whose Wikipedia lookup failed (element pages, misspelt dataset names). Crystal system and
# Strunz class (1 elements, 2 sulfides, 4 oxides, 5 carbonates, 7 sulfates/tungstates, 8 phosphates,
# 9 silicates) from standard mineralogy references. Mineraloids and rocks are left out on purpose.
OVERRIDES = {
    "gold": ("cubic", "1"), "silver": ("cubic", "1"), "copper": ("cubic", "1"), "sulfur": ("orthorhombic", "1"),
    "arsenic": ("trigonal", "1"), "bismuth": ("trigonal", "1"),
    "antimonite": ("orthorhombic", "2"), "bournonite": ("orthorhombic", "2"), "cobaltin": ("orthorhombic", "2"),
    "sphalerite": ("cubic", "2"),
    "quartz": ("trigonal", "4"), "goethite": ("orthorhombic", "4"),
    "dolomite": ("trigonal", "5"), "siderite": ("trigonal", "5"),
    "celestine": ("orthorhombic", "7"), "scheelit": ("tetragonal", "7"),
    "apatite": ("hexagonal", "8"), "fluorapatite": ("hexagonal", "8"), "vivianite": ("monoclinic", "8"),
    "labrador": ("triclinic", "9"), "spodumene": ("monoclinic", "9"), "actinolite": ("monoclinic", "9"),
    "nephritis": ("monoclinic", "9"), "kyanite": ("triclinic", "9"), "cancrinit": ("hexagonal", "9"),
    "vesuvian": ("tetragonal", "9"), "prehnite": ("orthorhombic", "9"), "elbait": ("trigonal", "9"),
}


def main():
    ds = load_dataset("Nech-C/mineralimage5K-98", split="train")
    lab = pd.read_csv(ROOT / "data/minerals_labels.csv").fillna("")
    lab["system"] = lab.system.astype(str).str.lower().str.extract(f"({SYSTEMS})")[0].fillna("").replace({"isometric": "cubic"})
    lab["strunz_class"] = lab.strunz.astype(str).str.extract(r"^(\d+)")[0].fillna("")
    for name, (system, cls) in OVERRIDES.items():
        m = lab.name == name
        lab.loc[m & (lab.system == ""), "system"] = system
        lab.loc[m & (lab.strunz_class == ""), "strunz_class"] = cls
    labels = np.array(ds["name"])
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(labels == s)[0], size=min(PER_SPECIES, int((labels == s).sum())), replace=False)
                          for s in np.unique(labels)])
    sub = ds.select(idx)
    df = pd.DataFrame({"label": labels[idx]}).merge(lab, on="label", how="left")
    u = df.drop_duplicates("label")
    print(f"species with crystal system: {(u.system != '').sum()} of {len(u)}; with Strunz class: {(u.strunz_class != '').sum()}")
    print("systems:", u.system.value_counts().to_dict())
    print("Strunz classes:", u.strunz_class.value_counts().to_dict())
    images = [ex["image"] for ex in sub]
    RUNS.mkdir(parents=True, exist_ok=True)
    out = []
    for enc in ENCODERS:
        X = cached_embeddings(RUNS / f"emb_{PER_SPECIES}", enc, images)
        r = run_labels(X, df, ["name", "system", "strunz_class"], ks=(7, 10, 20, 50, 98), encoder_name=enc)
        out.append(r)
        print(f"\n== {enc}, n={len(df)}")
        print(r.drop(columns=["encoder"]).to_string(index=False))
    pd.concat(out).to_csv(RUNS / "scores.csv", index=False)


if __name__ == "__main__":
    main()
