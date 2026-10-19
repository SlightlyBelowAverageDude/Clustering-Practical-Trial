"""Photography-confound test for the instrument spike.

3 image variants per object: raw photo, object only (background removed with rembg, white fill), and
background only (the outer 15 percent frame of the photo, centre painted mid-grey). If the region
stays recoverable from the background-only crops, the signal is photography, not the object.

Usage: python spikes/confound_run.py [encoders, comma-separated]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from rembg import new_session, remove
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_embeddings  # noqa: E402
from labels import run_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["clip_b32", "dinov2_s"]
CFG = {"data": "data/instruments", "primary": "hs1", "secondary": "region"}
DATA, RUNS = ROOT / CFG["data"], ROOT / "runs/confound_instruments"


def make_variants(df):
    (DATA / "object_only").mkdir(exist_ok=True)
    (DATA / "background_only").mkdir(exist_ok=True)
    sess = new_session("u2net")
    for f in tqdm(df.file, desc="variants", leave=False):
        src = DATA / f
        o, b = DATA / "object_only" / src.name, DATA / "background_only" / src.name
        if o.exists() and b.exists():
            continue
        im = Image.open(src).convert("RGB")
        if not o.exists():
            remove(im, session=sess, bgcolor=(255, 255, 255, 255)).convert("RGB").save(o, quality=92)
        if not b.exists():
            bg = im.copy()
            w, h = bg.size
            m = 0.15
            ImageDraw.Draw(bg).rectangle([int(w * m), int(h * m), int(w * (1 - m)), int(h * (1 - m))], fill=(128, 128, 128))
            bg.save(b, quality=92)


def main():
    df = pd.read_csv(DATA / "objects.csv").fillna("")
    df["culture1"] = df.culture.astype(str).str.split(",|;|\\(").str[0].str.strip()
    df["region"] = df.country.astype(str).where(df.country.astype(str) != "", df.culture1)
    make_variants(df)
    RUNS.mkdir(parents=True, exist_ok=True)
    prim, sec = CFG["primary"], CFG["secondary"]
    rows = []
    for variant, folder in (("raw", "images"), ("object_only", "object_only"), ("background_only", "background_only")):
        paths = [str(DATA / folder / Path(f).name) for f in df.file]
        for enc in ENCODERS:
            X = cached_embeddings(RUNS / f"emb_{variant}", enc, paths)
            r = run_labels(X, df, [prim, sec], ks=(5, 7, 10, 20), encoder_name=enc)
            r.insert(0, "variant", variant)
            rows.append(r)
            # secondary label inside each primary class
            within = []
            for cls, g in df.groupby(prim):
                if g[sec].nunique() >= 3 and len(g) >= 50:
                    rr = run_labels(X[g.index.values], g.reset_index(drop=True), [sec], ks=(3, 5, 8), encoder_name=enc)
                    if len(rr) and rr.classes.iloc[0] >= 2:
                        within.append({"variant": variant, "encoder": enc, "label": f"{sec} within {cls}", "n": rr.n.iloc[0],
                                       "classes": rr.classes.iloc[0], "best_AMI": rr.best_AMI.iloc[0], "probe_acc": rr.probe_acc.iloc[0],
                                       "majority": rr.majority.iloc[0]})
            rows.append(pd.DataFrame(within))
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(RUNS / "scores.csv", index=False)
    pd.set_option("display.width", 200)
    for enc in ENCODERS:
        print(f"\n== instruments, {enc}: variant x label (AMI best, probe, majority)")
        sub = res[res.encoder == enc].pivot_table(index="label", columns="variant", values=["best_AMI", "probe_acc"], aggfunc="first")
        sub = sub.reindex(columns=["raw", "object_only", "background_only"], level=1)
        print(sub.round(3).to_string())
        print("majority baselines:", res[res.encoder == enc].drop_duplicates("label").set_index("label").majority.round(3).to_dict())


if __name__ == "__main__":
    main()
