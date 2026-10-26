"""Background removal for the gap test: object-only copies of the MIMO thumbnails in the 12,000-row methods subset.

rembg u2net mask, object composited on white. Writes data/mimo_objonly/<file name> and runs/tests/gap/objonly_mask.csv
(share of the image the mask keeps). Resumable: existing outputs are skipped.
Usage: python spikes/mimo_objonly.py [workers]
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/mimo"
DST = ROOT / "data/mimo_objonly"
OUT = ROOT / "runs/tests/gap"
_sess = None


def init():
    global _sess
    os.environ["OMP_NUM_THREADS"] = "4"
    from rembg import new_session
    _sess = new_session("u2net")


def work(f):
    from rembg import remove
    src, dst = SRC / f, DST / Path(f).name
    im = Image.open(src).convert("RGB")
    mask = remove(im, session=_sess, only_mask=True).convert("L")
    m = np.asarray(mask, dtype=np.float32) / 255.0
    if not dst.exists():
        a = np.asarray(im, dtype=np.float32)
        out = a * m[..., None] + 255.0 * (1 - m[..., None])
        Image.fromarray(out.round().clip(0, 255).astype(np.uint8)).save(dst, quality=92)
    return f, float(m.mean())


def main():
    DST.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    ids = pd.read_csv(ROOT / "runs/clean/methods_subset_ids.csv", dtype=str).id
    files = rec.set_index("id").loc[ids, "file"].tolist()
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    rows = []
    with Pool(workers, initializer=init) as pool:
        for i, r in enumerate(pool.imap(work, files, chunksize=8)):
            rows.append(r)
            if i % 500 == 0:
                print(f"{i} of {len(files)}", flush=True)
    pd.DataFrame(rows, columns=["file", "mask_share"]).to_csv(OUT / "objonly_mask.csv", index=False)
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
