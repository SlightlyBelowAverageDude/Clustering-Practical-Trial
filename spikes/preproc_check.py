"""Does the centre crop in the default DINOv2 transform hurt? Compare the default transform (resize short side to 518,
centre crop 518x518, which cuts the ends of non-square thumbnails) with pad-to-square (white) then resize, on a sample
of MIMO objects with H-S code, family and name. Scores: k-means AMI at k = classes, probe accuracy. Output: runs/preproc_check.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/preproc_check"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "runs/mimo_full/sample_full.csv", dtype=str).fillna("")
    lab = pd.read_csv(ROOT / "runs/mimo_names/labels.csv", dtype=str).fillna("")
    df["name"], df["family"] = lab.name.values, lab.family.values
    df["hs3"] = df.notation.str.replace(".", "", regex=False).str[:3]
    m = (df.name != "") & (df.family != "") & (df.hs3.str.len() == 3)
    sub = df[m]
    for col in ("hs3", "family", "name"):
        vc = sub[col].value_counts(); sub = sub[sub[col].map(vc) >= 30]
    sub = sub.sample(N, random_state=0).reset_index(drop=True)
    sizes = [Image.open(ROOT / "data/mimo" / f).size for f in sub.file]
    ar = np.array([max(w, h) / min(w, h) for w, h in sizes])
    print(f"{len(sub)} objects; aspect ratio median {np.median(ar):.2f}, share > 1.33: {(ar > 1.33).mean():.2f}, share > 2: {(ar > 2).mean():.2f}", flush=True)

    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0).cuda().eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    default_tf = timm.data.create_transform(**cfg)

    def pad_tf(im):
        im = ImageOps.pad(im, (max(im.size), max(im.size)), color=(255, 255, 255))  # square canvas, object intact
        return default_tf(im)

    def embed(tf):
        out = []
        for i in range(0, len(sub), 64):
            batch = torch.stack([tf(Image.open(ROOT / "data/mimo" / f).convert("RGB")) for f in sub.file.iloc[i:i + 64]]).cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                e = model(batch)
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    rows = []
    for tag, tf in (("default: resize + centre crop", default_tf), ("pad to square, then resize", pad_tf)):
        X = embed(tf); np.save(OUT / f"emb_{tag.split(':')[0].split(',')[0].replace(' ', '_')}.npy", X)
        Z = PCA(50, random_state=0).fit_transform(X)
        for col in ("hs3", "family", "name"):
            y = np.array(sub[col].tolist(), dtype=object); k = len(set(y))
            a = score(y, KMeans(k, n_init=10, random_state=0).fit_predict(Z))["AMI"]
            pr = cross_val_score(LogisticRegression(max_iter=2000, C=0.5), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
            rows.append({"preprocessing": tag, "label": col, "classes": k, "AMI": round(a, 3), "probe": round(float(pr), 3)})
            print(rows[-1], flush=True)
        # the same, restricted to the elongated thumbnails (aspect > 1.5), where the crop removes the most
        sel = ar > 1.5
        for col in ("hs3", "name"):
            y = np.array(sub[col][sel].tolist(), dtype=object); vc = pd.Series(y).value_counts(); keep = pd.Series(y).map(vc).values >= 20
            yk = y[keep]; k = len(set(yk))
            a = score(yk, KMeans(k, n_init=10, random_state=0).fit_predict(Z[sel][keep]))["AMI"]
            rows.append({"preprocessing": tag, "label": f"{col}, elongated only (n={int(keep.sum())})", "classes": k, "AMI": round(a, 3), "probe": None})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)


if __name__ == "__main__":
    main()
