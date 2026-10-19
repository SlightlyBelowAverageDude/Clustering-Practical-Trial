"""FSD50K family check: do blind sound clusters recover the Western instrument families the way blind photo clusters do?

Clips with exactly one family among their labels are kept (families: string, woodwind, brass, percussion, keyboard),
capped per family, embedded with CLAP (3 s loudest window, averaged over up to 3 windows), clustered with k-means and
scored against the family and against the instrument leaf label. Outputs: runs/fsd50k_family.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from librosa import resample
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import adjusted_mutual_info_score as ami

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DL, OUT = ROOT / "dl/fsd50k", ROOT / "runs/fsd50k_family"
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
SR, CHUNK, NCHUNK = 48000, 10, 3
FAMILY = {
    "string": ["Guitar", "Acoustic_guitar", "Electric_guitar", "Bass_guitar", "Bowed_string_instrument", "Plucked_string_instrument", "Harp", "Strum"],
    "woodwind": ["Wind_instrument_and_woodwind_instrument", "Harmonica"],
    "brass": ["Brass_instrument", "Trumpet"],
    "percussion": ["Percussion", "Drum", "Drum_kit", "Snare_drum", "Bass_drum", "Hi-hat", "Cymbal", "Crash_cymbal", "Gong", "Chime", "Cowbell",
                   "Tambourine", "Rattle_(instrument)", "Mallet_percussion", "Marimba_and_xylophone", "Glockenspiel", "Tabla"],
    "keyboard": ["Keyboard_(musical)", "Piano", "Organ", "Accordion"],
}
FAM_OF = {c: f for f, cs in FAMILY.items() for c in cs}
GENERIC = {"Guitar", "Bowed_string_instrument", "Plucked_string_instrument", "Wind_instrument_and_woodwind_instrument", "Brass_instrument",
           "Percussion", "Drum", "Mallet_percussion", "Keyboard_(musical)", "Musical_instrument"}


def load_labels():
    rows = []
    for split, fn in (("dev", "dev.csv"), ("eval", "eval.csv")):
        d = pd.read_csv(DL / "FSD50K.ground_truth" / fn)
        d["split"] = split
        rows.append(d[["fname", "labels", "split"]])
    d = pd.concat(rows, ignore_index=True)
    labs = d.labels.str.split(",")
    d["families"] = labs.apply(lambda L: sorted({FAM_OF[x] for x in L if x in FAM_OF}))
    d["leaf"] = labs.apply(lambda L: next((x for x in L if x in FAM_OF and x not in GENERIC), ""))
    d = d[d.families.apply(len) == 1].copy()
    d["family"] = d.families.str[0]
    d["path"] = [str(DL / ("FSD50K.dev_audio" if s == "dev" else "FSD50K.eval_audio") / f"{f}.wav") for f, s in zip(d.fname, d.split)]
    return d


def clap_embed(paths):
    from transformers import ClapModel, ClapProcessor
    m = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval().cuda()
    p = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    out = []
    for i, f in enumerate(paths):
        try:
            y, sr = sf.read(f, dtype="float32", always_2d=True); y = y.mean(1)
            if sr != SR:
                y = resample(y, orig_sr=sr, target_sr=SR)
        except Exception as e:  # noqa: BLE001
            print("  unreadable", f, type(e).__name__, flush=True); out.append(np.full(512, np.nan, dtype=np.float32)); continue
        L = SR * CHUNK
        chunks = [np.pad(y, (0, L - len(y)))] if len(y) <= L else [y[s:s + L] for s in np.linspace(0, len(y) - L, NCHUNK).astype(int)]
        inp = p(audio=chunks, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            e = m.get_audio_features(**{k: v.cuda() for k, v in inp.items()})
            e = e if isinstance(e, torch.Tensor) else e.pooler_output
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        if i % 500 == 0:
            print(f"  {i} clips embedded", flush=True)
    return np.stack(out)


def evaluate(X, y, tag, ks):
    y = np.array(y, dtype=object); k = len(set(y))
    Z = PCA(50, random_state=0).fit_transform(X)
    res = {kk: round(score(y, KMeans(kk, n_init=10, random_state=0).fit_predict(Z))["AMI"], 3) for kk in sorted(set(ks) | {k})}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), X, y, cv=cv)
    r = {"label": tag, "n": len(y), "classes": k, "AMI_at_classes": res[k], "best_AMI": max(res.values()), "best_k": max(res, key=res.get),
         "probe_acc": round(float((pred == y).mean()), 3), "probe_AMI": round(float(ami(y, pred)), 3),
         "majority": round(float(pd.Series(y).value_counts(normalize=True).iloc[0]), 3)}
    print(r, flush=True)
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_labels()
    print("clips with exactly one family:", len(d), d.family.value_counts().to_dict(), flush=True)
    d = d[[Path(p).exists() for p in d.path]]
    print("of which on disk:", len(d), flush=True)
    rng = np.random.default_rng(0)
    d = pd.concat([g.sample(min(CAP, len(g)), random_state=0) for _, g in d.groupby("family")]).reset_index(drop=True)
    print("sample:", d.family.value_counts().to_dict(), "| leaf-labelled:", (d.leaf != "").sum(), flush=True)
    d.to_csv(OUT / "sample.csv", index=False)
    cache = OUT / "emb_clap.npy"
    X = np.load(cache) if cache.exists() else clap_embed(d.path.tolist())
    np.save(cache, X)
    good = ~np.isnan(X).any(1); d, X = d[good].reset_index(drop=True), X[good]
    rows = [evaluate(X, d.family, "family (5)", ks=(5, 10, 20))]
    leaf = d[d.leaf != ""]
    vc = leaf.leaf.value_counts(); leaf = leaf[leaf.leaf.map(vc) >= 30]
    rows.append(evaluate(X[leaf.index.values], leaf.leaf, "instrument leaf (30+)", ks=(10, 20, 40)))
    for fam, g in d.groupby("family"):
        gl = g[g.leaf != ""]; vc = gl.leaf.value_counts(); gl = gl[gl.leaf.map(vc) >= 30]
        if gl.leaf.nunique() >= 2:
            rows.append(evaluate(X[gl.index.values], gl.leaf, f"leaf within {fam}", ks=(gl.leaf.nunique(),)))
    pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)
    # confusion of 5 blind sound groups vs family
    Z = PCA(50, random_state=0).fit_transform(X)
    km = KMeans(5, n_init=10, random_state=0).fit_predict(Z)
    ct = pd.crosstab(pd.Series(km, name="sound group"), d.family)
    print(ct.to_string(), flush=True); ct.to_csv(OUT / "confusion_k5.csv")


if __name__ == "__main__":
    main()
