"""MIMO audio: do instruments that cluster together visually also sound alike?

Objects with a museum recording (data/mimo_audio) are embedded with CLAP (audio) and matched to their DINOv2-B image
embeddings (runs/mimo_full). Tests:
  1. audio clusters vs H-S levels, instrument name and museum (same scoring as the image runs);
  2. agreement between image clusters and audio clusters of the same objects (AMI), against H-S as the reference;
  3. does sharing an image cluster predict audio similarity? AUC of CLAP cosine similarity for same-image-cluster
     pairs vs different-cluster pairs, compared with same-H-S-code pairs and same-museum pairs;
  4. zero-shot CLAP text labels for the audio clusters.
Outputs: runs/mimo_audio.
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
from sklearn.metrics import adjusted_mutual_info_score as ami, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hs_level, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "data/mimo_audio"
OUT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs/mimo_audio")
SR, CHUNK, NCHUNK = 48000, 10, 3
HS1 = {"1": "Idiophone", "2": "Membranophone", "3": "Chordophone", "4": "Aerophone", "5": "Electrophone"}


def clap_embed(files):
    from transformers import ClapModel, ClapProcessor
    cache = OUT / "emb_clap.npy"
    if cache.exists():
        return np.load(cache)
    m = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval().cuda()
    p = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    out = []
    for i, f in enumerate(files):
        try:
            y, sr = sf.read(f, dtype="float32", always_2d=True)
            y = y.mean(1)
            if sr != SR:
                y = resample(y, orig_sr=sr, target_sr=SR)
        except Exception as e:  # noqa: BLE001
            print("  unreadable", f, type(e).__name__, flush=True); out.append(np.full(512, np.nan, dtype=np.float32)); continue
        n = len(y); L = SR * CHUNK
        if n <= L:
            chunks = [np.pad(y, (0, L - n))]
        else:
            starts = np.linspace(0, n - L, NCHUNK).astype(int)
            chunks = [y[s:s + L] for s in starts]
        inp = p(audio=chunks, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            e = m.get_audio_features(**{k: v.cuda() for k, v in inp.items()})
            e = e if isinstance(e, torch.Tensor) else e.pooler_output
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        if i % 100 == 0:
            print(f"  {i} audio files embedded", flush=True)
    X = np.stack(out); np.save(cache, X)
    return X


def clap_text(prompts):
    from transformers import ClapModel, ClapProcessor
    m = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval().cuda()
    p = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    t = p(text=prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        e = m.get_text_features(**{k: v.cuda() for k, v in t.items()})
        e = e if isinstance(e, torch.Tensor) else e.pooler_output
    return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def labels_eval(X, df, cols, ks, min_count=20, tag=""):
    rows = []
    Z = PCA(min(50, X.shape[1], len(X) - 1), random_state=0).fit_transform(X)
    for col in cols:
        y = np.array(df[col].astype(str).tolist(), dtype=object)
        vc = pd.Series(y).value_counts(); keep = (pd.Series(y).map(vc) >= min_count).values & (y != "") & (y != "nan")
        yc = y[keep]; k = len(set(yc))
        if k < 2:
            continue
        res = {kk: score(yc, KMeans(kk, n_init=10, random_state=0).fit_predict(Z[keep]))["AMI"] for kk in sorted(set(ks) | {k})}
        pr = cross_val_score(LogisticRegression(max_iter=3000, C=0.5), X[keep], yc, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        r = {"space": tag, "label": col, "n": int(keep.sum()), "classes": k, "AMI_at_classes": round(res[k], 3), "best_AMI": round(max(res.values()), 3),
             "probe": round(float(pr), 3), "majority": round(float(pd.Series(yc).value_counts(normalize=True).iloc[0]), 3)}
        rows.append(r); print(r, flush=True)
    return rows


def pair_auc(S, groups, rng, n_pairs=200000):
    """AUC of similarity for same-group vs different-group pairs (groups == "" excluded)."""
    ok = np.where(groups != "")[0]
    i = rng.choice(ok, n_pairs); j = rng.choice(ok, n_pairs); m = i != j
    i, j = i[m], j[m]
    same = (groups[i] == groups[j]).astype(int)
    if same.sum() < 50 or same.sum() > len(same) - 50:
        return float("nan"), int(same.sum())
    return round(float(roc_auc_score(same, S[i, j])), 3), int(same.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(AUD / "audio.csv", dtype=str).fillna("")
    a = a[a.ok == "True"].drop_duplicates("id").reset_index(drop=True)
    # a recording attached to several objects (an ensemble piece in Paris, one sample for many accordions in Soinuenea) is
    # not a recording of that object: drop every object whose file is byte-identical to another object's file
    import hashlib
    a["audio_hash"] = [hashlib.md5((AUD / f).read_bytes()).hexdigest() if (AUD / f).exists() else "" for f in a.file]
    shared = a.audio_hash.duplicated(keep=False) & (a.audio_hash != "")
    print(f"objects whose recording is shared with another object: {int(shared.sum())} (dropped)")
    a = a[~shared].reset_index(drop=True)
    # image side: default is the first full run; pass a table with name and family columns plus its embedding file to use another
    IMG_TABLE = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "runs/mimo_full/sample_full.csv")
    IMG_EMB = ROOT / (sys.argv[3] if len(sys.argv) > 3 else "runs/mimo_full/emb/dinov2_b.npy")
    full = pd.read_csv(IMG_TABLE, dtype={"notation": str}).fillna("")
    if "name" not in full.columns:
        names = pd.read_csv(ROOT / "runs/mimo_names/labels.csv", dtype=str).fillna("")
        full["name"], full["family"] = names.name.values, names.family.values
    full["row"] = np.arange(len(full))
    df = a.merge(full, on="id", how="inner", suffixes=("_a", ""))
    print(f"{len(a)} objects with audio, {len(df)} also in the image set")
    df["hs1"] = pd.Series(hs_level(df.notation, 1)).map(HS1).fillna("").values
    df["hs2"], df["hs3"] = hs_level(df.notation, 2), hs_level(df.notation, 3)
    print("museums:", df.provider.value_counts().head(8).to_dict())
    print("H-S classes:", df.hs1.value_counts().to_dict(), "| names with 20+:", (df.name.value_counts() >= 20).sum(), flush=True)

    XA = clap_embed([str(AUD / f) for f in df.file_a])
    good = ~np.isnan(XA).any(1)
    df = df[good].reset_index(drop=True); XA = XA[good]
    XI = np.load(IMG_EMB)[df.row.values]
    print(f"{len(df)} objects with both embeddings", flush=True)

    print("\n== 1. what the audio clusters follow (CLAP), and the same for images on the same objects")
    rows = labels_eval(XA, df, ["hs1", "hs2", "hs3", "name", "provider", "region"], ks=(5, 10, 20, 40), tag="audio")
    rows += labels_eval(XI, df, ["hs1", "hs2", "hs3", "name", "provider", "region"], ks=(5, 10, 20, 40), tag="image")
    pd.DataFrame(rows).to_csv(OUT / "labels.csv", index=False)

    print("\n== 2. image clusters vs audio clusters of the same objects (AMI), with H-S as reference")
    ZA = PCA(50, random_state=0).fit_transform(XA); ZI = PCA(50, random_state=0).fit_transform(XI)
    rows = []
    for k in (5, 10, 20, 40):
        ca = KMeans(k, n_init=10, random_state=0).fit_predict(ZA); ci = KMeans(k, n_init=10, random_state=0).fit_predict(ZI)
        m3 = (df.hs3 != "").values
        r = {"k": k, "image_vs_audio": round(ami(ci, ca), 3), "image_vs_hs3": round(ami(ci[m3], df.hs3[m3]), 3), "audio_vs_hs3": round(ami(ca[m3], df.hs3[m3]), 3),
             "image_vs_museum": round(ami(ci, df.provider), 3), "audio_vs_museum": round(ami(ca, df.provider), 3)}
        rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "image_vs_audio.csv", index=False)

    print("\n== 3. does sharing an image cluster predict sounding alike? AUC of CLAP cosine for same vs different group")
    S = XA @ XA.T
    rng = np.random.default_rng(0)
    rows = []
    for k in (10, 20, 40):
        ci = KMeans(k, n_init=10, random_state=0).fit_predict(ZI).astype(str)
        auc, npos = pair_auc(S, ci, rng); rows.append({"grouping": f"image cluster k={k}", "AUC": auc, "same_pairs": npos})
    for col in ("hs1", "hs2", "hs3", "name", "provider"):
        auc, npos = pair_auc(S, df[col].values.astype(str), rng); rows.append({"grouping": col, "AUC": auc, "same_pairs": npos})
    perm = rng.permutation(df.hs3.values.astype(str)); auc, npos = pair_auc(S, perm, rng); rows.append({"grouping": "hs3 permuted (chance)", "AUC": auc, "same_pairs": npos})
    # within one museum, so recording setup is held fixed
    for prov, g in df.groupby("provider"):
        if len(g) < 150:
            continue
        ii = g.index.values; Sg = S[np.ix_(ii, ii)]
        ci = KMeans(10, n_init=10, random_state=0).fit_predict(ZI[ii]).astype(str)
        auc_i, _ = pair_auc(Sg, ci, rng, 50000); auc_h, _ = pair_auc(Sg, g.hs3.values.astype(str), rng, 50000)
        rows.append({"grouping": f"within {prov[:30]}: image cluster k=10", "AUC": auc_i, "same_pairs": len(g)})
        rows.append({"grouping": f"within {prov[:30]}: hs3", "AUC": auc_h, "same_pairs": len(g)})
    t = pd.DataFrame(rows); print(t.to_string(index=False), flush=True); t.to_csv(OUT / "pair_auc.csv", index=False)

    print("\n== 4. CLAP zero-shot names for 20 audio clusters (prompt 'the sound of a <name>'), majority image name beside it")
    vocab = [n for n, c in df.name.value_counts().items() if n and c >= 5]
    T = clap_text([f"the sound of a {n}" for n in vocab])
    ca = KMeans(20, n_init=10, random_state=0).fit_predict(ZA)
    rows = []
    for c in range(20):
        ii = np.where(ca == c)[0]
        cen = XA[ii].mean(0); cen /= np.linalg.norm(cen)
        top = np.argsort(-(T @ cen))[:3]
        maj = df.name.iloc[ii].value_counts(); mh = df.hs1.iloc[ii].value_counts()
        r = {"cluster": c, "n": len(ii), "clap_names": ", ".join(vocab[t] for t in top), "majority_name": f"{maj.index[0]} ({maj.iloc[0] / len(ii):.2f})" if len(maj) else "",
             "majority_hs1": f"{mh.index[0]} ({mh.iloc[0] / len(ii):.2f})" if len(mh) else ""}
        rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "audio_cluster_names.csv", index=False)
    df[["id", "name", "hs1", "hs2", "hs3", "provider", "region", "file_a"]].to_csv(OUT / "objects.csv", index=False)


if __name__ == "__main__":
    main()
