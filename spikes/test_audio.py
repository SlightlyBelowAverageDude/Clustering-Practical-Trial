"""Audio add-on tests on the 868 objects with their own recording (runs/clean/audio_dedup).

3 sound representations:
  clap   CLAP audio embeddings (cached in runs/clean/audio_dedup; trained with text that names instruments)
  mfcc   timbre statistics from the spectrum (MFCC 1 to 19, centroid, bandwidth, roll-off, flatness, contrast, zero
         crossings; mean and std over frames, averaged over the 3 windows, z-scored over objects); no training
  hubert HuBERT base (facebook/hubert-base-ls960), self-supervised on unlabelled speech; mean over time and over the
         12 transformer layers
Same 3 windows of 10 seconds as the CLAP run. Photo side: padded DINOv2-B embeddings of the same objects.
Tests per representation: what the sound groups follow; photo groups vs sound groups (AMI); pair AUC (does a shared
photo group predict sound similarity); one-to-one matched photo and sound groups at k = 10 against a permutation test;
the pair test inside Stockholm and Berlin. Outputs: runs/tests/audio.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from librosa import resample
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score as ami, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "data/mimo_audio"
RUN = ROOT / "runs/clean/audio_dedup"
OUT = ROOT / "runs/tests/audio"
CHUNK, NCHUNK = 10, 3


def windows(f, sr):
    y, s0 = sf.read(f, dtype="float32", always_2d=True)
    y = y.mean(1)
    if s0 != sr:
        y = resample(y, orig_sr=s0, target_sr=sr)
    n, L = len(y), sr * CHUNK
    if n <= L:
        return [np.pad(y, (0, L - n))]
    return [y[s:s + L] for s in np.linspace(0, n - L, NCHUNK).astype(int)]


def mfcc_features(files):
    import librosa
    cache = OUT / "emb_mfcc.npy"
    if cache.exists():
        return np.load(cache)
    sr, out = 22050, []
    for i, f in enumerate(files):
        feats = []
        for y in windows(f, sr):
            parts = [librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[1:], librosa.feature.spectral_centroid(y=y, sr=sr),
                     librosa.feature.spectral_bandwidth(y=y, sr=sr), librosa.feature.spectral_rolloff(y=y, sr=sr),
                     librosa.feature.spectral_flatness(y=y), librosa.feature.spectral_contrast(y=y, sr=sr), librosa.feature.zero_crossing_rate(y)]
            M = np.vstack(parts)
            feats.append(np.concatenate([M.mean(1), M.std(1)]))
        out.append(np.mean(feats, 0))
        if i % 200 == 0:
            print(f"  mfcc {i}", flush=True)
    X = np.stack(out).astype(np.float32)
    np.save(cache, X)
    return X


def hubert_features(files):
    from transformers import AutoFeatureExtractor, HubertModel
    cache = OUT / "emb_hubert.npy"
    if cache.exists():
        return np.load(cache)
    name = "facebook/hubert-base-ls960"
    fe = AutoFeatureExtractor.from_pretrained(name)
    m = HubertModel.from_pretrained(name).eval().cuda()
    out = []
    for i, f in enumerate(files):
        embs = []
        for y in windows(f, 16000):
            inp = fe(y, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                hs = m(inp.input_values.cuda(), output_hidden_states=True).hidden_states[1:]
            embs.append(torch.stack([h.mean(1)[0] for h in hs]).mean(0).cpu().numpy())
        out.append(np.mean(embs, 0))
        if i % 200 == 0:
            print(f"  hubert {i}", flush=True)
    X = np.stack(out).astype(np.float32)
    np.save(cache, X)
    return X


def prep(X, kind):
    """Representation used for clustering and cosine similarity."""
    if kind == "mfcc":
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    elif kind == "hubert":
        X = X - X.mean(0)
    return normalize(X)


def pair_auc(S, groups, rng, n_pairs=200000):
    ok = np.where(groups != "")[0]
    i = rng.choice(ok, n_pairs)
    j = rng.choice(ok, n_pairs)
    m = i != j
    i, j = i[m], j[m]
    same = (groups[i] == groups[j]).astype(int)
    if same.sum() < 50 or same.sum() > len(same) - 50:
        return float("nan")
    return round(float(roc_auc_score(same, S[i, j])), 3)


def km(Z, k, seed=0):
    return KMeans(k, n_init=10, random_state=seed).fit_predict(Z)


def matched_share(a, b):
    C = pd.crosstab(a, b).values
    ri, cj = linear_sum_assignment(-C)
    return C[ri, cj].sum() / len(a)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RUN / "objects.csv", dtype=str).fillna("")
    V = {c: np.array(df[c].tolist(), dtype=object) for c in df.columns}
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    row = dict(zip(rec.id, range(len(rec))))
    XI = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")[[row[i] for i in df.id]]
    ZI = PCA(50, random_state=0).fit_transform(XI)
    files = [str(AUD / f) for f in df.file_a]
    spaces = {"clap": np.load(RUN / "emb_clap.npy"), "mfcc": mfcc_features(files), "hubert": hubert_features(files)}
    print({k: v.shape for k, v in spaces.items()}, flush=True)
    m3 = V["hs3"] != ""
    lab, cross, pairs, match, within = [], [], [], [], []
    for kind, X in spaces.items():
        A = prep(X, kind)
        ZA = PCA(50, random_state=0).fit_transform(A)
        S = A @ A.T
        rng = np.random.default_rng(0)
        # 1. what the sound groups follow
        for col in ("hs1", "hs2", "hs3", "name", "provider"):
            y = V[col]
            vc = pd.Series(y).value_counts()
            keep = (pd.Series(y).map(vc) >= 20).values & (y != "")
            k = len(set(y[keep]))
            probe = cross_val_score(LogisticRegression(max_iter=3000, C=0.5), A[keep], y[keep], cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
            r = {"space": kind, "label": col, "n": int(keep.sum()), "classes": k, "AMI": round(float(ami(y[keep], km(ZA[keep], k))), 3),
                 "probe_acc": round(float(probe), 3)}
            lab.append(r)
            print(r, flush=True)
        # 2. photo groups vs sound groups
        for k in (5, 10, 20, 40):
            ci, ca = km(ZI, k), km(ZA, k)
            r = {"space": kind, "k": k, "photo_vs_sound": round(float(ami(ci, ca)), 3), "sound_vs_hs3": round(float(ami(V["hs3"][m3], ca[m3])), 3),
                 "sound_vs_museum": round(float(ami(V["provider"], ca)), 3)}
            cross.append(r)
            print(r, flush=True)
        # 3. pair AUC
        r = {"space": kind}
        aucs = [pair_auc(S, km(ZI, 40, s).astype(str), rng) for s in range(5)]
        r["photo_group_k40_mean"], r["photo_group_k40_min"], r["photo_group_k40_max"] = round(float(np.mean(aucs)), 3), min(aucs), max(aucs)
        ci40 = km(ZI, 40).astype(str)
        r["photo_group_k40_coded_only"] = pair_auc(S, np.where(m3, ci40, ""), rng)
        for col in ("hs1", "hs3", "name", "provider"):
            r[col] = pair_auc(S, V[col].astype(str), rng)
        r["hs3_permuted"] = pair_auc(S, rng.permutation(V["hs3"].astype(str)), rng)
        pairs.append(r)
        print(r, flush=True)
        # 4. one-to-one matched groups at k = 10
        ci, ca = km(ZI, 10), km(ZA, 10)
        obs = matched_share(ci, ca)
        perm = np.array([matched_share(ci, rng.permutation(ca)) for _ in range(1000)])
        match.append({"space": kind, "k": 10, "objects_on_matched_pairs": round(float(obs), 3), "permutation_mean": round(float(perm.mean()), 3),
                      "permutation_max": round(float(perm.max()), 3), "p_value": round(float(((perm >= obs).sum() + 1) / 1001), 4)})
        print(match[-1], flush=True)
        # 5. inside one museum
        for prov in ("Swedish Museum of Performing Arts", "Ethnological Museum, Staatliche Museen zu Berlin"):
            ii = np.where(V["provider"] == prov)[0]
            Sg = S[np.ix_(ii, ii)]
            ci = km(ZI[ii], 10).astype(str)
            within.append({"space": kind, "museum": prov, "n": len(ii), "photo_group_k10": pair_auc(Sg, ci, rng, 50000),
                           "hs3": pair_auc(Sg, V["hs3"][ii].astype(str), rng, 50000)})
            print(within[-1], flush=True)
    pd.DataFrame(lab).to_csv(OUT / "labels.csv", index=False)
    pd.DataFrame(cross).to_csv(OUT / "photo_vs_sound.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUT / "pair_auc.csv", index=False)
    pd.DataFrame(match).to_csv(OUT / "matched_groups.csv", index=False)
    pd.DataFrame(within).to_csv(OUT / "within_museum.csv", index=False)



def granularity():
    """Pair AUC at matched granularity: photo groups with as many groups as the code level has classes on these objects."""
    df = pd.read_csv(RUN / "objects.csv", dtype=str).fillna("")
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    row = dict(zip(rec.id, range(len(rec))))
    r_idx = [row[i] for i in df.id]
    V = {c: np.array(rec[c].tolist(), dtype=object)[r_idx] for c in ("hs3", "hs4", "hs5", "hs6", "name", "provider")}
    XI = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")[r_idx]
    ZI = PCA(50, random_state=0).fit_transform(XI)
    spaces = {"clap": np.load(RUN / "emb_clap.npy"), "mfcc": np.load(OUT / "emb_mfcc.npy"), "hubert": np.load(OUT / "emb_hubert.npy")}
    rows = []
    for kind, X in spaces.items():
        A = prep(X, kind)
        S = A @ A.T
        rng = np.random.default_rng(0)
        for lv in ("hs3", "hs4", "hs5", "hs6", "name"):
            y = V[lv]
            has = y != ""
            k = len(set(y[has]))
            photo = np.where(has, km(ZI, k).astype(str), "")
            rows.append({"space": kind, "label": lv, "objects_with_label": int(has.sum()), "classes": k,
                         "AUC_label": pair_auc(S, y.astype(str), rng), "AUC_photo_groups_same_k_same_objects": pair_auc(S, photo, rng)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "pair_auc_granularity.csv", index=False)


if __name__ == "__main__":
    import sys
    granularity() if sys.argv[1:] == ["granularity"] else main()
