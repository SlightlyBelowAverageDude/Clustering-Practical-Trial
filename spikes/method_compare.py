"""Point 1 check: can anything beat k-means on frozen DINOv2-B features, on the same rows and labels?

12,000 MIMO objects with H-S level 3, Western family and instrument name. Methods: k-means (PCA 50, 3 seeds), spherical
k-means, GMM, Ward, spectral (kNN graph), UMAP + k-means, UMAP + HDBSCAN, per-museum centering + k-means, stronger and
ensembled encoders + k-means. Score: AMI at k = number of classes for each label. Outputs: runs/method_compare.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import cached_embeddings, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/method_compare"
N = 12000
common.ENCODERS["siglip_b16"] = ("open_clip", "ViT-B-16-SigLIP", "webli")
common.ENCODERS["dinov2_l"] = ("timm", "vit_large_patch14_dinov2.lvd142m", None)
LABELS = ["hs3", "family", "name"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "runs/mimo_full/sample_full.csv", dtype={"notation": str}).fillna("")
    lab = pd.read_csv(ROOT / "runs/mimo_names/labels.csv", dtype=str).fillna("")
    df["name"], df["family"] = lab.name.values, lab.family.values
    nt = df.notation.str.replace(".", "", regex=False)
    df["hs3"] = np.where(nt.str.len() >= 3, nt.str[:3], "")
    m = (df.name != "") & (df.family != "") & (df.hs3 != "")
    sub = df[m]
    for col in LABELS:
        vc = sub[col].value_counts(); sub = sub[sub[col].map(vc) >= 30]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(sub.index.values, N, replace=False))
    sub = df.loc[idx]
    sub.to_csv(OUT / "subset.csv", index=False)
    Y = {col: np.array(sub[col].tolist(), dtype=object) for col in LABELS}
    K = {col: len(set(Y[col])) for col in LABELS}
    print("rows", len(sub), "classes", K, flush=True)
    X = np.load(ROOT / "runs/mimo_full/emb/dinov2_b.npy")[idx]
    rows = []

    def report(method, pred, extra=""):
        r = {"method": method, **{f"AMI_{c}": round(score(Y[c], p)["AMI"], 3) for c, p in pred.items()}, "note": extra}
        rows.append(r); print(r, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)

    def kmeans_all(Z, seed=0):
        return {c: KMeans(K[c], n_init=10, random_state=seed).fit_predict(Z) for c in LABELS}

    Z = PCA(50, random_state=0).fit_transform(X)
    t0 = time.time()
    for seed in (0, 1, 2):
        report(f"k-means, PCA 50, seed {seed}", kmeans_all(Z, seed), f"{time.time() - t0:.0f}s")
    report("k-means, PCA 50, 40 restarts", {c: KMeans(K[c], n_init=40, random_state=0).fit_predict(Z) for c in LABELS})
    report("spherical k-means (L2-normalised, 768-d)", kmeans_all(normalize(X)))
    report("k-means, PCA 128", kmeans_all(PCA(128, random_state=0).fit_transform(X)))
    report("GMM diagonal, PCA 50", {c: GaussianMixture(K[c], covariance_type="diag", random_state=0, n_init=3).fit_predict(Z) for c in LABELS})
    report("Ward agglomerative, PCA 50", {c: AgglomerativeClustering(K[c], linkage="ward").fit_predict(Z) for c in LABELS})
    report("average-link cosine, PCA 50", {c: AgglomerativeClustering(K[c], linkage="average", metric="cosine").fit_predict(Z) for c in LABELS})
    try:
        report("spectral, kNN graph (15), PCA 50", {c: SpectralClustering(K[c], affinity="nearest_neighbors", n_neighbors=15, random_state=0, assign_labels="cluster_qr").fit_predict(Z) for c in LABELS})
    except Exception as e:  # noqa: BLE001
        print("spectral failed", type(e).__name__, e, flush=True)
    import umap
    for nd in (10, 30):
        U = umap.UMAP(n_components=nd, n_neighbors=30, min_dist=0.0, random_state=0).fit_transform(Z)
        report(f"UMAP {nd}-d + k-means", kmeans_all(U))
        report(f"UMAP {nd}-d + Ward", {c: AgglomerativeClustering(K[c], linkage="ward").fit_predict(U) for c in LABELS})
    import hdbscan
    U = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.0, random_state=0).fit_transform(Z)
    for mcs in (30, 80):
        h = hdbscan.HDBSCAN(min_cluster_size=mcs).fit_predict(U)
        ncl = len(set(h)) - (1 if -1 in h else 0)
        report(f"UMAP 10-d + HDBSCAN (min size {mcs})", {c: h for c in LABELS}, f"{ncl} clusters, noise {np.mean(h == -1):.2f}")

    # photography factor: centre each museum's embeddings on its own mean before PCA
    prov = sub.provider.values
    Xc = X.copy()
    for p in np.unique(prov):
        sel = prov == p
        if sel.sum() >= 20:
            Xc[sel] -= Xc[sel].mean(0)
    report("per-museum centering + k-means, PCA 50", kmeans_all(PCA(50, random_state=0).fit_transform(Xc)))
    report("per-museum centering + spherical k-means", kmeans_all(normalize(Xc)))

    # encoders
    paths = [str(ROOT / "data/mimo" / f) for f in sub.file]
    embs = {"dinov2_b": X}
    for enc in ("dinov2_l", "clip_l14", "siglip_b16"):
        t0 = time.time()
        E = cached_embeddings(OUT / "emb", enc, paths)
        embs[enc] = E
        report(f"{enc} + k-means, PCA 50", kmeans_all(PCA(50, random_state=0).fit_transform(E)), f"embedded in {time.time() - t0:.0f}s")
    for combo in (("dinov2_b", "dinov2_l"), ("dinov2_l", "siglip_b16"), ("dinov2_b", "dinov2_l", "clip_l14", "siglip_b16")):
        E = np.hstack([normalize(embs[e]) for e in combo])
        report(f"ensemble {'+'.join(combo)} + k-means, PCA 50", kmeans_all(PCA(50, random_state=0).fit_transform(E)))
    best = max(("dinov2_b", "dinov2_l", "siglip_b16"), key=lambda e: [r for r in rows if r["method"].startswith(e + " +") or (e == "dinov2_b" and r["method"].startswith("k-means, PCA 50, seed 0"))][0]["AMI_name"])
    E = embs[best].copy()
    for p in np.unique(prov):
        sel = prov == p
        if sel.sum() >= 20:
            E[sel] -= E[sel].mean(0)
    U = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.0, random_state=0).fit_transform(PCA(50, random_state=0).fit_transform(E))
    report(f"{best} + per-museum centering + UMAP 10-d + k-means", kmeans_all(U))
    report(f"{best} + per-museum centering + UMAP 10-d + Ward", {c: AgglomerativeClustering(K[c], linkage="ward").fit_predict(U) for c in LABELS})


if __name__ == "__main__":
    main()
