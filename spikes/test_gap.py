"""Can blind clustering get closer to the probe? Tests on the 12,000-row methods subset (runs/clean/methods_subset_ids.csv).

Labels hs3, family and name (classes with 5+ objects in the subset), k = number of classes, scored with AMI.
Blocks (pass any as arguments; default: base scan temi text):
  base     padded DINOv2-B: k-means on PCA 50, average-link cosine, probe (same setup as runs/clean/methods.csv)
  objonly  the same on object-only images (data/mimo_objonly: rembg u2net mask, object on white)
  scan     SCAN on frozen DINOv2-B features: a head trained so that neighbours share a cluster, 10 seeds, the head with the
           lowest loss is kept and refined by self-labelling
  temi     TEMI on frozen DINOv2-B features with the official settings, trained on all 59,661 clean photos without
           labels, the head with the lowest loss is kept; k-means fit the same way as the comparison
  input    DINOv3-B/16 and DINOv3-L/16 as the encoder; DINOv2-B with the photo and its mirror image averaged
  text     language-guided: SigLIP B/16 image similarity to "a photo of a <name>, a musical instrument" for the 945
           instrument names of the Wikidata items linked to MIMO (the list contains the 129 name classes)
No block uses the labels except for scoring. Model selection uses the unsupervised loss only.
Outputs: runs/tests/gap/<block>.csv
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import cached_embeddings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/tests/gap"
LABELS = ("hs3", "family", "name")
DEV = "cuda"
N_NEIGHBOURS = 20


def kept(y, min_count=5):
    y = np.array(pd.Series(y).astype(str).tolist(), dtype=object)
    vc = pd.Series(y).value_counts()
    return (pd.Series(y).map(vc).values >= min_count) & (y != "") & (y != "nan"), y


def load():
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    ids = pd.read_csv(ROOT / "runs/clean/methods_subset_ids.csv", dtype=str).id
    pos = dict(zip(rec.id, range(len(rec))))
    idx = np.array([pos[i] for i in ids])
    sub = rec.iloc[idx].reset_index(drop=True)
    X = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")[idx]
    labels = {c: kept(sub[c].values) for c in LABELS}
    return sub, X, labels, idx


def classic(X, labels, variant, with_probe=True):
    Z = PCA(50, random_state=0).fit_transform(X)
    rows = []
    for method in ("k-means", "average-link cosine"):
        r = {"variant": variant, "method": method}
        for c, (keep, y) in labels.items():
            yk = y[keep]
            k = len(set(yk))
            if method == "k-means":
                p = KMeans(k, n_init=10, random_state=0).fit_predict(Z[keep])
            else:
                p = AgglomerativeClustering(k, linkage="average", metric="cosine").fit_predict(Z[keep])
            r[c] = round(float(ami(yk, p)), 3)
        rows.append(r)
        print(r, flush=True)
    if with_probe:
        r = {"variant": variant, "method": "probe"}
        for c, (keep, y) in labels.items():
            yk = y[keep]
            p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), X[keep], yk, cv=StratifiedKFold(5, shuffle=True, random_state=0))
            r[c] = round(float(ami(yk, p)), 3)
        rows.append(r)
        print(r, flush=True)
    return rows


# ---------------------------------------------------------------- SCAN and TEMI on frozen features

def neighbours(Xt, n=N_NEIGHBOURS):
    S = Xt @ Xt.T
    S.fill_diagonal_(-2)
    return S.topk(n, dim=1).indices


def mlp(d, k, hidden=512):
    return torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, k))


def aug(x, p):
    return F.dropout(x, p, training=True)


def scan_loss(pa, pn, w_ent):
    cons = -torch.log((pa * pn).sum(1).clamp_min(1e-8)).mean()
    pm = pa.mean(0)
    return cons + w_ent * (pm * torch.log(pm.clamp_min(1e-8))).sum()


def scan_head(Xt, nb, k, seed, epochs=100, bs=256, w_ent=5.0):
    torch.manual_seed(seed)
    head = mlp(Xt.shape[1], k).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    N = len(Xt)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(N, device=DEV)
        for b in range(0, N - bs + 1, bs):
            i = perm[b:b + bs]
            j = nb[i, torch.randint(0, nb.shape[1], (len(i),), device=DEV)]
            loss = scan_loss(head(aug(Xt[i], 0.1)).softmax(1), head(aug(Xt[j], 0.1)).softmax(1), w_ent)
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        P = head(Xt).softmax(1)
        L = float(scan_loss(P.repeat_interleave(nb.shape[1], 0), P[nb.reshape(-1)], w_ent))
    return head, L


def self_label(head, Xt, epochs=50, bs=256, thr=0.99):
    opt = torch.optim.Adam(head.parameters(), lr=1e-4, weight_decay=1e-4)
    N = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(N, device=DEV)
        for b in range(0, N - bs + 1, bs):
            i = perm[b:b + bs]
            head.eval()
            with torch.no_grad():
                pw = head(aug(Xt[i], 0.1)).softmax(1)
            conf, target = pw.max(1)
            m = conf > thr
            if m.sum() < 2:
                continue
            head.train()
            counts = torch.bincount(target[m], minlength=pw.shape[1]).float()
            w = torch.where(counts > 0, 1.0 / counts.clamp_min(1), torch.zeros_like(counts))
            w = w / w[counts > 0].mean()
            loss = F.cross_entropy(head(aug(Xt[i][m], 0.3)), target[m], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


def run_scan(X, labels):
    Xt = torch.tensor(normalize(X), device=DEV, dtype=torch.float32)
    rows = []
    for c, (keep, y) in labels.items():
        yk = y[keep]
        k = len(set(yk))
        Xk = Xt[torch.tensor(np.where(keep)[0], device=DEV)]
        nb = neighbours(Xk)
        heads = []
        for seed in range(10):
            h, L = scan_head(Xk, nb, k, seed)
            with torch.no_grad():
                p = h(Xk).argmax(1).cpu().numpy()
            heads.append((L, seed, h, round(float(ami(yk, p)), 3), len(set(p))))
        heads.sort(key=lambda t: t[0])
        L, seed, h, a_sel, used = heads[0]
        h = self_label(h, Xk)
        with torch.no_grad():
            p = h(Xk).argmax(1).cpu().numpy()
        r = {"label": c, "k": k, "AMI_selected_head": a_sel, "clusters_used": used, "AMI_after_self_label": round(float(ami(yk, p)), 3),
             "clusters_used_after": len(set(p)), "AMI_all_heads_min": min(t[3] for t in heads), "AMI_all_heads_max": max(t[3] for t in heads),
             "selected_seed": seed}
        rows.append(r)
        print("SCAN", r, flush=True)
    return rows


class DinoHead(torch.nn.Module):
    """TEMI's default head: 2 layers (512 hidden, GELU) into a 256-d bottleneck, L2 normalised, weight-normalised last layer."""

    def __init__(self, d, k, hidden=512, bottleneck=256):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, bottleneck))
        self.last = torch.nn.utils.parametrizations.weight_norm(torch.nn.Linear(bottleneck, k, bias=False))

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))


class MultiHead(torch.nn.Module):
    def __init__(self, d, k, H):
        super().__init__()
        self.heads = torch.nn.ModuleList([DinoHead(d, k) for _ in range(H)])

    def forward(self, x):
        return torch.stack([h(x) for h in self.heads])  # H x B x k


def temi_loss(ls_i, lt_j, ls_j, lt_i, q, beta, per_head=False):
    """Log-probabilities in, all H x B x k: ls_* student, lt_* teacher; q: H x k teacher marginals.
    pmi^beta = log sum_c (q_s(c|x) q_t(c|x'))^beta / q(c), computed in log space: the same value as the official
    linear-space form, without the infinite gradient of the power 0.6 at probability 0."""
    w = (lt_i.exp() * lt_j.exp()).sum(2).mean(0)  # B, teacher agreement of the pair, averaged over heads
    lq = torch.log(q)[:, None, :]
    a = torch.logsumexp(beta * (ls_i + lt_j) - lq, dim=2)
    b = torch.logsumexp(beta * (ls_j + lt_i) - lq, dim=2)
    L = -0.5 * w[None, :] * (a + b)  # H x B
    return L.mean(1) if per_head else L.mean()


def neighbours_chunked(Xt, n, chunk=4096):
    out = []
    for s in range(0, len(Xt), chunk):
        S = Xt[s:s + chunk] @ Xt.T
        S[torch.arange(S.shape[0]), torch.arange(s, s + S.shape[0])] = -2
        out.append(S.topk(n, dim=1).indices)
    return torch.cat(out)


def run_temi(labels, sub_idx, H=16, epochs=100, warmup=20, bs=1024, lr=1e-4, wd=1e-4, beta=0.6, tau=0.1, mom=0.996, knn=50):
    """Settings of the official TEMI command for precomputed embeddings (HHU-MMBS/TEMI-official-BMVC2023 README:
    batch 1024, lr 1e-4 with 20 warm-up epochs, 100 epochs, knn 50, 16 heads, beta 0.6, teacher momentum 0.996) and its
    argument defaults (student and teacher temperature 0.1, marginal momentum 0.9, AdamW, weight decay 1e-4, no
    embedding augmentation, head as in DinoHead). Trained on all clean photos without labels; the 12,000 subset rows are
    then assigned. For comparison, k-means is fit the same way (all clean photos, PCA 50) and its assignment scored."""
    Xall = np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy")
    Xt = torch.tensor(normalize(Xall), device=DEV, dtype=torch.float32)
    nb = neighbours_chunked(Xt, knn)
    N = len(Xt)
    Zall = PCA(50, random_state=0).fit_transform(Xall)
    rows = []
    for c, (keep, y) in labels.items():
        yk = y[keep]
        k = len(set(yk))
        rows_k = sub_idx[keep]
        kmp = KMeans(k, n_init=10, random_state=0).fit(Zall).labels_[rows_k]
        torch.manual_seed(0)
        student = MultiHead(Xt.shape[1], k, H).to(DEV)
        teacher = copy.deepcopy(student)
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        q = torch.full((H, k), 1.0 / k, device=DEV)
        opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=wd)
        steps_per_epoch = N // bs
        for ep in range(epochs):
            perm = torch.randperm(N, device=DEV)
            for st in range(steps_per_epoch):
                it = ep * steps_per_epoch + st
                for g in opt.param_groups:
                    g["lr"] = lr * min(1.0, (it + 1) / (warmup * steps_per_epoch))
                i = perm[st * bs:(st + 1) * bs]
                j = nb[i, torch.randint(0, knn, (len(i),), device=DEV)]
                ls_i, ls_j = (student(Xt[i]) / tau).log_softmax(2), (student(Xt[j]) / tau).log_softmax(2)
                with torch.no_grad():
                    lt_i, lt_j = (teacher(Xt[i]) / tau).log_softmax(2), (teacher(Xt[j]) / tau).log_softmax(2)
                loss = temi_loss(ls_i, lt_j, ls_j, lt_i, q, beta)
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    for ps, pt in zip(student.parameters(), teacher.parameters()):
                        pt.mul_(mom).add_(ps.detach(), alpha=1 - mom)
                    q = 0.9 * q + 0.1 * torch.cat([lt_i, lt_j], 1).exp().mean(1)
            if not torch.isfinite(loss):
                raise RuntimeError(f"TEMI loss not finite at epoch {ep}")
        with torch.no_grad():
            per_head = torch.zeros(H, device=DEV)
            count = 0
            for s in range(0, N, 4096):
                i = torch.arange(s, min(s + 4096, N), device=DEV)
                for t_ in range(0, knn, 10):
                    j = nb[i, t_]
                    ls_i, ls_j = (student(Xt[i]) / tau).log_softmax(2), (student(Xt[j]) / tau).log_softmax(2)
                    lt_i, lt_j = (teacher(Xt[i]) / tau).log_softmax(2), (teacher(Xt[j]) / tau).log_softmax(2)
                    per_head += temi_loss(ls_i, lt_j, ls_j, lt_i, q, beta, per_head=True) * len(i)
                    count += len(i)
            per_head /= count
            preds = teacher(Xt[torch.tensor(rows_k, device=DEV)]).argmax(2).cpu().numpy()
        amis = [round(float(ami(yk, preds[h])), 3) for h in range(H)]
        best = int(per_head.argmin())
        r = {"label": c, "k": k, "TEMI_AMI_selected_head": amis[best], "clusters_used_on_subset": len(set(preds[best])),
             "TEMI_AMI_all_heads_min": min(amis), "TEMI_AMI_all_heads_max": max(amis), "selected_head": best,
             "kmeans_fit_on_all_photos_AMI": round(float(ami(yk, kmp)), 3)}
        rows.append(r)
        print("TEMI", r, flush=True)
    return rows


# ---------------------------------------------------------------- input side

def run_input(sub, X, labels):
    """Other encoders and mirror averaging on the same 12,000 photos: DINOv3-B/16 and DINOv3-L/16 (timm, lvd1689m),
    padded; DINOv2-B embeddings of the photo and of its mirror image averaged (both padded, the published embedding
    plus one pass over the mirrored photos)."""
    from PIL import Image, ImageOps
    common.ENCODERS["dinov3_b"] = ("timm", "vit_base_patch16_dinov3.lvd1689m", None)
    common.ENCODERS["dinov3_l"] = ("timm", "vit_large_patch16_dinov3.lvd1689m", None)
    common.ENCODERS["dinov3_b_512"] = ("timm", "vit_base_patch16_dinov3.lvd1689m", None, {"img_size": 512})
    paths = [str(ROOT / "data/mimo" / f) for f in sub.file]
    rows = []
    for enc, name in (("dinov3_b", "DINOv3-B/16, 256 px"), ("dinov3_l", "DINOv3-L/16, 256 px"), ("dinov3_b_512", "DINOv3-B/16, 512 px")):
        XE = cached_embeddings(OUT / "emb_input", enc, paths, pad_square=True)
        rows += classic(XE, labels, name)
    cache = OUT / "emb_input" / "dinov2_b_pad_mirror.npy"
    if cache.exists():
        XM = np.load(cache)
    else:
        e = common.Encoder("dinov2_b", pad_square=True)
        parts = []
        for i in range(0, len(paths), 500):
            ims = [ImageOps.mirror(Image.open(q).convert("RGB")) for q in paths[i:i + 500]]
            parts.append(e.embed_images(ims))
        XM = np.concatenate(parts)
        np.save(cache, XM)
    rows += classic(XM, labels, "DINOv2-B, mirror image only", with_probe=False)
    rows += classic(normalize(X + XM), labels, "DINOv2-B, photo and mirror image averaged")
    pd.DataFrame(rows).to_csv(OUT / "input.csv", index=False)
    return rows


# ---------------------------------------------------------------- language-guided

def run_text(sub, X, labels):
    common.ENCODERS["siglip_b16"] = ("open_clip", "ViT-B-16-SigLIP", "webli")
    paths = [str(ROOT / "data/mimo" / f) for f in sub.file]
    XS = cached_embeddings(OUT / "emb_text", "siglip_b16", paths, pad_square=True)
    wd = pd.read_csv(ROOT / "dl/wd_mimo_instruments.csv", dtype=str).fillna("")
    vocab = sorted({v.strip() for v in wd.itemLabel if v.strip() and not v.strip().startswith("Q")})
    enc = common.Encoder("siglip_b16")
    T = np.concatenate([enc.embed_texts([f"a photo of a {v}, a musical instrument" for v in vocab[i:i + 128]]) for i in range(0, len(vocab), 128)])
    pd.DataFrame({"name": vocab}).to_csv(OUT / "text_vocabulary.csv", index=False)
    print("vocabulary", len(vocab), "names", flush=True)
    S = XS @ T.T
    prof = (S - S.mean(0)) / (S.std(0) + 1e-8)
    rows = []
    rows += classic(XS, labels, "SigLIP B/16 image embedding", with_probe=False)
    rows += classic(prof, labels, "text profile (similarity to 945 names)", with_probe=False)
    both = np.hstack([normalize(PCA(50, random_state=0).fit_transform(X)), normalize(PCA(50, random_state=0).fit_transform(prof))])
    rows += classic(both, labels, "DINOv2-B + text profile", with_probe=False)
    top = np.array(vocab, dtype=object)[S.argmax(1)]
    keep, y = labels["name"]
    rows.append({"variant": "nearest name, zero-shot", "method": "accuracy against the name label", "name": round(float((top[keep] == y[keep]).mean()), 3)})
    rows.append({"variant": "nearest name, zero-shot", "method": "AMI against the name label", "name": round(float(ami(y[keep], top[keep])), 3),
                 "hs3": round(float(ami(labels["hs3"][1][labels["hs3"][0]], top[labels["hs3"][0]])), 3),
                 "family": round(float(ami(labels["family"][1][labels["family"][0]], top[labels["family"][0]])), 3)})
    print(rows[-2:], flush=True)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = sys.argv[1:] or ["base", "scan", "temi", "text"]
    sub, X, labels, idx = load()
    print(len(sub), "rows;", {c: (int(k.sum()), len(set(y[k]))) for c, (k, y) in labels.items()}, flush=True)
    if "base" in blocks:
        pd.DataFrame(classic(X, labels, "DINOv2-B, photo as published")).to_csv(OUT / "base.csv", index=False)
    if "objonly" in blocks:
        mask = pd.read_csv(OUT / "objonly_mask.csv")
        share = dict(zip(mask.file, mask.mask_share))
        empty = np.array([share[f] < 0.01 for f in sub.file])
        paths = [str(ROOT / "data/mimo" / f) if e else str(ROOT / "data/mimo_objonly" / Path(f).name) for f, e in zip(sub.file, empty)]
        print("objects whose mask keeps under 1 percent of the image (raw photo used):", int(empty.sum()), flush=True)
        XO = cached_embeddings(OUT / "emb_objonly", "dinov2_b", paths, pad_square=True)
        rows = classic(XO, labels, "DINOv2-B, object only")
        for r in rows:
            r["raw_fallback"] = int(empty.sum())
        pd.DataFrame(rows).to_csv(OUT / "objonly.csv", index=False)
    if "scan" in blocks:
        pd.DataFrame(run_scan(X, labels)).to_csv(OUT / "scan.csv", index=False)
    if "temi" in blocks:
        pd.DataFrame(run_temi(labels, idx)).to_csv(OUT / "temi_official_settings.csv", index=False)
    if "input" in blocks:
        run_input(sub, X, labels)
    if "text" in blocks:
        pd.DataFrame(run_text(sub, X, labels)).to_csv(OUT / "text.csv", index=False)


if __name__ == "__main__":
    main()
