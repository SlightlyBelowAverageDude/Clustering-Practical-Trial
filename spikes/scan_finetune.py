"""SCAN end to end: fine-tune the DINOv2-B encoder with the SCAN loss on all clean MIMO photos, without labels.

Neighbours (20 per photo) come from the frozen padded DINOv2-B embeddings (runs/clean/emb/dinov2_b_pad.npy). The
encoder is timm vit_base_patch14_dinov2.lvd142m at 224 x 224; the patch embedding and the first 8 of 12 blocks stay
frozen, the last 4 blocks, the final norm and 3 cluster heads are trained. The 3 heads have k = 13, 6 and 129 (the class
counts of level 3, family and name on the 12,000-row subset), each with its own SCAN loss: consistency between a photo
and a random neighbour under independent augmentations, plus an entropy term with weight 5 (1 head per k, no
self-labelling stage). AdamW, lr 1e-5 for the encoder and 1e-3 for the heads, 200 warm-up steps then cosine decay, fp16.
Scoring on the 12,000-row subset: head predictions (AMI), k-means on PCA 50 and the probe on the fine-tuned features,
next to the same on the frozen features at 224 (the published numbers use 518).
Usage: python spikes/scan_finetune.py [epochs | probe] [batch of anchors]
Outputs: runs/tests/scan_ft/scores.csv, emb_subset_frozen224.npy, emb_subset_ft.npy, heads_subset.npy, checkpoint.pt
"""
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/tests/scan_ft"
KS = {"hs3": 13, "family": 6, "name": 129}
FREEZE = 8
N_NB = 20
W_ENT = 5.0
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Photos(Dataset):
    def __init__(self, paths, nb=None):
        self.paths, self.nb = paths, nb
        self.aug = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                              T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), T.RandomGrayscale(0.2), T.ToTensor(), T.Normalize(MEAN, STD)])
        self.plain = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])

    def load(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        return ImageOps.pad(im, (max(im.size), max(im.size)), color=(255, 255, 255))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.nb is None:
            return self.plain(self.load(i)), i
        j = int(self.nb[i][random.randrange(self.nb.shape[1])])
        return self.aug(self.load(i)), self.aug(self.load(j))


def features(model, x):
    """CLS feature with the frozen part of the encoder run without a graph."""
    with torch.no_grad():
        x = model.patch_embed(x)
        x = model._pos_embed(x)
        x = model.patch_drop(x)
        x = model.norm_pre(x)
        for b in model.blocks[:FREEZE]:
            x = b(x)
    for b in model.blocks[FREEZE:]:
        x = b(x)
    return model.norm(x)[:, 0]


def scan_loss(pa, pn):
    cons = -torch.log((pa * pn).sum(1).clamp_min(1e-8)).mean()
    pm = pa.mean(0)
    return cons + W_ENT * (pm * torch.log(pm.clamp_min(1e-8))).sum()


@torch.no_grad()
def embed_all(model, heads, paths, bs=256):
    model.eval()
    dl = DataLoader(Photos(paths), batch_size=bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=False)
    feats, preds = [], []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            f = features(model, x.cuda(non_blocking=True))
        f = f.float()
        feats.append(f.cpu().numpy())
        preds.append(np.stack([h(f).argmax(1).cpu().numpy() for h in heads], 1))
    return np.concatenate(feats), np.concatenate(preds)


def score(Fs, P, labels, tag):
    rows = []
    Z = PCA(50, random_state=0).fit_transform(Fs)
    for method in ("head", "k-means", "probe"):
        r = {"features": tag, "method": method}
        for hi, (c, (keep, y)) in enumerate(labels.items()):
            yk = y[keep]
            k = len(set(yk))
            if method == "head":
                p = P[keep, hi] if P is not None else None
            elif method == "k-means":
                p = KMeans(k, n_init=10, random_state=0).fit_predict(Z[keep])
            else:
                p = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), Fs[keep], yk, cv=StratifiedKFold(5, shuffle=True, random_state=0))
            r[c] = round(float(ami(yk, p)), 3) if p is not None else None
            if method == "head" and p is not None:
                r[f"{c}_clusters_used"] = int(len(set(p)))
        rows.append(r)
        print(r, flush=True)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "15"
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    epochs = 1 if mode == "probe" else int(mode)
    torch.manual_seed(0)
    random.seed(0)
    rec = pd.read_csv(ROOT / "runs/clean/records.csv", dtype=str).fillna("")
    paths = [str(ROOT / "data/mimo" / f) for f in rec.file]
    ids = pd.read_csv(ROOT / "runs/clean/methods_subset_ids.csv", dtype=str).id
    pos = dict(zip(rec.id, range(len(rec))))
    sub = np.array([pos[i] for i in ids])
    labels = {}
    for c in KS:
        y = np.array(rec[c].iloc[sub].tolist(), dtype=object)
        vc = pd.Series(y).value_counts()
        labels[c] = ((pd.Series(y).map(vc).values >= 5) & (y != ""), y)
        assert len(set(y[labels[c][0]])) == KS[c], (c, len(set(y[labels[c][0]])))

    E = torch.tensor(np.load(ROOT / "runs/clean/emb/dinov2_b_pad.npy"), device="cuda")
    nb = []
    for s in range(0, len(E), 4096):
        S = E[s:s + 4096] @ E.T
        S[torch.arange(S.shape[0]), torch.arange(s, s + S.shape[0])] = -2
        nb.append(S.topk(N_NB, dim=1).indices.cpu())
    nb = torch.cat(nb).numpy()
    del E
    print("neighbours done", nb.shape, flush=True)

    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = [p for b in model.blocks[FREEZE:] for p in b.parameters()] + list(model.norm.parameters())
    for p in trainable:
        p.requires_grad_(True)
    heads = torch.nn.ModuleList([torch.nn.Linear(768, k) for k in KS.values()]).cuda()
    print("trainable encoder parameters", sum(p.numel() for p in trainable) // 1_000_000, "M", flush=True)

    if mode != "probe":
        F0, _ = embed_all(model, heads, [paths[i] for i in sub])
        np.save(OUT / "emb_subset_frozen224.npy", F0)
        rows = score(F0, None, labels, "frozen, 224")
    else:
        rows = []

    dl = DataLoader(Photos(paths, nb), batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    opt = torch.optim.AdamW([{"params": trainable, "lr": 1e-5}, {"params": heads.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    base = [g["lr"] for g in opt.param_groups]
    total = epochs * len(dl)
    warm = 200
    scaler = torch.amp.GradScaler("cuda")
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for b in model.blocks[:FREEZE]:
            b.eval()
        run = 0.0
        for it, (xa, xn) in enumerate(dl):
            f_lr = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total))))
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * f_lr
            x = torch.cat([xa, xn]).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                f = features(model, x)
            fa, fn = f.float().chunk(2)
            loss = sum(scan_loss(h(fa).softmax(1), h(fn).softmax(1)) for h in heads)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += float(loss)
            step += 1
            if mode == "probe" and it == 9:
                torch.cuda.synchronize()
                t0 = time.time()
            if mode == "probe" and it == 39:
                torch.cuda.synchronize()
                dt = (time.time() - t0) / 30
                print(f"probe: {dt:.2f} s per step of {2 * bs} images, {len(dl)} steps per epoch, about {dt * len(dl) / 60:.1f} min per epoch, "
                      f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
                return
            if it % 100 == 0:
                print(f"epoch {ep} step {it}/{len(dl)} loss {run / (it + 1):.3f} lr {opt.param_groups[0]['lr']:.2e} {(time.time() - t0) / 60:.1f} min", flush=True)
        print(f"epoch {ep} mean loss {run / len(dl):.3f}", flush=True)
        torch.save({"blocks": [b.state_dict() for b in model.blocks[FREEZE:]], "norm": model.norm.state_dict(), "heads": heads.state_dict(), "epoch": ep},
                   OUT / "checkpoint.pt")

    F1, P1 = embed_all(model, heads, [paths[i] for i in sub])
    np.save(OUT / "emb_subset_ft.npy", F1)
    np.save(OUT / "heads_subset.npy", P1)
    rows += score(F1, P1, labels, f"SCAN fine-tuned, {epochs} epochs")
    pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)


if __name__ == "__main__":
    main()
