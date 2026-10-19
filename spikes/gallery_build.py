"""Build the "seen and heard" gallery page: 20 blind photo clusters of MIMO instruments, 10 objects each, with thumbnail,
a 3-second excerpt of the museum's recording, name, H-S class, museum and the object's blind sound cluster.

Thumbnails are embedded as data URIs; audio excerpts are written as 16 kHz mono WAV files next to the page.
Output: runs/gallery/index.html and runs/gallery/audio/*.wav.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from librosa import resample
from PIL import Image, ImageOps

import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs/mimo_audio")
OUT = ROOT / "runs/gallery"
PER_CLUSTER, CLIP_S, SR = 10, 3.0, 16000
HSC = {"Aerophone": "#2a78d6", "Chordophone": "#eb6834", "Idiophone": "#1baf7a", "Membranophone": "#eda100", "Electrophone": "#e87ba4", "": "#8a8884"}


def excerpt(path, out):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != SR:
        y = resample(y, orig_sr=sr, target_sr=SR)
    L = int(CLIP_S * SR)
    if len(y) <= L:
        y = np.pad(y, (0, L - len(y)))
    else:  # loudest window, in 0.25 s steps
        step = SR // 4
        best, bi = -1, 0
        for s in range(0, len(y) - L, step):
            r = float(np.sqrt(np.mean(y[s:s + L] ** 2)))
            if r > best:
                best, bi = r, s
        y = y[bi:bi + L]
    fade = int(0.05 * SR)
    y[:fade] *= np.linspace(0, 1, fade); y[-fade:] *= np.linspace(1, 0, fade)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 0.9
    sf.write(out, y, SR, subtype="PCM_16")


def thumb(file):
    im = Image.open(ROOT / "data/mimo" / file).convert("RGB")
    im = ImageOps.pad(im, (180, 180), color=(250, 249, 247))
    b = io.BytesIO(); im.save(b, "JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    g = json.load(open(RUN / "gallery.json", encoding="utf-8"))
    full = pd.read_csv(ROOT / "runs/mimo_full/sample_full.csv", dtype=str).fillna("")
    file_of = dict(zip(full.id, full.file))
    aud = pd.read_csv(ROOT / "data/mimo_audio/audio.csv", dtype=str).fillna("")
    afile = {u.replace("http://www.mimo-international.com", "https://mimo-international.com"): f for u, f in zip(aud.audio_url, aud.file)}
    by_c = {}
    for it in g["items"]:
        by_c.setdefault(it["cluster"], []).append(it)
    clusters, cards = [], []
    for c in g["clusters"]:
        chosen = by_c[c["cluster"]][:PER_CLUSTER]
        for k, it in enumerate(chosen):
            wav = f"audio/c{c['cluster']:02d}_{k:02d}.wav"
            excerpt(ROOT / "data/mimo_audio" / afile[it["audio"]], OUT / wav)
            cards.append({"c": c["cluster"], "img": thumb(file_of[it["id"]]), "wav": wav, "name": it["name"] or it["title"].split(".")[0][:40],
                          "named": bool(it["name"]), "hs1": it["hs1"], "hs3": it["hs3"], "museum": it["museum"], "sound": it["audio_cluster"]})
        clusters.append({**c, "shown": len(chosen)})
        print(c["cluster"], c["majority_name"], len(chosen), flush=True)
    data = {"clusters": clusters, "cards": cards, "hsc": HSC}
    html = (ROOT / "spikes/gallery_template.html").read_text(encoding="utf-8").replace("/*DATA*/", json.dumps(data, ensure_ascii=False))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("page", round((OUT / "index.html").stat().st_size / 1e6, 2), "MB;", len(cards), "cards")


if __name__ == "__main__":
    main()
