"""MIMO audio: find every MIMO record on Europeana with an audio/mpeg web resource, read the audio URLs from the full
record, and download the files. Output: data/mimo_audio/audio.csv (record id, audio URL, file, museum, H-S) and
data/mimo_audio/files/*.mp3. Joins to data/mimo/records.csv through the record id.
"""
import re
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

try:  # mimo-international.com serves a chain that certifi rejects; the Windows store accepts it
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mimo_audio"
UA = {"User-Agent": "ClusteringPraktikum/0.1 (LMU Munich student project; contact via course Moodle)"}
SEARCH = ("https://api.europeana.eu/record/v2/search.json?wskey=api2demo&rows=100&query=*&profile=minimal"
          "&qf=PROVIDER:%22MIMO%20-%20Musical%20Instrument%20Museums%20Online%22&qf=MIME_TYPE:%22audio/mpeg%22")


def main():
    (OUT / "files").mkdir(parents=True, exist_ok=True)
    s = requests.Session(); s.headers.update(UA)
    if (OUT / "audio.csv").exists() and "file" in pd.read_csv(OUT / "audio.csv", nrows=1).columns:
        df = pd.read_csv(OUT / "audio.csv", dtype=str).fillna("")
        print("reusing", len(df), "audio resources from audio.csv", flush=True)
        download(df, s)
        return
    ids, cursor = [], "*"
    while cursor:
        r = s.get(SEARCH + "&cursor=" + urllib.parse.quote(cursor), timeout=60).json()
        ids += [it["id"] for it in r.get("items", [])]
        cursor = r.get("nextCursor")
        time.sleep(0.25)
    print(len(ids), "records with audio/mpeg", flush=True)
    rows = []
    for i, rid in enumerate(ids):
        for attempt in range(3):
            try:
                j = s.get(f"https://api.europeana.eu/record/v2{rid}.json?wskey=api2demo", timeout=60).json()
                break
            except Exception:  # noqa: BLE001
                time.sleep(5)
        else:
            continue
        obj = j.get("object", {})
        agg = (obj.get("aggregations") or [{}])[0]
        wr = [w for w in agg.get("webResources", []) if str(w.get("ebucoreHasMimeType", "")).startswith("audio")]
        for k, w in enumerate(wr):
            rows.append({"id": rid, "provider": (agg.get("edmDataProvider", {}) or {}).get("def", [""])[0] if isinstance(agg.get("edmDataProvider"), dict) else "",
                         "audio_url": w.get("about"), "mime": w.get("ebucoreHasMimeType"), "bytes": w.get("ebucoreFileByteSize"),
                         "duration_ms": w.get("ebucoreDuration"), "k": k})
        if i % 100 == 0:
            print(i, len(rows), flush=True); pd.DataFrame(rows).to_csv(OUT / "audio.csv", index=False)
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df["file"] = ["files/" + re.sub(r"[^A-Za-z0-9]+", "_", i.strip("/")) + f"_{k}.mp3" for i, k in zip(df.id, df.k)]
    df.to_csv(OUT / "audio.csv", index=False)
    print(len(df), "audio resources on", df.id.nunique(), "records;", df.provider.value_counts().to_dict(), flush=True)
    download(df, s)


def download(df, s):
    ok = []
    t0 = time.time()
    for i, r in enumerate(df.itertuples()):
        fn = OUT / r.file
        if fn.exists():
            ok.append(True); continue
        good = False
        for attempt in range(3):
            try:
                x = s.get(r.audio_url, timeout=120)
                if x.status_code == 200 and len(x.content) > 1000:
                    fn.write_bytes(x.content); good = True; break
                if x.status_code == 404:  # missing on the server (Paris and Soinuenea); no point retrying
                    break
                if x.status_code == 429:
                    time.sleep(30)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)
        ok.append(good)
        if i % 100 == 0:
            print(f"  {i} files, {sum(ok)} ok, {time.time() - t0:.0f}s", flush=True)
        time.sleep(0.3)
    df["ok"] = ok
    df.to_csv(OUT / "audio.csv", index=False)
    print(f"{sum(ok)} of {len(df)} audio files downloaded in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
