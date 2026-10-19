"""Minerals spike, step 1: map the 98 MineralImage5k species to crystal system and Strunz class via Wikipedia infoboxes."""
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "ClusteringPraktikum/0.1 (student research)"}
names = requests.get("https://huggingface.co/api/datasets/Nech-C/mineralimage5K-98", headers=UA, timeout=60).json()
names = names["cardData"]["dataset_info"]["features"][1]["dtype"]["class_label"]["names"]
rows = []
for idx, name in sorted(names.items(), key=lambda kv: int(kv[0])):
    title = name.replace("_", " ")
    r = {}
    for attempt in range(4):
        resp = requests.get("https://en.wikipedia.org/w/api.php", params={"action": "parse", "page": title, "prop": "wikitext",
                            "format": "json", "formatversion": 2, "redirects": 1}, headers=UA, timeout=60)
        if resp.status_code == 200 and "json" in resp.headers.get("content-type", ""):
            r = resp.json()
            break
        print("retry", title, resp.status_code, resp.headers.get("content-type"))
        time.sleep(3 * (attempt + 1))
    if "error" in r:
        print("no page:", title, r["error"].get("code"))
    wt = r.get("parse", {}).get("wikitext", "")
    def field(key):
        m = re.search(r"\|\s*" + key + r"\s*=\s*([^\n|]+)", wt)
        return re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", m.group(1)).strip() if m else ""
    rows.append({"label": int(idx), "name": name, "page": r.get("parse", {}).get("title", ""),
                 "system": field("system"), "strunz": field("strunz"), "category": field("category"), "formula": field("formula")})
    time.sleep(0.2)
df = pd.DataFrame(rows)
df.to_csv(ROOT / "data/minerals_labels.csv", index=False)
print(df.to_string())
print("\nsystem counts:", df.system.str.lower().str.split().str[0].value_counts().to_dict())
print("strunz top-level:", df.strunz.str[:1].value_counts().to_dict())
