"""Small polite client for Wikipedia raw page source and Commons files, with Retry-After handling."""
import re
import time

import requests

UA = {"User-Agent": "ClusteringPraktikum/0.1 (LMU Munich student project; contact via course Moodle)"}


def session():
    s = requests.Session()
    s.trust_env = False
    s.headers.update(UA)
    return s


def get(s, url, params=None, tries=5, want="text"):
    """GET with backoff on 429/5xx. Returns the Response or None."""
    for t in range(tries):
        try:
            r = s.get(url, params=params, timeout=60)
        except Exception as e:  # noqa: BLE001
            print("  error", type(e).__name__, url[:60])
            time.sleep(2 * (t + 1))
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (429, 503, 502):
            wait = float(r.headers.get("Retry-After", 5 * (t + 1)))
            print(f"  http {r.status_code}, waiting {wait:.0f}s")
            time.sleep(min(wait, 60))
            continue
        return None  # 404 and other hard errors
    return None


def wikitext(s, title):
    """Raw wikitext of a Wikipedia page, following redirects. Empty string if missing."""
    for _ in range(3):  # action=raw does not resolve redirects, so follow "#REDIRECT [[Target]]" by hand
        r = get(s, "https://en.wikipedia.org/w/index.php", {"title": title, "action": "raw"})
        if r is None:
            return ""
        m = re.match(r"\s*#REDIRECT\s*\[\[([^\]|#]+)", r.text, re.I)
        if not m:
            return r.text
        title = m.group(1).strip()
    return ""


def infobox_field(wt, key):
    m = re.search(r"\|\s*" + key + r"\s*=\s*([^\n]+)", wt)
    if not m:
        return ""
    v = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", m.group(1))
    v = re.sub(r"\{\{[^}]*\}\}", "", v)
    v = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", v)
    v = re.sub(r"<[^>]+>", "", v)
    return v.strip(" |").strip()
