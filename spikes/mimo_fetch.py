"""MIMO step 2: stratified image sample through Europeana preview thumbnails (400 px).

Keeps records with an H-S notation. Labels: hs1 = first digit (1 idiophone, 2 membranophone, 3 chordophone,
4 aerophone, 5 electrophone), hs2 = first 2 digits, hs3 = first 3 digits, country of manufacture parsed from
the place label, museum (data provider), year. Sampling caps each museum and each hs1 class so that Paris
and chordophones do not dominate.

Usage: python spikes/mimo_fetch.py --n 4000 --per-provider 600
"""
import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mimo"
UA = {"User-Agent": "ClusteringPraktikum/0.1 (LMU Munich student project; contact via course Moodle)"}
COUNTRY_WORDS = {  # German, French, English, Italian, Dutch place labels seen in MIMO
    "Frankreich": "France", "France": "France", "Francia": "France", "Paris": "France", "Deutschland": "Germany", "Allemagne": "Germany",
    "Germany": "Germany", "Germania": "Germany", "Italien": "Italy", "Italie": "Italy", "Italy": "Italy", "Italia": "Italy",
    "England": "United Kingdom", "Angleterre": "United Kingdom", "Großbritannien": "United Kingdom", "United Kingdom": "United Kingdom",
    "Royaume-Uni": "United Kingdom", "London": "United Kingdom", "Spanien": "Spain", "Espagne": "Spain", "Spain": "Spain",
    "Belgien": "Belgium", "Belgique": "Belgium", "Belgium": "Belgium", "Niederlande": "Netherlands", "Pays-Bas": "Netherlands",
    "Netherlands": "Netherlands", "Österreich": "Austria", "Autriche": "Austria", "Austria": "Austria", "Wien": "Austria",
    "Schweiz": "Switzerland", "Suisse": "Switzerland", "Switzerland": "Switzerland", "Tschechien": "Czechia", "Böhmen": "Czechia",
    "Bohême": "Czechia", "Czech": "Czechia", "Japan": "Japan", "Japon": "Japan", "China": "China", "Chine": "China", "Indien": "India",
    "Inde": "India", "India": "India", "Afrika": "Africa", "Afrique": "Africa", "Africa": "Africa", "Türkei": "Turkey", "Turquie": "Turkey",
    "Turkey": "Turkey", "Iran": "Iran", "Persien": "Iran", "Indonesien": "Indonesia", "Indonésie": "Indonesia", "Indonesia": "Indonesia",
    "Java": "Indonesia", "Bali": "Indonesia", "USA": "United States", "États-Unis": "United States", "Vereinigte Staaten": "United States",
    "United States": "United States", "Brasilien": "Brazil", "Brésil": "Brazil", "Mexiko": "Mexico", "Mexique": "Mexico",
    "Kongo": "Congo", "Congo": "Congo", "Kamerun": "Cameroon", "Cameroun": "Cameroon", "Nigeria": "Nigeria", "Ghana": "Ghana",
    "Senegal": "Senegal", "Sénégal": "Senegal", "Mali": "Mali", "Marokko": "Morocco", "Maroc": "Morocco", "Algerien": "Algeria",
    "Algérie": "Algeria", "Tunesien": "Tunisia", "Tunisie": "Tunisia", "Ägypten": "Egypt", "Égypte": "Egypt", "Egypt": "Egypt",
    "Russland": "Russia", "Russie": "Russia", "Polen": "Poland", "Pologne": "Poland", "Ungarn": "Hungary", "Hongrie": "Hungary",
    "Vietnam": "Vietnam", "Viêt Nam": "Vietnam", "Thailand": "Thailand", "Thaïlande": "Thailand", "Birma": "Myanmar", "Myanmar": "Myanmar",
    "Korea": "Korea", "Corée": "Korea", "Tibet": "Tibet", "Nepal": "Nepal", "Népal": "Nepal", "Peru": "Peru", "Pérou": "Peru",
    "Bolivien": "Bolivia", "Bolivie": "Bolivia", "Schweden": "Sweden", "Suède": "Sweden", "Sweden": "Sweden", "Norwegen": "Norway",
    "Norvège": "Norway", "Finnland": "Finland", "Finlande": "Finland", "Suomi": "Finland", "Portugal": "Portugal", "Griechenland": "Greece",
    "Grèce": "Greece", "Greece": "Greece", "Australien": "Australia", "Australie": "Australia", "Neuguinea": "New Guinea",
    "Papua New Guinea": "New Guinea", "New Guinea": "New Guinea", "Papua": "New Guinea",
    "Nouvelle-Guinée": "New Guinea", "Ozeanien": "Oceania", "Océanie": "Oceania", "Philippinen": "Philippines", "Philippines": "Philippines",
}

EXTRA = {  # Europeana enrichment labels (Greek, German), plain English country names, frequent cities
    "Vereinigtes Königreich": "United Kingdom", "Λονδίνο": "United Kingdom", "Γλασκώβη": "United Kingdom", "Isle of Man": "United Kingdom",
    "Isle Of Man": "United Kingdom", "Glasgow": "United Kingdom", "Edinburgh": "United Kingdom", "Manchester": "United Kingdom",
    "Γερμανία": "Germany", "Markneukirchen": "Germany", "München": "Germany", "Berlin": "Germany", "Nürnberg": "Germany", "Leipzig": "Germany",
    "Ιταλία": "Italy", "Firenze": "Italy", "Napoli": "Italy", "Milano": "Italy", "Cremona": "Italy", "Venezia": "Italy", "Roma": "Italy",
    "Γαλλία": "France", "Παρίσι": "France", "Mirecourt": "France", "Lyon": "France", "Ισπανία": "Spain", "Barcelona": "Spain", "Madrid": "Spain",
    "Βέλγιο": "Belgium", "Bruxelles": "Belgium", "Brussel": "Belgium", "Antwerpen": "Belgium", "Ολλανδία": "Netherlands", "Amsterdam": "Netherlands",
    "Αυστρία": "Austria", "Ελβετία": "Switzerland", "Σουηδία": "Sweden", "Stockholm": "Sweden", "Νορβηγία": "Norway", "Δανία": "Denmark",
    "Dänemark": "Denmark", "Danemark": "Denmark", "Denmark": "Denmark", "Ιρλανδία": "Ireland", "Ireland": "Ireland", "Irland": "Ireland",
    "Πορτογαλία": "Portugal", "Τσεχία": "Czechia", "Czechia": "Czechia", "Czech Republic": "Czechia", "Πολωνία": "Poland", "Poland": "Poland",
    "Ουγγαρία": "Hungary", "Hungary": "Hungary", "Ρωσία": "Russia", "Russia": "Russia", "Ελλάδα": "Greece", "Τουρκία": "Turkey",
    "Αφρική": "Africa", "Afrika": "Africa", "Togo": "Togo", "Tanzania": "Tanzania", "Tanzanija": "Tanzania", "United Republic of Tanzania": "Tanzania",
    "Tansania": "Tanzania", "Côte d’Ivoire": "Ivory Coast", "Côte d'Ivoire": "Ivory Coast", "Ivory Coast": "Ivory Coast", "Elfenbeinküste": "Ivory Coast",
    "Guinea": "Guinea", "Guinée": "Guinea", "Burkina Faso": "Burkina Faso", "Uganda": "Uganda", "Ouganda": "Uganda", "Malawi": "Malawi",
    "Sudan": "Sudan", "Soudan": "Sudan", "Kenya": "Kenya", "Kenia": "Kenya", "Madagascar": "Madagascar", "Madagaskar": "Madagascar",
    "Benin": "Benin", "Bénin": "Benin", "Gabon": "Gabon", "Gabonese Republic": "Gabon", "Cameroon": "Cameroon", "Morocco": "Morocco", "Μαρόκο": "Morocco",
    "Τυνησία": "Tunisia", "Tunisia": "Tunisia", "Αλγερία": "Algeria", "Algeria": "Algeria", "Αίγυπτος": "Egypt", "Ethiopia": "Ethiopia", "Äthiopien": "Ethiopia",
    "Éthiopie": "Ethiopia", "Αιθιοπία": "Ethiopia", "Zaïre": "Congo", "Zaire": "Congo", "Rwanda": "Rwanda", "Ruanda": "Rwanda", "Burundi": "Burundi",
    "Angola": "Angola", "Zambia": "Zambia", "Sambia": "Zambia", "Zimbabwe": "Zimbabwe", "Simbabwe": "Zimbabwe", "Mozambique": "Mozambique",
    "Mosambik": "Mozambique", "South Africa": "South Africa", "Südafrika": "South Africa", "Afrique du Sud": "South Africa", "Namibia": "Namibia",
    "Liberia": "Liberia", "Sierra Leone": "Sierra Leone", "Gambia": "Gambia", "Gambie": "Gambia", "Niger": "Niger", "Chad": "Chad", "Tschad": "Chad",
    "Tchad": "Chad", "Central African Republic": "Central African Republic", "Zentralafrikanische Republik": "Central African Republic",
    "Ινδία": "India", "Κίνα": "China", "Ιαπωνία": "Japan", "Βιετνάμ": "Vietnam", "Sri Lanka": "Sri Lanka", "Ceylon": "Sri Lanka", "Pakistan": "Pakistan",
    "Afghanistan": "Afghanistan", "Bangladesh": "Bangladesh", "Cambodia": "Cambodia", "Kambodscha": "Cambodia", "Cambodge": "Cambodia", "Laos": "Laos",
    "Malaysia": "Malaysia", "Malaisie": "Malaysia", "Mongolia": "Mongolia", "Mongolei": "Mongolia", "Mongolie": "Mongolia", "Ασία": "Asia", "Asie": "Asia",
    "Asien": "Asia", "Asia": "Asia", "Irak": "Iraq", "Iraq": "Iraq", "Syrien": "Syria", "Syrie": "Syria", "Syria": "Syria", "Israel": "Israel",
    "Saudi": "Saudi Arabia", "Yemen": "Yemen", "Jemen": "Yemen", "Oman": "Oman", "Usbekistan": "Uzbekistan", "Uzbekistan": "Uzbekistan",
    "Argentina": "Argentina", "Argentinien": "Argentina", "Argentine": "Argentina", "Chile": "Chile", "Chili": "Chile", "Colombia": "Colombia",
    "Kolumbien": "Colombia", "Colombie": "Colombia", "Ecuador": "Ecuador", "Équateur": "Ecuador", "Venezuela": "Venezuela", "Guatemala": "Guatemala",
    "Cuba": "Cuba", "Kuba": "Cuba", "Haiti": "Haiti", "Haïti": "Haiti", "Jamaica": "Jamaica", "Canada": "Canada", "Kanada": "Canada",
    "Ευρώπη": "Europe", "Europa": "Europe", "Europe": "Europe", "New Zealand": "New Zealand", "Neuseeland": "New Zealand", "Nouvelle-Zélande": "New Zealand",
    "Fiji": "Fiji", "Fidschi": "Fiji", "Samoa": "Samoa", "Tonga": "Tonga", "Hawaii": "United States", "Solomon": "Solomon Islands", "Vanuatu": "Vanuatu",
}
COUNTRY_WORDS.update(EXTRA)
# continents stand in only when no country matched: try country names first, continents last
CONTINENTS = {"Africa", "Asia", "Europe", "Oceania"}


def country_of(place_fields):
    text = " | ".join(str(p) for p in place_fields if p)
    best = ""
    for w, c in sorted(COUNTRY_WORDS.items(), key=lambda kv: -len(kv[0])):  # longest word first: "New Guinea" before "Guinea"
        if re.search(r"(?<![A-Za-zÀ-ÿΑ-ω])" + re.escape(w) + r"(?![A-Za-zÀ-ÿΑ-ω])", text):
            if c not in CONTINENTS:
                return c
            best = best or c
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--per-provider", type=int, default=600)
    ap.add_argument("--per-hs1", type=int, default=1200)
    a = ap.parse_args()
    rec = pd.read_csv(DATA / "records.csv").fillna("")
    voc = pd.read_csv(DATA / "hs_vocab.csv").fillna("")
    df = rec.merge(voc, on="hs_uri", how="left")
    df = df[(df.notation.astype(str) != "") & (df.preview != "")].copy()
    df["notation"] = df.notation.astype(str)
    df["hs1"] = df.notation.str[0]
    df["hs2"] = df.notation.str[:2]
    df["hs3"] = df.notation.str[:3]
    df["country"] = [country_of([a_, b_, c_]) for a_, b_, c_ in zip(df.place_en, df.place, df.places_all)]
    df["year_num"] = pd.to_numeric(df.year.astype(str).str[:4], errors="coerce")
    print(len(df), "records with H-S notation and preview;", (df.country != "").sum(), "with a parsed country;", df.year_num.notna().sum(), "with year")
    print("hs1:", df.hs1.value_counts().to_dict())
    print("providers:", df.provider.value_counts().head(12).to_dict())
    print("countries:", df.country.value_counts().head(15).to_dict())
    rng = 0
    parts = []
    for prov, g in df.groupby("provider"):
        parts.append(g.sample(min(len(g), a.per_provider), random_state=rng))
    s = pd.concat(parts)
    parts = []
    for h, g in s.groupby("hs1"):
        parts.append(g.sample(min(len(g), a.per_hs1), random_state=rng))
    s = pd.concat(parts)
    if len(s) > a.n:
        s = s.sample(a.n, random_state=rng)
    s = s.reset_index(drop=True)
    s["file"] = ["images/" + re.sub(r"[^A-Za-z0-9]+", "_", i.strip("/")) + ".jpg" for i in s.id]
    (DATA / "images").mkdir(exist_ok=True)
    sess = requests.Session()
    sess.headers.update(UA)

    def fetch(row):
        fn = DATA / row.file
        if fn.exists():
            return True
        for attempt in range(3):
            try:
                r = sess.get(row.preview, timeout=60)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                    fn.write_bytes(r.content)
                    return True
                time.sleep(2 * (attempt + 1))
            except Exception:  # noqa: BLE001
                time.sleep(2 * (attempt + 1))
        return False

    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        ok = list(ex.map(fetch, s.itertuples()))
    s = s[ok].reset_index(drop=True)
    s.to_csv(DATA / "sample.csv", index=False)
    print(f"{len(s)} images in {time.time() - t0:.0f}s")
    print(s.groupby("hs1").size().to_dict()); print(s.provider.value_counts().head(10).to_dict()); print(s.country.value_counts().head(12).to_dict())


if __name__ == "__main__":
    main()
