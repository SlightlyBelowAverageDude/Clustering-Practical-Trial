"""Dogs spike, step 1: map the 120 Stanford Dogs breeds to FCI group, AKC group and country via Wikipedia infoboxes."""
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wiki import infobox_field, session, wikitext  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
info = requests.get("https://huggingface.co/api/datasets/maurice-fp/stanford-dogs", timeout=60).json()
names = info["cardData"]["dataset_info"]["features"][1]["dtype"]["class_label"]["names"]
ALIAS = {"Japanese_spaniel": "Japanese Chin", "Maltese_dog": "Maltese dog", "Blenheim_spaniel": "Cavalier King Charles Spaniel",
         "toy_terrier": "English Toy Terrier (Black & Tan)", "Walker_hound": "Treeing Walker Coonhound", "bluetick": "Bluetick Coonhound",
         "redbone": "Redbone Coonhound", "Scotch_terrier": "Scottish Terrier", "Brabancon_griffon": "Griffon Bruxellois",
         "Pembroke": "Pembroke Welsh Corgi", "Cardigan": "Cardigan Welsh Corgi", "Mexican_hairless": "Xoloitzcuintle",
         "African_hunting_dog": "African wild dog", "dhole": "Dhole", "dingo": "Dingo", "Eskimo_dog": "Canadian Eskimo Dog",
         "Great_Pyrenees": "Great Pyrenees", "Bouvier_des_Flandres": "Bouvier des Flandres", "Sussex_spaniel": "Sussex Spaniel",
         "curly-coated_retriever": "Curly-coated Retriever", "flat-coated_retriever": "Flat-coated Retriever",
         "soft-coated_wheaten_terrier": "Soft-coated Wheaten Terrier", "West_Highland_white_terrier": "West Highland White Terrier",
         "Staffordshire_bullterrier": "Staffordshire Bull Terrier", "American_Staffordshire_terrier": "American Staffordshire Terrier",
         "Dandie_Dinmont": "Dandie Dinmont Terrier", "Lhasa": "Lhasa Apso", "Newfoundland": "Newfoundland dog",
         "kelpie": "Australian Kelpie", "collie": "Rough Collie", "groenendael": "Belgian Shepherd", "malinois": "Belgian Shepherd",
         "papillon": "Papillon dog", "basset": "Basset Hound", "clumber": "Clumber Spaniel", "cocker_spaniel": "American Cocker Spaniel",
         "toy_poodle": "Poodle", "miniature_poodle": "Poodle", "standard_poodle": "Poodle", "boxer": "Boxer (dog)",
         "chow": "Chow Chow", "Pomeranian": "Pomeranian dog", "Samoyed": "Samoyed dog", "Leonberg": "Leonberger",
         "Appenzeller": "Appenzeller Sennenhund", "EntleBucher": "Entlebucher Mountain Dog", "malamute": "Alaskan Malamute",
         "Saint_Bernard": "St. Bernard (dog)", "Doberman": "Dobermann", "bull_mastiff": "Bullmastiff", "Boston_bull": "Boston Terrier",
         "Pekinese": "Pekingese", "Shih-Tzu": "Shih Tzu", "silky_terrier": "Australian Silky Terrier",
         "wire-haired_fox_terrier": "Wire Fox Terrier", "Brittany_spaniel": "Brittany dog", "English_springer": "English Springer Spaniel",
         "Airedale": "Airedale Terrier", "cairn": "Cairn Terrier", "Chihuahua": "Chihuahua (dog)",
         "German_short-haired_pointer": "German Shorthaired Pointer", "black-and-tan_coonhound": "Black and Tan Coonhound"}

s = session()
rows = []
for idx, name in sorted(names.items(), key=lambda kv: int(kv[0])):
    short = name.split("-", 1)[1]
    title = ALIAS.get(short, short.replace("_", " "))
    wt = wikitext(s, title)
    if not wt:
        print("no page:", short, "->", title, flush=True)
    fcistd = infobox_field(wt, "fcistd")  # e.g. http://www.fci.be/Nomenclature/Standards/161g06-en.pdf: standard 161, group 6
    m = re.search(r"/(\d+)g(\d\d)", fcistd)
    rows.append({"label": int(idx), "wnid_name": name, "breed": short, "title": title,
                 "fcistd": fcistd, "fci_standard": m.group(1) if m else "", "fcigroup": str(int(m.group(2))) if m else "",
                 "akcgroup": infobox_field(wt, "akcgroup"), "country": infobox_field(wt, "country")})
    time.sleep(0.7)
df = pd.DataFrame(rows)
df.to_csv(ROOT / "data/dogs_labels.csv", index=False)
print(df.to_string())
print("\nfcigroup counts:", df.fcigroup.value_counts().to_dict())
print("akcgroup counts:", df.akcgroup.value_counts().to_dict())
print("missing fcigroup:", df[df.fcigroup == ""].breed.tolist())
