"""Check each number quoted in IDEAS.md section 1 and the explainer against the clean result tables.
Prints claim, quoted value, table value, and OK / DIFF (difference after rounding to the quoted precision)."""
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "runs/clean"
rec = pd.read_csv(R / "records.csv", dtype=str).fillna("")
same = pd.read_csv(R / "same_rows.csv").set_index("label")
dep = pd.read_csv(R / "depth.csv").set_index("level")
tax = pd.read_csv(R / "taxonomy_vs_taxonomy.csv")
org = pd.read_csv(R / "origin.csv").set_index("label")
wm = pd.read_csv(R / "within_museum.csv")
btl = pd.read_csv(R / "blind_then_label.csv").set_index("k")
cr = pd.read_csv(R / "crossing_names.csv")
meth = pd.read_csv(R / "methods.csv").set_index("method")
aud = pd.read_csv(R / "audio/labels.csv"); auc = pd.read_csv(R / "audio/pair_auc.csv").set_index("grouping")
ivsa = pd.read_csv(R / "audio/image_vs_audio.csv").set_index("k")
au = lambda sp, lab, col="AMI_at_classes": aud[(aud.space == sp) & (aud.label == lab)][col].iloc[0]
tx = lambda a, b: tax[(tax.a == a) & (tax.b == b)].AMI.iloc[0]

claims = [
    ("records kept", 59661, len(rec)), ("coded", 40697, int((rec.hs1 != "").sum())), ("named", 40230, int((rec.name != "").sum())),
    ("with family", 34098, int((rec.family != "").sum())), ("with region", 35419, int((rec.region != "").sum())),
    ("with year", 6075, int((rec.year != "").sum())), ("museums", 216, rec.provider.nunique()),
    ("same rows n", 27348, int(same.loc["hs3", "n"])),
]
for lab, q in [("hs1", 0.35), ("hs2", 0.53), ("hs3", 0.55), ("hs4", 0.55), ("hs5", 0.57), ("hs6", 0.59), ("family", 0.57), ("name", 0.59)]:
    claims.append((f"same rows {lab} AMI", q, same.loc[lab, "AMI_at_classes"]))
for lab, a, pa in [("hs1", 0.97, 0.86), ("hs2", 0.97, 0.88), ("hs3", 0.92, 0.83), ("hs4", 0.86, 0.77), ("hs5", 0.87, 0.79), ("hs6", 0.88, 0.81), ("family", 0.96, 0.87), ("name", 0.67, 0.71)]:
    claims.append((f"same rows {lab} probe acc", a, same.loc[lab, "probe_acc"])); claims.append((f"same rows {lab} probe AMI", pa, same.loc[lab, "probe_AMI"]))
for lab, c in [("hs1", 5), ("hs2", 8), ("hs3", 13), ("hs4", 18), ("hs5", 30), ("hs6", 30), ("family", 6), ("name", 129)]:
    claims.append((f"same rows {lab} classes", c, int(same.loc[lab, "classes"])))
for lab in ("hs4", "hs5", "hs6"):
    claims.append((f"same rows {lab} n (rows actually scored)", 27348, int(same.loc[lab, "n"])))
claims += [("name vs hs3", 0.66, tx("name", "hs3")), ("name vs hs4", 0.74, tx("name", "hs4")), ("family vs hs1", 0.74, tx("family", "hs1")),
           ("family vs hs2", 0.80, tx("family", "hs2")), ("hs6 vs hs3", 0.81, tx("hs6", "hs3"))]
for lvl, k, d, c in [(1, 0.33, 0.32, 5), (2, 0.48, 0.51, 11), (3, 0.53, 0.55, 22), (4, 0.54, 0.54, 31), (5, 0.56, 0.57, 47), (6, 0.57, 0.58, 42)]:
    claims += [(f"depth L{lvl} kmeans", k, dep.loc[lvl, "AMI_kmeans"]), (f"depth L{lvl} dendro", d, dep.loc[lvl, "AMI_dendro"]), (f"depth L{lvl} classes", c, int(dep.loc[lvl, "classes"]))]
claims += [("btl k100 right hs3", 0.84, btl.loc[100, "objects_right_hs3"]), ("btl k100 right name", 0.50, btl.loc[100, "objects_right_name"]),
           ("btl k100 nameable", 46, int(btl.loc[100, "groups_nameable"])), ("btl k400 right hs3", 0.90, btl.loc[400, "objects_right_hs3"]),
           ("btl k400 right name", 0.62, btl.loc[400, "objects_right_name"]), ("btl k400 nameable", 250, int(btl.loc[400, "groups_nameable"])),
           ("crossing names (50+) count", 8, int(cr[(cr.n >= 50)].crosses.sum())), ("names with 50+", 112, int((cr.n >= 50).sum()))]
for m, vals in [("k-means", (0.55, 0.57, 0.57)), ("Ward", (0.56, 0.56, 0.59)), ("average-link cosine", (0.58, 0.69, 0.58)), ("UMAP 10-d + k-means", (0.57, 0.59, 0.58)),
                ("probe (supervised ceiling)", (0.81, 0.85, 0.68))]:
    for col, v in zip(("AMI_hs3", "AMI_family", "AMI_name"), vals):
        claims.append((f"methods {m} {col}", v, meth.loc[m, col]))
claims += [("region AMI", 0.13, org.loc["region", "AMI_at_classes"]), ("region probe acc", 0.84, org.loc["region", "probe_acc"]), ("region majority", 0.65, org.loc["region", "majority"]),
           ("country AMI", 0.20, org.loc["country", "AMI_at_classes"]), ("country classes", 76, int(org.loc["country", "classes"])),
           ("museum AMI", 0.18, org.loc["provider", "AMI_at_classes"]), ("museum classes", 84, int(org.loc["provider", "classes"]))]
h = wm[wm.label == "hs1n"]; rg = wm[wm.label == "region"]
claims += [("within museum hs1 min", 0.10, h.AMI_at_classes.min()), ("within museum hs1 max", 0.57, h.AMI_at_classes.max()), ("within museum hs1 count", 13, len(h)),
           ("within museum hs1 above 0.2", 12, int((h.AMI_at_classes > 0.2).sum())),
           ("within museum region max excl Berlin", 0.08, rg[~rg.provider.str.contains("Berlin")].AMI_at_classes.max()),
           ("within museum region min", 0.01, rg.AMI_at_classes.min()), ("Berlin region", 0.25, rg[rg.provider.str.contains("Berlin")].AMI_at_classes.iloc[0])]
claims += [("audio hs1", 0.62, au("audio", "hs1")), ("audio hs2", 0.57, au("audio", "hs2")), ("audio hs3", 0.74, au("audio", "hs3")), ("audio name", 0.94, au("audio", "name")),
           ("audio name classes", 7, int(au("audio", "name", "classes"))), ("image hs1 (audio objects)", 0.39, au("image", "hs1")), ("image hs2", 0.49, au("image", "hs2")),
           ("image hs3", 0.69, au("image", "hs3")), ("image name", 0.87, au("image", "name"))]
for k, v in [(5, 0.48), (10, 0.61), (20, 0.65), (40, 0.65)]:
    claims.append((f"image vs audio k={k}", v, ivsa.loc[k, "image_vs_audio"]))
claims += [("AUC image k40", 0.90, auc.loc["image cluster k=40", "AUC"]), ("AUC hs3", 0.90, auc.loc["hs3", "AUC"]), ("AUC name", 0.98, auc.loc["name", "AUC"]),
           ("AUC museum", 0.62, auc.loc["provider", "AUC"]), ("AUC permuted", 0.48, auc.loc["hs3 permuted (chance)", "AUC"])]

bad = 0
for name, q, v in claims:
    if isinstance(q, int):
        ok = q == v
    else:
        ok = abs(round(v, 2) - q) < 1e-9 or abs(v - q) <= 0.005 + 1e-9
    bad += not ok
    print(f"{'OK  ' if ok else 'DIFF'} {name:45s} quoted {q!s:>7}  table {v if isinstance(v, (int,)) else round(float(v), 4)!s:>8}")
print(f"\n{len(claims)} claims, {bad} differ")
