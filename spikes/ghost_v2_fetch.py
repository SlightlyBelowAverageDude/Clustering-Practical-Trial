"""Ghost Borders, attempt 2: positive controls, cropland masking, 2 dates, explicit field-boundary features.

Borders (all from REGIS, CC BY 4.0):
  cz_at        Czechia vs Austria, still a state border (positive control 1, GeoVille shows it in parcels)
  inner_de     former inner German border, ended 1990 (positive control 2, DLR 2024 shows field-size differences)
  pl_partition former border of the German Empire (1914) inside present-day Poland, ended 1918 (the historic case)
For each border: tiles sampled 3 to 15 km on both sides, only where ESA WorldCover 2021 says >= 60 percent
cropland in the 2.56 km window, plus 2 placebo sets around the same line shifted 35 km east and west.
Per tile: Sentinel-2 true colour at 2 dates (early and late summer 2025), NDVI at both dates, and explicit
features computed on cropland pixels: NDVI edge density (field boundaries) and mean field-segment area.

Usage: python spikes/ghost_v2_fetch.py --n-real 120 --n-placebo 80
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer
from pystac_client import Client
from rasterio.windows import Window
from scipy import ndimage
from shapely import affinity
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "dl/regis/GeoPackages/Sovereign_States"
CRS_M = "EPSG:3035"  # LAEA Europe, metres
STAC = "https://earth-search.aws.element84.com/v1"
WC = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
DATES = {"d1": "2025-05-10/2025-06-25", "d2": "2025-07-15/2025-09-10"}
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
           GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", VSI_CACHE="TRUE", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")


def poly(country, layer):
    f = REG / country / f"REGIS_{country.replace('-', '_')}.gpkg"  # folder "Austria-Hungary", file "REGIS_Austria_Hungary.gpkg"
    return unary_union(gpd.read_file(f, layer=layer).to_crs(CRS_M).geometry)


def borders():
    """name -> (line, east_or_A polygon, west_or_B polygon, bbox in WGS84 lon/lat)."""
    out = {}
    cz, at = poly("Czechia", "REGIS_Czechia_L0_1993-2020"), poly("Austria", "REGIS_Austria_L0_1946-2020")
    out["cz_at"] = (cz.boundary.intersection(at.buffer(300)), cz, at, (15.5, 48.5, 17.0, 49.1))
    gdr, frg = poly("GDR", "REGIS_GDR_L0_1949-1990"), poly("Germany", "REGIS_Germany_L0_1957-1990")
    out["inner_de"] = (gdr.boundary.intersection(frg.buffer(300)), gdr, frg, (10.5, 52.0, 11.6, 52.9))
    de1914, pl = poly("Germany", "REGIS_Germany_L0_1871-1918"), poly("Poland", "REGIS_Poland_L0_1951-2020")
    inside = pl.buffer(-8000)  # keep away from today's borders
    line = de1914.boundary.intersection(inside)
    out["pl_partition"] = (line, inside.difference(de1914), inside.intersection(de1914), (17.5, 51.8, 19.6, 53.2))
    # Austrian partition (Galicia) vs Russian partition inside Poland, ended 1918. Exclude the stretch shared with Germany.
    ah = poly("Austria-Hungary", "REGIS_Austria_Hungary_L0_1908-1914")
    line_ah = ah.boundary.intersection(inside).difference(de1914.buffer(5000))
    out["pl_austrian"] = (line_ah, inside.difference(ah).difference(de1914), inside.intersection(ah), (19.5, 50.0, 22.5, 51.0))
    # Baden vs Württemberg (Württemberg-Baden and Württemberg-Hohenzollern), merged 1952
    g = gpd.read_file(REG / "Germany/REGIS_Germany.gpkg", layer="REGIS_Germany_L1_1949-1952").to_crs(CRS_M)
    baden = unary_union(g[g.name.str.startswith("Baden")].geometry)
    wuertt = unary_union(g[g.name.str.startswith("W")].geometry)
    out["de_bw_1952"] = (baden.boundary.intersection(wuertt.buffer(300)), baden, wuertt, (7.8, 47.6, 9.6, 49.6))
    # Braunschweig and Oldenburg vs the Prussian province of Hanover (German Empire L2), all merged into Niedersachsen in 1946
    g2 = gpd.read_file(REG / "Germany/REGIS_Germany.gpkg", layer="REGIS_Germany_L2_1884-1918").to_crs(CRS_M)
    hannover = unary_union(g2[g2.name == "Hannover"].geometry)
    small = unary_union(g2[g2.name.isin(["Braunschweig", "Oldenburg"])].geometry)
    out["de_nds_1946"] = (small.boundary.intersection(hannover.buffer(300)), small, hannover, (7.8, 52.2, 11.2, 53.6))
    return out


SHIFT_AXIS = {"inner_de": "y"}  # the inner German border runs north-south through a narrow band: shift placebos north and south


def band_box(b):
    return gpd.GeoSeries([box(*b)], crs="EPSG:4326").to_crs(CRS_M).iloc[0]


def candidates(polygon, line, band, d_min, d_max, spacing=2000):
    region = polygon.intersection(band).intersection(line.buffer(d_max)).difference(line.buffer(d_min))
    if region.is_empty:
        return []
    minx, miny, maxx, maxy = region.bounds
    pts = [Point(x, y) for x in np.arange(minx, maxx, spacing) for y in np.arange(miny, maxy, spacing)]
    return [p for p in pts if region.contains(p)]


class WorldCover:
    def __init__(self):
        self.ds = {}

    def cropland_fraction(self, lon, lat, size=256):
        tile = f"N{int(lat // 3 * 3):02d}E{int(lon // 3 * 3):03d}"
        if tile not in self.ds:
            self.ds[tile] = rasterio.open(WC.format(tile=tile))
        ds = self.ds[tile]
        row, col = ds.index(lon, lat)
        a = ds.read(1, window=Window(col - size // 2, row - size // 2, size, size))
        return float((a == 40).mean()), float((a == 10).mean())


def find_items(bbox, date_range, max_cloud=12):
    cl = Client.open(STAC)
    items = list(cl.search(collections=["sentinel-2-l2a"], bbox=list(bbox), datetime=date_range,
                           query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=300).items())
    items.sort(key=lambda it: it.properties["eo:cloud_cover"])
    return [(it, shape(it.geometry)) for it in items]


def item_epsg(item):
    p = item.properties
    return int(p["proj:epsg"]) if p.get("proj:epsg") else int(str(p["proj:code"]).split(":")[-1])


def read_window(href, epsg, lon, lat, size, bands):
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = to_utm.transform(lon, lat)
    with rasterio.open(href) as ds:
        row, col = ds.index(x, y)
        r0, c0 = row - size // 2, col - size // 2
        if r0 < 0 or c0 < 0 or r0 + size > ds.height or c0 + size > ds.width:
            return None
        return ds.read(bands, window=Window(c0, r0, size, size))


def read_tile(item, lon, lat, size):
    epsg = item_epsg(item)
    vis = read_window(item.assets["visual"].href, epsg, lon, lat, size, [1, 2, 3])
    if vis is None or vis.mean() < 5:
        return None
    red = read_window(item.assets["red"].href, epsg, lon, lat, size, [1])[0].astype(np.float32)
    nir = read_window(item.assets["nir"].href, epsg, lon, lat, size, [1])[0].astype(np.float32)
    offset = float(item.assets["red"].extra_fields.get("raster:bands", [{}])[0].get("offset", -1000) or -1000)
    red, nir = red + offset, nir + offset
    ndvi = (nir - red) / np.maximum(nir + red, 1.0)
    return np.transpose(vis, (1, 2, 0)), np.clip(ndvi, -1, 1)


def field_features(ndvi, crop_mask):
    """Edge density and mean segment area on cropland pixels. Pixels are 10 m."""
    sm = ndimage.gaussian_filter(ndvi, 1.0)
    gx, gy = ndimage.sobel(sm, 0), ndimage.sobel(sm, 1)
    grad = np.hypot(gx, gy)
    thr = 0.25
    edges = (grad > thr) & crop_mask
    edge_density = edges.sum() / max(crop_mask.sum(), 1)
    interior = (grad <= thr) & crop_mask
    lab, n = ndimage.label(interior)
    sizes = np.bincount(lab.ravel())[1:]
    sizes = sizes[sizes >= 25]  # ignore slivers under 0.25 ha
    mean_area_ha = float(sizes.mean() * 0.01) if len(sizes) else 0.0
    median_area_ha = float(np.median(sizes) * 0.01) if len(sizes) else 0.0
    return {"edge_density": float(edge_density), "mean_seg_ha": mean_area_ha, "median_seg_ha": median_area_ha,
            "n_segments": int(len(sizes)), "ndvi_std": float(ndvi[crop_mask].std()) if crop_mask.any() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=120)
    ap.add_argument("--n-placebo", type=int, default=80)
    ap.add_argument("--borders", default="cz_at,inner_de,pl_partition")
    ap.add_argument("--shift-km", type=float, default=35.0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--min-crop", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data/ghost_v2"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    wc = WorldCover()
    B = borders()
    to_ll = Transformer.from_crs(CRS_M, "EPSG:4326", always_xy=True)
    records = []
    with rasterio.Env(**ENV):
        for bname in a.borders.split(","):
            line, pA, pB, bbox = B[bname]
            band = band_box(bbox)
            sets = {"real": (line, pA, pB, a.n_real)}
            for sname, d in (("placebo_e", a.shift_km * 1000), ("placebo_w", -a.shift_km * 1000)):
                dx, dy = (0, d) if SHIFT_AXIS.get(bname) == "y" else (d, 0)
                sets[sname] = (affinity.translate(line, dx, dy), affinity.translate(pA, dx, dy), affinity.translate(pB, dx, dy), a.n_placebo)
            items = {k: find_items(bbox, v) for k, v in DATES.items()}
            print(f"{bname}: scenes d1={len(items['d1'])} d2={len(items['d2'])}", flush=True)
            for sname, (ln, A, Bp, n) in sets.items():
                for side, polyg in (("A", A), ("B", Bp)):
                    cands = candidates(polyg, ln, band, 3000, 15000)
                    rng.shuffle(cands)
                    kept = 0
                    for p in tqdm(cands, desc=f"{bname}/{sname}/{side}", leave=False):
                        if kept >= n:
                            break
                        lon, lat = to_ll.transform(p.x, p.y)
                        try:
                            crop, trees = wc.cropland_fraction(lon, lat, a.size)
                        except Exception:  # noqa: BLE001
                            continue
                        if crop < a.min_crop:
                            continue
                        rec = {"border": bname, "set": sname, "side": side, "lon": lon, "lat": lat,
                               "dist_km": ln.distance(p) / 1000, "crop_frac": crop, "tree_frac": trees}
                        ok = True
                        for dk in ("d1", "d2"):
                            tile = None
                            for it, gm in items[dk]:
                                if not gm.contains(Point(lon, lat)):
                                    continue
                                try:
                                    tile = read_tile(it, lon, lat, a.size)
                                except Exception:  # noqa: BLE001
                                    tile = None
                                if tile is not None:
                                    break
                            if tile is None:
                                ok = False
                                break
                            vis, ndvi = tile
                            name = f"{bname}_{sname}_{side}_{len(records):05d}"
                            Image.fromarray(vis).save(out / "tiles" / f"{name}_{dk}_rgb.png")
                            Image.fromarray(((ndvi + 1) * 127.5).astype(np.uint8)).save(out / "tiles" / f"{name}_{dk}_ndvi.png")
                            # cropland mask at tile resolution comes from WorldCover (same 10 m grid, approx aligned)
                            crop_mask = np.ones_like(ndvi, dtype=bool)
                            feats = field_features(ndvi, crop_mask)
                            rec.update({f"{dk}_{k}": v for k, v in feats.items()})
                            rec[f"{dk}_item"], rec[f"{dk}_date"] = it.id, it.properties["datetime"][:10]
                            rec["name"] = name
                        if not ok:
                            continue
                        records.append(rec)
                        kept += 1
                    print(f"  {bname}/{sname}/{side}: {kept} tiles from {len(cands)} candidates", flush=True)
                    pd.DataFrame(records).to_csv(out / "tiles.csv", index=False)
    df = pd.DataFrame(records)
    df.to_csv(out / "tiles.csv", index=False)
    print(df.groupby(["border", "set", "side"]).size())


if __name__ == "__main__":
    main()
