"""Ghost Borders spike, step 1: sample Sentinel-2 tiles on both sides of the former inner German border.

Border geometry comes from REGIS (CC BY 4.0): the GDR polygon 1949-1990 and the FRG polygon 1957-1990.
Their shared boundary is the inner German border. Tiles are sampled 3 to 15 km from the line on each
side, inside one latitude band, plus 2 placebo sets around the same line shifted east and west.
Imagery: Sentinel-2 L2A true-colour COGs (10 m) from the Earth Search STAC API, no account needed.

Usage: python spikes/ghost_borders_fetch.py --n-per-side 100 --lat-min 52.0 --lat-max 52.9
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
from shapely import affinity
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
GDR = ROOT / "dl/regis/GeoPackages/Sovereign_States/GDR/REGIS_GDR.gpkg"
GER = ROOT / "dl/regis/GeoPackages/Sovereign_States/Germany/REGIS_Germany.gpkg"
CRS_M = "EPSG:25832"  # UTM 32N, metres
STAC = "https://earth-search.aws.element84.com/v1"


def inner_border(lat_min, lat_max):
    gdr = gpd.read_file(GDR, layer="REGIS_GDR_L0_1949-1990").to_crs(CRS_M)
    frg = gpd.read_file(GER, layer="REGIS_Germany_L0_1957-1990").to_crs(CRS_M)
    g, f = unary_union(gdr.geometry), unary_union(frg.geometry)
    line = g.boundary.intersection(f.buffer(300))  # shared boundary, 300 m tolerance for digitising slack
    band = gpd.GeoSeries([box(9.0, lat_min, 15.0, lat_max)], crs="EPSG:4326").to_crs(CRS_M).iloc[0]
    return line.intersection(band), g, f, band


def sample_side(polygon, line, band, d_min, d_max, n, rng, spacing=2500):
    """Grid points inside polygon and band whose distance to line is in [d_min, d_max]; random n of them."""
    region = polygon.intersection(band).intersection(line.buffer(d_max)).difference(line.buffer(d_min))
    minx, miny, maxx, maxy = region.bounds
    pts = [Point(x, y) for x in np.arange(minx, maxx, spacing) for y in np.arange(miny, maxy, spacing)]
    pts = [p for p in pts if region.contains(p)]
    idx = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
    return [pts[i] for i in idx]


def build_points(n_per_side, lat_min, lat_max, shift_km, seed):
    rng = np.random.default_rng(seed)
    line, g, f, band = inner_border(lat_min, lat_max)
    sets = {"real": (line, g, f)}
    for name, dx in (("placebo_east", shift_km * 1000), ("placebo_west", -shift_km * 1000)):
        sets[name] = (affinity.translate(line, dx, 0), affinity.translate(g, dx, 0), affinity.translate(f, dx, 0))
    rows = []
    for set_name, (ln, east_poly, west_poly) in sets.items():
        for side, poly in (("east", east_poly), ("west", west_poly)):
            for p in sample_side(poly, ln, band, 3000, 15000, n_per_side, rng):
                rows.append({"set": set_name, "side": side, "x": p.x, "y": p.y, "dist_km": ln.distance(p) / 1000})
    df = gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy([r["x"] for r in rows], [r["y"] for r in rows]), crs=CRS_M)
    ll = df.to_crs("EPSG:4326")
    df["lon"], df["lat"] = ll.geometry.x, ll.geometry.y
    return df, line


def find_items(bounds_wgs84, date_range, max_cloud):
    cl = Client.open(STAC)
    search = cl.search(collections=["sentinel-2-l2a"], bbox=list(bounds_wgs84), datetime=date_range,
                       query={"eo:cloud_cover": {"lt": max_cloud}}, max_items=400)
    items = list(search.items())
    items.sort(key=lambda it: it.properties["eo:cloud_cover"])
    return items


def item_epsg(item):
    p = item.properties
    if p.get("proj:epsg"):
        return int(p["proj:epsg"])
    return int(str(p["proj:code"]).split(":")[-1])


def read_tile(href, epsg, lon, lat, size):
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = to_utm.transform(lon, lat)
    with rasterio.open(href) as ds:
        row, col = ds.index(x, y)
        r0, c0 = row - size // 2, col - size // 2
        if r0 < 0 or c0 < 0 or r0 + size > ds.height or c0 + size > ds.width:
            return None
        arr = ds.read([1, 2, 3], window=Window(c0, r0, size, size))
    return np.transpose(arr, (1, 2, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-side", type=int, default=100)
    ap.add_argument("--lat-min", type=float, default=52.0)
    ap.add_argument("--lat-max", type=float, default=52.9)
    ap.add_argument("--shift-km", type=float, default=35.0)
    ap.add_argument("--size", type=int, default=256, help="tile edge in pixels (10 m each)")
    ap.add_argument("--dates", default="2025-06-01/2025-09-15")
    ap.add_argument("--max-cloud", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data/ghost_borders"))
    a = ap.parse_args()

    out = Path(a.out)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    pts, line = build_points(a.n_per_side, a.lat_min, a.lat_max, a.shift_km, a.seed)
    print(pts.groupby(["set", "side"]).size())
    gpd.GeoSeries([line], crs=CRS_M).to_file(out / "border_line.gpkg", driver="GPKG")

    items = find_items(pts.to_crs("EPSG:4326").total_bounds, a.dates, a.max_cloud)
    print(f"{len(items)} Sentinel-2 items, cloud cover {items[0].properties['eo:cloud_cover']:.1f}"
          f" to {items[-1].properties['eo:cloud_cover']:.1f}")
    geoms = [shape(it.geometry) for it in items]

    records = []
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                       GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", VSI_CACHE="TRUE")
    with env:
        for i, r in tqdm(list(pts.iterrows()), desc="tiles"):
            p = Point(r.lon, r.lat)
            tile, used = None, None
            for it, gm in zip(items, geoms):
                if not gm.contains(p):
                    continue
                try:
                    tile = read_tile(it.assets["visual"].href, item_epsg(it), r.lon, r.lat, a.size)
                except Exception:  # noqa: BLE001
                    tile = None
                if tile is not None and tile.mean() > 5:  # skip no-data windows
                    used = it
                    break
                tile = None
            if tile is None:
                continue
            name = f"{r['set']}_{r['side']}_{i:04d}.png"
            Image.fromarray(tile).save(out / "tiles" / name)
            records.append({**r.drop("geometry").to_dict(), "file": f"tiles/{name}", "item": used.id,
                            "date": used.properties["datetime"][:10], "cloud": used.properties["eo:cloud_cover"]})
    df = pd.DataFrame(records)
    df.to_csv(out / "tiles.csv", index=False)
    print(df.groupby(["set", "side"]).size())
    print("saved", len(df), "tiles to", out)


if __name__ == "__main__":
    main()
