import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyproj
import rasterio
import requests
import shapely
import geopandas as gpd
from rasterio.features import rasterize

API = "https://api.ohsome.org/v1/contributions/latest/geometry"
START = "2007-10-08"
VALUES = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link",
          "tertiary", "tertiary_link", "residential", "unclassified", "living_street", "service"]
FILTER = "type:way and highway in (" + ",".join(VALUES) + ")"


def tiles(poly, deg):
    x0, y0, x1, y1 = poly.bounds
    out = []
    for x in np.arange(np.floor(x0 / deg) * deg, x1, deg):
        for y in np.arange(np.floor(y0 / deg) * deg, y1, deg):
            b = shapely.box(x, y, x + deg, y + deg)
            if poly.intersects(b):
                out.append((round(float(x), 4), round(float(y), 4), deg))
    return out


def request(geom, date):
    fc = dict(type="FeatureCollection", features=[dict(type="Feature", properties=dict(id="t"), geometry=shapely.geometry.mapping(geom))])
    r = requests.post(API, data=dict(bpolys=json.dumps(fc), time=f"{START},{date}", filter=FILTER, properties="tags"), timeout=700)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return [f["geometry"] for f in r.json()["features"] if f.get("geometry") and not f["properties"].get("@deletion")]


def fetch(poly, tile, date, depth=0):
    x, y, d = tile
    geom = shapely.intersection(poly, shapely.box(x, y, x + d, y + d))
    if geom.is_empty or geom.area == 0:
        return []
    for k in range(3):
        try:
            return request(geom, date)
        except Exception as e:
            err = str(e)[:60]
            time.sleep(5 * (k + 1))
    if depth < 3:
        h = d / 2
        out = []
        for sx, sy in ((x, y), (x + h, y), (x, y + h), (x + h, y + h)):
            out += fetch(poly, (sx, sy, h), date, depth + 1)
        return out
    raise RuntimeError(f"tile {tile} failed: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso3", required=True)
    ap.add_argument("--countries", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile_deg", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    ne = gpd.read_file(a.countries)
    col = "ADM0_A3" if "ADM0_A3" in ne.columns else "ISO_A3"
    poly = shapely.make_valid(ne[ne[col] == a.iso3].to_crs("EPSG:4326").geometry.union_all()).buffer(0.05).simplify(0.01)
    with rasterio.open(a.grid) as g:
        prof = g.profile.copy()
        T, H, W, crs = g.transform, g.height, g.width, g.crs
    to = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ck = a.out + ".ckpt.npz"
    done = set()
    grid = np.zeros((H, W), np.uint8)
    km = 0.0
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        grid = np.unpackbits(z["grid"])[: H * W].reshape(H, W)
        done = set(tuple(t) for t in z["done"].tolist())
        km = float(z["km"])
    todo = [t for t in tiles(poly, a.tile_deg) if t not in done]
    print(a.iso3, a.date, "tiles to do", len(todo), "done", len(done), flush=True)

    def work(t):
        feats = fetch(poly, t, a.date)
        geoms = [shapely.geometry.shape(f) for f in feats]
        geoms = [g for g in geoms if g.geom_type in ("LineString", "MultiLineString")]
        if not geoms:
            return t, [], 0.0
        G = shapely.transform(np.asarray(geoms, dtype=object), lambda xy: np.column_stack(to.transform(xy[:, 0], xy[:, 1])))
        return t, list(G), float(shapely.length(G).sum() / 1000.0)

    n = 0
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for t, G, k in ex.map(work, todo):
            if G:
                r = rasterize(((g, 1) for g in G), out_shape=(H, W), transform=T, fill=0, dtype="uint8", all_touched=False)
                grid |= r
            km += k
            done.add(t)
            n += 1
            if n % 25 == 0 or n == len(todo):
                np.savez(ck + ".tmp.npz", grid=np.packbits(grid.ravel()), done=np.array(sorted(done)), km=km)
                os.replace(ck + ".tmp.npz", ck)
                print(n, "of", len(todo), "tiles", int(grid.sum()), "cells", round(km), "km", round(time.time() - t0), "s", flush=True)
    prof.update(dtype="uint8", count=1, nodata=None, compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(a.out, "w", **prof) as w:
        w.write(grid, 1)
        w.update_tags(km=str(round(km, 1)), date=a.date, source="OpenStreetMap via ohsome API", filter=FILTER)
    print("DONE", a.out, int(grid.sum()), "cells", round(km), "km", flush=True)


if __name__ == "__main__":
    main()
