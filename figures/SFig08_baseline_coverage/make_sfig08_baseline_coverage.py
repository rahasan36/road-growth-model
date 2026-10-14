import argparse
import glob
import gc
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DATA = os.path.join(REPO, "data", "figures")

def read_vector(path, **kw):
    if os.path.exists(path):
        return gpd.read_file(path, **kw)
    if os.path.exists(path + ".zip"):
        return gpd.read_file(path + ".zip", **kw)
    stem = path[:-4] if path.endswith(".shp") else path
    parts = sorted(glob.glob(glob.escape(stem) + ".part*.shp.zip"),
                   key=lambda p: int(re.search(r"\.part(\d+)\.shp\.zip$", p).group(1)))
    if not parts:
        raise FileNotFoundError(path)
    gs = [gpd.read_file(p, **kw) for p in parts]
    return gpd.GeoDataFrame(pd.concat(gs, ignore_index=True), geometry="geometry", crs=gs[0].crs)

MM = 1 / 25.4
PT = 7.0
C_ROAD = "#0072b2"
C_EDGE = "#7f7f7f"

OSM_CLASSES = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
               "secondary", "secondary_link", "tertiary", "tertiary_link",
               "residential", "unclassified", "living_street", "service"]

_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Arial", "Liberation Sans", "DejaVu Sans"] if f in _avail), "DejaVu Sans")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": PT,
    "xtick.labelsize": PT, "ytick.labelsize": PT,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "axes.edgecolor": "black", "text.color": "black",
    "xtick.color": "black", "ytick.color": "black",
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})


DST = "ESRI:102953"


def upright(arrays, T, crs, bounds, res=300.0):
    l, b, r, t = bounds
    w, h = int(np.ceil((r - l) / res)), int(np.ceil((t - b) / res))
    fine_T = rasterio.transform.from_origin(l, t, res / 2, res / 2)
    out = []
    for a in arrays:
        fine = np.zeros((2 * h, 2 * w), np.uint8)
        reproject(np.asarray(a, np.uint8), fine, src_transform=T, src_crs=crs, dst_transform=fine_T, dst_crs=DST,
                  resampling=Resampling.nearest)
        blk = fine.reshape(h, 2, w, 2).transpose(0, 2, 1, 3).reshape(h, w, 4)
        n = max(int(blk.max()), 1)
        cnt = np.stack([(blk == v).sum(-1) for v in range(1, n + 1)], -1)
        out.append(np.where(cnt.max(-1) > 0, cnt.argmax(-1) + 1, 0).astype(np.uint8))
    return out, rasterio.transform.from_origin(l, t, res, res), (h, w)


def map_gprtp(v, n):
    if n == 5:
        return {1: 1, 2: 1, 3: 2, 4: 3, 5: 4}.get(v, 0)
    return v


def burn(geoms, shape, transform, all_touched=False):
    return rasterize([(g, 1) for g in geoms if g is not None and not g.is_empty],
                     out_shape=shape, transform=transform, fill=0, dtype=np.uint8,
                     all_touched=all_touched)


def edge_ticks(ax, transform_xy, r0, r1, c0, c1, tf_inv, lons, lats):
    cols = np.linspace(c0, c1, 400)
    xs, ys = transform_xy(np.full_like(cols, r1), cols)
    lon_b, _ = tf_inv.transform(xs, ys)
    xt, xl = [], []
    for L in lons:
        if lon_b.min() < L < lon_b.max():
            xt.append(np.interp(L, lon_b, cols) - c0)
            xl.append(f"{L:g}\u00b0E")
    rows = np.linspace(r0, r1, 400)
    xs, ys = transform_xy(rows, np.full_like(rows, c0))
    _, lat_l = tf_inv.transform(xs, ys)
    order = np.argsort(lat_l)
    yt, yl = [], []
    for P in lats:
        if lat_l.min() < P < lat_l.max():
            yt.append(np.interp(P, lat_l[order], rows[order]) - r0)
            yl.append(f"{P:g}\u00b0N")
    ax.set_xticks(xt, xl)
    ax.set_yticks(yt, yl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roads_dir", default=os.path.join(FIG_DATA, "roads"))
    ap.add_argument("--osm2015", default=os.path.join(FIG_DATA, "osm", "bangladesh-150101-free.shp"))
    ap.add_argument("--osm2026", default=os.path.join(FIG_DATA, "osm", "bangladesh-260606-free.shp"))
    ap.add_argument("--coast", default=os.path.join(FIG_DATA, "naturalearth", "ne_10m_admin_0_countries_BGD_NGA.shp"),
                    help="Natural Earth 1:10m countries; used only to trim the drawn outline to the coast")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with rasterio.open(os.path.join(a.roads_dir, "BGD_all_final_roads_2030_ssp1.tif")) as src:
        T, crs, shape = src.transform, src.crs, (src.height, src.width)

    dist = read_vector(os.path.join(a.roads_dir, "bgd_dist.shp")).to_crs(crs)
    land = dist.geometry.union_all()
    mask = burn([land], shape, T, all_touched=True)
    print(f"country mask: {int(mask.sum()):,} pixels")

    g = read_vector(os.path.join(a.roads_dir, "bgd_roads_2015.shp"))
    g = g.to_crs(crs) if g.crs != crs else g
    n_u = len(g["GP_RTP"].unique())
    keep = [geom for geom, v in zip(g.geometry, g["GP_RTP"]) if map_gprtp(int(v), n_u) > 0]
    grip_all = burn(keep, shape, T)
    grip = grip_all * mask
    del g, keep; gc.collect()

    layers = {"GRIP 2015": grip}
    whole = {"GRIP 2015": int(grip_all.sum())}
    for lab, d in [("OSM 2015", a.osm2015), ("OSM 2026", a.osm2026)]:
        o = read_vector(os.path.join(d, "gis_osm_roads_free_1.shp"))
        o = o[o["fclass"].isin(OSM_CLASSES)]
        o = o.to_crs(crs) if o.crs != crs else o
        full = burn(o.geometry, shape, T)
        whole[lab] = int(full.sum())
        layers[lab] = full * mask
        del o; gc.collect()

    counts = {k: int((v > 0).sum()) for k, v in layers.items()}
    for k, v in counts.items():
        print(f"{k}: {v:,} road pixels inside the boundary, {whole[k]:,} in the whole extract")
    pd.DataFrame([dict(layer=k, road_pixels_inside_boundary=v, road_pixels_whole_extract=whole[k])
                  for k, v in counts.items()]).to_csv(
        os.path.join(a.out, "SFig08_baseline_coverage_counts.csv"), index=False)

    land = dist.to_crs(DST).geometry.union_all()
    l, b, r, t = land.bounds
    up, T, shape = upright(list(layers.values()), T, crs, (l - 3000, b - 3000, r + 3000, t + 3000))
    layers = dict(zip(layers, up))
    crs = DST
    mask = burn([land], shape, T, all_touched=True)
    rr, cc = np.nonzero(mask)
    pad = 8
    r0, r1 = max(0, rr.min() - pad), min(shape[0], rr.max() + pad)
    c0, c1 = max(0, cc.min() - pad), min(shape[1], cc.max() + pad)
    h, w = r1 - r0, c1 - c0

    def rc_to_xy(rows, cols):
        xs = T.c + (np.asarray(cols) + 0.5) * T.a
        ys = T.f + (np.asarray(rows) + 0.5) * T.e
        return xs, ys

    tf_inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    outline = land
    if a.coast and os.path.exists(a.coast):
        ne = read_vector(a.coast)
        bgd = ne[ne["ADM0_A3"] == "BGD"].to_crs(crs).geometry.union_all()
        outline = land.intersection(bgd.buffer(1000))
        print("outline trimmed to the Natural Earth coastline")

    W = 180.0
    left, right, gap = 11.0, 1.0, 3.0
    top, bottom = 8.0, 6.0
    mw = (W - left - right - 2 * gap) / 3
    mh = mw * h / w
    H = top + mh + bottom
    fig = plt.figure(figsize=(W * MM, H * MM))
    print(f"figure {W:.0f} x {H:.0f} mm")
    cmap = ListedColormap(["white", C_ROAD])
    inv_a = ~T
    for k, (lab, arr) in enumerate(layers.items()):
        ax = fig.add_axes([(left + k * (mw + gap)) / W, bottom / H, mw / W, mh / H])
        ax.imshow(arr[r0:r1, c0:c1] > 0, cmap=cmap, vmin=0, vmax=1,
                  interpolation="antialiased", interpolation_stage="rgba", zorder=1)
        parts = [g for g in getattr(outline, "geoms", [outline]) if g.geom_type == "Polygon"]
        parts += [q for g in getattr(outline, "geoms", []) if g.geom_type == "MultiPolygon" for q in g.geoms]
        for p in parts:
            x, y = p.exterior.xy
            cols, rows = inv_a * (np.asarray(x), np.asarray(y))
            ax.plot(cols - c0 - 0.5, rows - r0 - 0.5, color=C_EDGE, lw=0.3, zorder=2)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        edge_ticks(ax, rc_to_xy, r0, r1, c0, c1, tf_inv,
                   lons=[89, 90, 91, 92], lats=[21, 22, 23, 24, 25, 26])
        if k > 0:
            ax.tick_params(labelleft=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(0, 1.012, "abc"[k], transform=ax.transAxes, ha="left", va="bottom",
                fontsize=PT, fontweight="bold")
        ax.text(1, 1.012, lab, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=PT, fontweight="bold")
        if k == 0:
            px = 100000 / T.a
            ax.add_patch(Rectangle((0.04 * w, 0.955 * h), px, 0.008 * h, color="black", lw=0, zorder=3))
            ax.text(0.04 * w + px / 2, 0.945 * h, "100 km", ha="center", va="bottom", fontsize=PT)

    base = os.path.join(a.out, "SFig08_baseline_coverage")
    fig.savefig(base + ".png", dpi=600)
    fig.savefig(base + ".pdf", dpi=600)
    fig.savefig(base + ".svg", dpi=600)
    if FONT != "Arial":
        with open(base + ".svg", encoding="utf-8") as f:
            svg = f.read()
        svg = svg.replace(f"font-family: '{FONT}', sans-serif", f"font-family: Arial, '{FONT}', sans-serif")
        svg = svg.replace(f"font-family: '{FONT}'", f"font-family: Arial, '{FONT}', sans-serif")
        with open(base + ".svg", "w", encoding="utf-8") as f:
            f.write(svg)
    plt.close(fig)
    print("Wrote", os.path.abspath(base) + ".{png,pdf,svg}")


if __name__ == "__main__":
    main()
