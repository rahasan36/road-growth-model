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
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

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
PT = 8.0
C_TP, C_FP, C_FN = "#0072b2", "#d55e00", "#000000"
C_EDGE = "#7f7f7f"
BUFFER_M = 300

OSM_CLASSES = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
               "secondary", "secondary_link", "tertiary", "tertiary_link",
               "residential", "unclassified", "living_street", "service"]

_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Arial", "Liberation Sans", "DejaVu Sans"] if f in _avail), "DejaVu Sans")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": PT,
    "xtick.labelsize": PT, "ytick.labelsize": PT, "legend.fontsize": PT,
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


def classify(pred, obs, mask, s):
    pb, ob = pred.astype(bool), obs.astype(bool)
    obuf, pbuf = binary_dilation(ob, structure=s), binary_dilation(pb, structure=s)
    out = np.zeros(pred.shape, dtype=np.uint8)
    out[pb & obuf] = 1
    out[pb & ~obuf] = 2
    out[ob & ~pbuf] = 3
    out *= mask
    return out, int((out == 1).sum()), int((out == 2).sum()), int((out == 3).sum())


def edge_ticks(ax, T, r0, r1, c0, c1, tf, lons, lats):
    cols = np.linspace(c0, c1, 400)
    lon, _ = tf.transform(T.c + (cols + 0.5) * T.a, np.full_like(cols, T.f + (r1 + 0.5) * T.e))
    xt = [(np.interp(L, lon, cols) - c0, f"{L:g}\u00b0E") for L in lons if lon.min() < L < lon.max()]
    rows = np.linspace(r0, r1, 400)
    _, lat = tf.transform(np.full_like(rows, T.c + (c0 + 0.5) * T.a), T.f + (rows + 0.5) * T.e)
    o = np.argsort(lat)
    yt = [(np.interp(P, lat[o], rows[o]) - r0, f"{P:g}\u00b0N") for P in lats if lat.min() < P < lat.max()]
    ax.set_xticks([t for t, _ in xt], [l for _, l in xt])
    ax.set_yticks([t for t, _ in yt], [l for _, l in yt])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roads_dir", default=os.path.join(FIG_DATA, "roads"))
    ap.add_argument("--osm15_model", default=os.path.join(FIG_DATA, "BGD_forecast_osm15_baseline", "BGD_all_final_roads.tif"))
    ap.add_argument("--osm2015", default=os.path.join(FIG_DATA, "osm", "bangladesh-150101-free.shp"))
    ap.add_argument("--osm2026", default=os.path.join(FIG_DATA, "osm", "bangladesh-260606-free.shp"))
    ap.add_argument("--coast", default=os.path.join(FIG_DATA, "naturalearth", "ne_10m_admin_0_countries_BGD_NGA.shp"))
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ref = os.path.join(a.roads_dir, "BGD_all_final_roads_2030_ssp1.tif")
    with rasterio.open(ref) as src:
        T, crs, shape, px = src.transform, src.crs, (src.height, src.width), src.res[0]
        fc = src.read(1)

    adm = read_vector(os.path.join(a.osm2026, "gis_osm_adminareas_a_free_1.shp"))
    mask = burn(adm[adm.fclass == "national"].to_crs(crs).geometry, shape, T, all_touched=True)
    del adm

    def osm(d):
        o = read_vector(os.path.join(d, "gis_osm_roads_free_1.shp"))
        o = o[o["fclass"].isin(OSM_CLASSES)]
        return burn((o.to_crs(crs) if o.crs != crs else o).geometry, shape, T) * mask

    osm15, osm26 = osm(a.osm2015), osm(a.osm2026)
    g = read_vector(os.path.join(a.roads_dir, "bgd_roads_2015.shp"))
    g = g.to_crs(crs) if g.crs != crs else g
    n_u = len(g["GP_RTP"].unique())
    grip = burn([geom for geom, v in zip(g.geometry, g["GP_RTP"]) if map_gprtp(int(v), n_u) > 0],
                shape, T) * mask
    del g; gc.collect()

    grip_fc = (fc > 0).astype(np.uint8) * mask
    with rasterio.open(a.osm15_model) as src:
        m15 = np.zeros(shape, dtype=np.float32)
        reproject(source=src.read(1), destination=m15, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=T, dst_crs=crs, resampling=Resampling.nearest)
    osm_fc = (m15 != 0).astype(np.uint8) * mask

    s = disk(max(1, round(BUFFER_M / px)))
    panels = [
        ("a", "GRIP 2015 baseline", *classify(grip_fc, ((osm26 > 0) | (grip > 0)).astype(np.uint8), mask, s),
         int(grip_fc.sum())),
        ("b", "OSM 2015 baseline", *classify(osm_fc, ((osm26 > 0) | (osm15 > 0)).astype(np.uint8), mask, s),
         int(osm_fc.sum())),
    ]
    rows = []
    for letter, title, _, tp, fp, fn, npred in panels:
        print(f"{title}: forecast {npred:,}  TP {tp:,}  commission {fp:,}  omission {fn:,}")
        rows.append(dict(panel=letter, baseline=title, forecast_pixels=npred, true_positive=tp,
                         commission=fp, omission=fn, buffer_m=BUFFER_M))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "SFig10_allroads_spatial_counts.csv"), index=False)

    land = read_vector(os.path.join(a.roads_dir, "bgd_dist.shp")).to_crs(DST).geometry.union_all()
    l, b, r, t = land.bounds
    up, T, shape = upright([p_[2] for p_ in panels], T, crs, (l - 3000, b - 3000, r + 3000, t + 3000))
    panels = [p_[:2] + (u,) + p_[3:] for p_, u in zip(panels, up)]
    crs = DST
    outline = land
    if a.coast and os.path.exists(a.coast):
        ne = read_vector(a.coast)
        outline = land.intersection(ne[ne["ADM0_A3"] == "BGD"].to_crs(crs).geometry.union_all().buffer(1000))
    polys = []
    for gm in getattr(outline, "geoms", [outline]):
        if gm.geom_type == "Polygon":
            polys.append(gm)
        elif gm.geom_type == "MultiPolygon":
            polys.extend(gm.geoms)

    rr, cc = np.nonzero(burn([land], shape, T, all_touched=True))
    pad = 8
    r0, r1 = max(0, rr.min() - pad), min(shape[0], rr.max() + pad)
    c0, c1 = max(0, cc.min() - pad), min(shape[1], cc.max() + pad)
    h, w = r1 - r0, c1 - c0
    tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    inv = ~T

    W = 180.0
    left, right, gap, top, bottom = 12.0, 1.0, 4.0, 7.0, 6.5
    mw = (W - left - right - gap) / 2
    mh = mw * h / w
    H = top + mh + bottom
    fig = plt.figure(figsize=(W * MM, H * MM))
    print(f"figure {W:.0f} x {H:.0f} mm")
    cmap = ListedColormap(["white", C_TP, C_FP, C_FN])
    for k, (letter, title, cls, tp, fp, fn, _) in enumerate(panels):
        ax = fig.add_axes([(left + k * (mw + gap)) / W, bottom / H, mw / W, mh / H])
        ax.imshow(cls[r0:r1, c0:c1], cmap=cmap, vmin=0, vmax=3, interpolation="nearest", zorder=1)
        for p in polys:
            x, y = p.exterior.xy
            cx, ry = inv * (np.asarray(x), np.asarray(y))
            ax.plot(cx - c0 - 0.5, ry - r0 - 0.5, color=C_EDGE, lw=0.35, zorder=2)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        edge_ticks(ax, T, r0, r1, c0, c1, tf, [89, 90, 91, 92], [21, 22, 23, 24, 25, 26])
        if k > 0:
            ax.tick_params(labelleft=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(0, 1.01, letter, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=PT, fontweight="bold")
        ax.text(1, 1.01, title, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=PT, fontweight="bold")
        handles = [Patch(facecolor=C_TP, label=f"True positive ({tp:,})"),
                   Patch(facecolor=C_FP, label=f"Commission ({fp:,})"),
                   Patch(facecolor=C_FN, label=f"Omission ({fn:,})")]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False,
                  handlelength=1.0, handleheight=0.8, handletextpad=0.4, labelspacing=0.25,
                  borderaxespad=0.3)
        if k == 0:
            L = 100000 / T.a
            sx, sy = 0.97 * w - L, 0.06 * h
            ax.add_patch(Rectangle((sx, sy), L, 0.007 * h, color="black", lw=0, zorder=3))
            ax.text(sx + L / 2, sy - 0.008 * h, "100 km", ha="center", va="bottom", fontsize=PT)

    base = os.path.join(a.out, "SFig10_allroads_spatial")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{base}.{ext}", dpi=600)
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
