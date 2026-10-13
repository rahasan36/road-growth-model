import argparse
import glob
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer
from shapely.geometry import Point, box

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

C_EXIST, C_NEW, C_OBS = "#bdbdbd", "#d55e00", "#000000"
C_TP, C_FP, C_FN = "#0072b2", "#d55e00", "#000000"
C_MWY = "#cc79a7"

CENTRE = (90.26, 23.46)
RADIUS_KM = 20
MAWA, JAJIRA = (90.260, 23.464), (90.264, 23.424)
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
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]


def degree_ticks(ax, tf, step=0.1, label_y=True):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = np.linspace(x0, x1, 300)
    lon, _ = tf.transform(xs, np.full_like(xs, y0))
    ys = np.linspace(y0, y1, 300)
    _, lat = tf.transform(np.full_like(ys, x0), ys)

    def vals(v):
        out = np.arange(np.ceil(v.min() / step) * step, v.max(), step)
        return out[(out > v.min() + 0.04 * np.ptp(v)) & (out < v.max() - 0.04 * np.ptp(v))]

    lv, pv = vals(lon), vals(lat)
    o = np.argsort(lon)
    ax.set_xticks(np.interp(lv, lon[o], xs[o]), [f"{v:.1f}\u00b0E" for v in lv])
    o = np.argsort(lat)
    ax.set_yticks(np.interp(pv, lat[o], ys[o]), [f"{v:.1f}\u00b0N" for v in pv] if label_y else [])


def lines(ax, gdf, **kw):
    segs = []
    for g in gdf.geometry:
        for p in (g.geoms if g.geom_type.startswith("Multi") else [g]):
            segs.append(np.asarray(p.coords)[:, :2])
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, **kw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roads_dir", default=os.path.join(FIG_DATA, "roads"))
    ap.add_argument("--tpfpfn", default=os.path.join(FIG_DATA, "validation", "BGD_spatial_TPFPFN_300m_20260607.tif"))
    ap.add_argument("--osm2026", default=os.path.join(FIG_DATA, "osm", "bangladesh-260606-free.shp"))
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with rasterio.open(os.path.join(a.roads_dir, "BGD_all_final_roads_2030_ssp1.tif")) as src:
        model, T, crs, (H0, W0) = src.read(1), src.transform, src.crs, src.shape
    with rasterio.open(a.tpfpfn) as src:
        cls = src.read(1)
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    bx, by = to_proj.transform(*CENTRE)
    c, r = ~T * (bx, by)
    c, r = int(c), int(r)
    R = int(RADIUS_KM * 1000 / abs(T.a))
    r0, r1, c0, c1 = max(0, r - R), min(H0, r + R), max(0, c - R), min(W0, c + R)
    ext = (T.c + c0 * T.a, T.c + c1 * T.a, T.f + r1 * T.e, T.f + r0 * T.e)
    mc, tc = model[r0:r1, c0:c1], cls[r0:r1, c0:c1]
    existing = (mc >= 1) & (mc <= 10)
    new = (mc >= 11) & (mc <= 14)
    tp, fp, fn = (int((tc == k).sum()) for k in (1, 2, 3))
    print(f"existing {existing.sum():,}  new {new.sum():,}  TP {tp:,}  commission {fp:,}  omission {fn:,}")
    pd.DataFrame([dict(existing_pixels=int(existing.sum()), new_pixels=int(new.sum()),
                       true_positive=tp, commission=fp, omission=fn, buffer_m=300)]).to_csv(
        os.path.join(a.out, "SFig12_padma_counts.csv"), index=False)

    lon0, lat1 = to_geo.transform(ext[0], ext[3])
    lon1, lat0 = to_geo.transform(ext[1], ext[2])
    osm = read_vector(os.path.join(a.osm2026, "gis_osm_roads_free_1.shp"),
                        bbox=box(lon0 - 0.05, lat0 - 0.05, lon1 + 0.05, lat1 + 0.05))
    osm = osm[osm["fclass"].isin(OSM_CLASSES)].to_crs(crs)
    mwy = osm[osm["fclass"].isin(["motorway", "motorway_link"])]
    mwy = mwy[mwy.geometry.distance(Point(bx, by)) < 6000]
    other = osm[~osm["fclass"].isin(["motorway", "motorway_link"])]
    ends = [to_proj.transform(*MAWA), to_proj.transform(*JAJIRA)]

    img_a = np.zeros(mc.shape, dtype=np.uint8)
    img_a[existing] = 1
    img_a[new] = 2

    W = 180.0
    left, gap, right = 14.0, 5.0, 1.0
    mw = (W - left - right - gap) / 2
    top, tick_h, leg_h = 6.0, 7.0, 15.0
    H = top + mw + tick_h + leg_h
    fig = plt.figure(figsize=(W * MM, H * MM))
    print(f"figure {W:.0f} x {H:.0f} mm")

    specs = [
        ("a", "Forecast 2030 (SSP1)", img_a, ListedColormap(["white", C_EXIST, C_NEW]), 2, True),
        ("b", "Agreement with observed roads", tc, ListedColormap(["white", C_TP, C_FP, C_FN]), 3, False),
    ]
    for k, (letter, title, img, cmap, vmax, show_obs) in enumerate(specs):
        ax = fig.add_axes([(left + k * (mw + gap)) / W, (tick_h + leg_h) / H, mw / W, mw / H])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", extent=ext, zorder=1)
        if show_obs and len(other):
            lines(ax, other, colors=C_OBS, linewidths=0.25, zorder=2)
        if len(mwy):
            lines(ax, mwy, colors=C_MWY, linewidths=1.8, zorder=4, path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
        for (ex_, ey_), lab, dy, va in [(ends[0], "Mawa end", 6, "bottom"), (ends[1], "Jajira end", -6, "top")]:
            ax.plot(ex_, ey_, "s", ms=4.5, mfc="black", mec="white", mew=0.8, zorder=6)
            ax.annotate(lab, (ex_, ey_), xytext=(7, dy), textcoords="offset points", ha="left", va=va,
                        fontsize=PT, fontweight="bold", zorder=7, path_effects=HALO)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        degree_ticks(ax, to_geo, label_y=(k == 0))
        ax.text(0, 1.015, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=PT, fontweight="bold")
        ax.text(1, 1.015, title, transform=ax.transAxes, ha="right", va="bottom", fontsize=PT, fontweight="bold")
        span = ext[3] - ext[2]
        L = 10000
        sx, sy = ext[1] - 0.04 * span - L, ext[2] + 0.05 * span
        ax.add_patch(Rectangle((sx - 0.02 * span, sy - 0.025 * span), L + 0.04 * span, 0.10 * span,
                               facecolor="white", edgecolor="none", alpha=0.9, zorder=8))
        ax.add_patch(Rectangle((sx, sy), L, 0.010 * span, color="black", lw=0, zorder=9))
        ax.text(sx + L / 2, sy + 0.022 * span, "10 km", ha="center", va="bottom", fontsize=PT, zorder=9)

        mwy_h = Line2D([0], [0], color=C_MWY, lw=1.8, label="Padma Bridge motorway",
                       path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
        end_h = Line2D([0], [0], ls="none", marker="s", ms=4.5, mfc="black", mec="white", mew=0.8,
                       label="Bridge ends")
        if k == 0:
            handles = [Patch(facecolor=C_EXIST, label="Existing roads (2015)"),
                       Patch(facecolor=C_NEW, label="Predicted new roads"),
                       Line2D([0], [0], color=C_OBS, lw=0.6, label="Observed roads (OSM 2026)"),
                       mwy_h, end_h]
            ncol = 2
        else:
            handles = [Patch(facecolor=C_TP, label=f"True positive ({tp:,})"),
                       Patch(facecolor=C_FP, label=f"Commission ({fp:,})"),
                       Patch(facecolor=C_FN, label=f"Omission ({fn:,})"),
                       mwy_h, end_h]
            ncol = 2
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.01, -0.085), frameon=False, ncol=ncol,
                  handlelength=1.2, handleheight=0.8, handletextpad=0.4, labelspacing=0.25,
                  columnspacing=1.2, borderaxespad=0.0)

    base = os.path.join(a.out, "SFig12_padma")
    for e in ("png", "pdf", "svg"):
        fig.savefig(f"{base}.{e}", dpi=600)
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
