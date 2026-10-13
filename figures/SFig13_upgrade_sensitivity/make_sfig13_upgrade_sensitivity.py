import argparse
import os

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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DATA = os.path.join(REPO, "data", "figures")
MM = 1 / 25.4
PT = 8.0
DISPLAY_M = 2000
RES = [("100m", "100 m grid"), ("300m", "300 m grid"), ("1000m", "1,000 m grid")]
CLASSES = [(1, "Primary", "#000000"), (2, "Secondary", "#d55e00"),
           (3, "Tertiary", "#0072b2"), (4, "Local", "#e69f00")]
C_EDGE = "#7f7f7f"

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


def to_display(arr, T, src_crs, dst_crs, ext):
    left, right, bottom, top = ext
    nx = int(np.ceil((right - left) / DISPLAY_M))
    ny = int(np.ceil((top - bottom) / DISPLAY_M))
    r, c = np.nonzero(arr)
    x = T.c + (c + 0.5) * T.a
    y = T.f + (r + 0.5) * T.e
    x, y = Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform(x, y)
    gi = ((top - y) // DISPLAY_M).astype(int)
    gj = ((x - left) // DISPLAY_M).astype(int)
    ok = (gi >= 0) & (gi < ny) & (gj >= 0) & (gj < nx)
    counts = np.zeros((4, ny, nx), dtype=np.int32)
    np.add.at(counts, (arr[r, c][ok].astype(int) - 1, gi[ok], gj[ok]), 1)
    out = np.where(counts.sum(0) > 0, counts.argmax(0) + 1, 0)
    return out, (left, left + nx * DISPLAY_M, top - ny * DISPLAY_M, top)


def degree_ticks(ax, tf, step=1.0, label_y=True):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = np.linspace(x0, x1, 400)
    lon, _ = tf.transform(xs, np.full_like(xs, y0))
    ys = np.linspace(y0, y1, 400)
    _, lat = tf.transform(np.full_like(ys, x0), ys)

    def vals(v):
        out = np.arange(np.ceil(v.min() / step) * step, v.max(), step)
        return out[(out > v.min() + 0.03 * np.ptp(v)) & (out < v.max() - 0.03 * np.ptp(v))]

    lv, pv = vals(lon), vals(lat)
    o = np.argsort(lon)
    ax.set_xticks(np.interp(lv, lon[o], xs[o]), [f"{v:g}\u00b0E" for v in lv])
    o = np.argsort(lat)
    ax.set_yticks(np.interp(pv, lat[o], ys[o]), [f"{v:g}\u00b0N" for v in pv] if label_y else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upgrade_dir", default=os.path.join(FIG_DATA, "sensitivity", "upgrade_locations"))
    ap.add_argument("--roads_dir", default=os.path.join(FIG_DATA, "roads"))
    ap.add_argument("--coast", default=os.path.join(FIG_DATA, "naturalearth", "ne_10m_admin_0_countries_BGD_NGA.shp"))
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    DST = "ESRI:102953"
    rasters, rows = [], []
    for key, lab in RES:
        with rasterio.open(os.path.join(a.upgrade_dir, f"BGD_upgrades_{key}_2030_SSP5_sc1.tif")) as src:
            arr, T, crs, bounds, res_m = src.read(1), src.transform, src.crs, tuple(src.bounds), src.res
        n = {k: int((arr == k).sum()) for k, _, _ in CLASSES}
        km = sum(n.values()) * int(key[:-1]) / 1000
        rasters.append((lab, arr, T, km))
        rows.append(dict(grid=lab, pixels_selected=sum(n.values()), upgraded_km=round(km, 1),
                         **{f"{name.lower()}_pixels": n[k] for k, name, _ in CLASSES}))
        print(f"{lab}: {sum(n.values()):,} pixels selected ({km:,.0f} km)")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "SFig13_upgrade_sensitivity_counts.csv"), index=False)

    land = gpd.read_file(os.path.join(a.roads_dir, "bgd_dist.shp")).to_crs(DST).geometry.union_all()
    if a.coast and os.path.exists(a.coast):
        ne = gpd.read_file(a.coast)
        land = land.intersection(ne[ne["ADM0_A3"] == "BGD"].to_crs(DST).geometry.union_all().buffer(1000))
    polys = []
    for g in getattr(land, "geoms", [land]):
        polys += [g] if g.geom_type == "Polygon" else list(getattr(g, "geoms", []))

    l, b, r, t = bounds
    ring = np.vstack([np.column_stack([np.linspace(l, r, 50), np.full(50, b)]),
                      np.column_stack([np.full(50, r), np.linspace(b, t, 50)]),
                      np.column_stack([np.linspace(r, l, 50), np.full(50, t)]),
                      np.column_stack([np.full(50, l), np.linspace(t, b, 50)])])
    rx, ry = Transformer.from_crs(crs, DST, always_xy=True).transform(ring[:, 0], ring[:, 1])
    lx0, ly0, lx1, ly1 = land.bounds
    x0, x1, y0, y1 = lx0, lx1, ly0, ly1
    m = 0.02 * (x1 - x0)
    xl, xr, yb, yt = x0 - m, x1 + m, y0 - m, y1 + m
    layers = []
    for lab, arr, T, n in rasters:
        disp, ext = to_display(arr, T, crs, DST, (xl, xr, yb, yt))
        layers.append((lab, disp, ext, n))
    tf = Transformer.from_crs(DST, "EPSG:4326", always_xy=True)

    W = 180.0
    left, gap, right = 11.0, 3.0, 1.0
    mw = (W - left - right - 2 * gap) / 3
    mh = mw * (yt - yb) / (xr - xl)
    top, tick_h, leg_h = 6.0, 6.0, 7.0
    H = top + mh + tick_h + leg_h
    fig = plt.figure(figsize=(W * MM, H * MM))
    print(f"figure {W:.0f} x {H:.0f} mm")
    cmap = ListedColormap(["white"] + [c for _, _, c in CLASSES])
    for k, (lab, disp, ext, km) in enumerate(layers):
        ax = fig.add_axes([(left + k * (mw + gap)) / W, (tick_h + leg_h) / H, mw / W, mh / H])
        ax.imshow(np.ma.masked_equal(disp, 0), cmap=cmap, vmin=0, vmax=4, interpolation="nearest",
                  extent=ext, zorder=2)
        for p in polys:
            x, y = p.exterior.xy
            ax.plot(x, y, color=C_EDGE, lw=0.4, zorder=1)
        ax.plot(rx, ry, color="black", lw=0.6, ls=(0, (3, 2)), zorder=3, clip_on=True)
        ax.set_xlim(xl, xr)
        ax.set_ylim(yb, yt)
        ax.set_aspect("equal")
        degree_ticks(ax, tf, label_y=(k == 0))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(0, 1.015, "abc"[k], transform=ax.transAxes, ha="left", va="bottom", fontsize=PT, fontweight="bold")
        ax.text(1, 1.015, lab, transform=ax.transAxes, ha="right", va="bottom", fontsize=PT, fontweight="bold")
        ax.text(0.97, 0.97, f"Upgraded road length\n{km:,.0f} km", transform=ax.transAxes, ha="right", va="top", fontsize=PT, zorder=6)
        L = 100000
        sx, sy = xl + 0.06 * (xr - xl), yb + 0.08 * (yt - yb)
        ax.add_patch(Rectangle((sx, sy), L, 0.008 * (yt - yb), color="black", lw=0, zorder=6))
        ax.text(sx + L / 2, sy + 0.018 * (yt - yb), "100 km", ha="center", va="bottom", fontsize=PT, zorder=6)

    handles = [Patch(facecolor=c, edgecolor="none", label=name) for _, name, c in CLASSES]
    handles.append(Patch(facecolor="none", edgecolor="black", lw=0.5, ls=(0, (3, 2)), label="Analysis area"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5 + (left - right) / 2 / W, 0.0),
               handlelength=1.2, handleheight=0.8, handletextpad=0.4, columnspacing=1.6)

    base = os.path.join(a.out, "SFig13_upgrade_sensitivity")
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
