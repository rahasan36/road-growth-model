#!/usr/bin/env python3
import argparse
import os
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
from matplotlib.patches import Circle, FancyArrow, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import ndimage

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DATA = os.path.join(REPO, "data", "figures")
DST = "ESRI:54009"
CELL = 1500.0
NGRID = 9
GRID_EXTENT = (243_300.0, 1_488_100.0, 460_000.0, 1_776_700.0)
BOXES = [
    dict(n=1, rows=(5, 7), cols=(1, 2), color="#006600", pos_b=(0.21, 0.50), pos_c=(0.17, 0.46)),
    dict(n=2, rows=(8, 9), cols=(4, 5), color="#00B0F0", pos_b=(0.74, 0.58), pos_c=(0.76, 0.58)),
    dict(n=3, rows=(2, 3), cols=(5, 5), color="#7030A0", pos_b=(0.48, 0.20), pos_c=(0.58, 0.19)),
    dict(n=4, rows=(5, 6), cols=(4, 4), color="#FFC000", pos_b=(0.50, 0.62), pos_c=(0.59, 0.80)),
]
C_GRID, C_NUM = "#767171", "#FFC000"
BOX_LW, GRID_LW = 1.1, 0.4
MAX_CENTERS = 8
SMOOTH_KM = 12
FS, FS_LETTER, FS_TITLE = 7.0, 12.0, 7.0
STACK_X, STACK_Y, STACK_W, STACK_H = 1.0, 15.0, 40.0, 61.0

C_EDGE = "#7F7F7F"
C_ROAD, C_UPG = "#BDBDBD", "#D55E00"
C_BOX, C_BADGE = "#000000", "#E69F00"

_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Arial", "Liberation Sans", "DejaVu Sans"] if f in _avail), "DejaVu Sans")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "text.color": "black", "xtick.color": "black", "ytick.color": "black",
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})


class Grid:
    def __init__(self, bounds, cell=CELL):
        x0, y0, x1, y1 = bounds
        self.nx, self.ny = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
        self.T = from_origin(x0, y1, cell, cell)
        self.ext = (x0, x0 + self.nx * cell, y1 - self.ny * cell, y1)

    def warp(self, path, how, band=1, nodata=None, valid=None):
        with rasterio.open(path) as src:
            a = src.read(band).astype(np.float32)
            nd = src.nodata if nodata is None else nodata
            if nd is not None:
                a[a == nd] = np.nan
            if valid is not None:
                a[~valid(a)] = np.nan
            out = np.full((self.ny, self.nx), np.nan, dtype=np.float32)
            reproject(a, out, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                      dst_transform=self.T, dst_crs=DST, dst_nodata=np.nan, resampling=how)
        return out


def rgba_ramp(a, mask, cmap, lo=2, hi=98, log=False, c0=0.08, c1=0.85):
    v = np.log10(a) if log else a.copy()
    ok = np.isfinite(v) & mask
    q0, q1 = np.nanpercentile(v[ok], [lo, hi])
    n = np.clip((v - q0) / max(q1 - q0, 1e-9), 0, 1)
    rgba = matplotlib.colormaps[cmap](c0 + (c1 - c0) * np.nan_to_num(n))
    rgba[~mask] = (0, 0, 0, 0)
    rgba[mask & ~np.isfinite(v)] = (1, 1, 1, 1)
    return rgba


def original_boxes(up_pts):
    xl, xr, yb, yt = GRID_EXTENT
    cw, ch = (xr - xl) / NGRID, (yt - yb) / NGRID
    xs, ys = up_pts
    out = []
    for b in BOXES:
        d = dict(b)
        d["x0"], d["x1"] = xl + (b["cols"][0] - 1) * cw, xl + b["cols"][1] * cw
        d["y1"], d["y0"] = yt - (b["rows"][0] - 1) * ch, yt - b["rows"][1] * ch
        d["px"] = int(((xs >= d["x0"]) & (xs < d["x1"]) & (ys >= d["y0"]) & (ys < d["y1"])).sum())
        out.append(d)
    return out, (xl, yt, cw, ch)


def degree_ticks(ax, tf, step=4, label_y=True):
    x0, x1 = ax.get_xlim()
    y0, y1 = sorted(ax.get_ylim())
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


def draw_grid_and_boxes(ax, grid_geom, boxes, key):
    gx0, gy1, cw, ch = grid_geom
    for i in range(NGRID + 1):
        ax.plot([gx0 + i * cw] * 2, [gy1, gy1 - NGRID * ch], color=C_GRID, lw=GRID_LW, zorder=3,
                solid_capstyle="projecting", clip_on=False)
        ax.plot([gx0, gx0 + NGRID * cw], [gy1 - i * ch] * 2, color=C_GRID, lw=GRID_LW, zorder=3,
                solid_capstyle="projecting", clip_on=False)
    r = 0.0315 * NGRID * cw
    for d in boxes:
        ax.add_patch(Rectangle((d["x0"], d["y0"]), d["x1"] - d["x0"], d["y1"] - d["y0"], fill=False,
                               ec=d["color"], lw=BOX_LW, joinstyle="miter", zorder=6, clip_on=False))
        u, v = d[key]
        bx, by = d["x0"] + u * (d["x1"] - d["x0"]), d["y1"] - v * (d["y1"] - d["y0"])
        ax.add_patch(Circle((bx, by), r, facecolor=C_NUM, edgecolor="none", zorder=7))
        ax.text(bx, by, str(d["n"]), ha="center", va="center", fontsize=FS, fontweight="bold", zorder=8)


def draw_stack(ax, x0, y0, w, h, layers, labels):
    eu = np.array([0.40, -0.31]) * w
    ev = np.array([0.60, 0.29]) * w
    n = len(layers)
    lay_h = abs(eu[1]) + ev[1]
    sp = (h - lay_h) / max(1, n - 1)
    for k in range(n - 1, -1, -1):
        A = np.array([x0, y0 + abs(eu[1]) + k * sp])
        quad = [A, A + eu, A + eu + ev, A + ev]
        z = 2 + (n - k) * 0.01
        ax.add_patch(Polygon(quad, closed=True, facecolor="white", edgecolor="none", zorder=z))
        top_left = A + eu
        tr = Affine2D(np.array([[ev[0], -eu[0], top_left[0]], [ev[1], -eu[1], top_left[1]], [0, 0, 1]])) \
            + ax.transData
        ax.imshow(layers[k], extent=(0, 1, 1, 0), origin="upper", interpolation="nearest",
                  transform=tr, zorder=z + 0.001)
        ax.add_patch(Polygon(quad, closed=True, facecolor="none", edgecolor="#555555", lw=0.5, zorder=z + 0.002))
        rc = A + eu + ev
        ax.plot([rc[0] + 0.4, rc[0] + 1.6], [rc[1], rc[1]], color="#555555", lw=0.4, zorder=z + 0.003)
        ax.text(rc[0] + 2.0, rc[1], labels[k], ha="left", va="center", fontsize=FS, linespacing=1.1,
                zorder=z + 0.003)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivers_dir", default=os.path.join(FIG_DATA, "NGA_drivers"))
    ap.add_argument("--run_dir", default=os.path.join(FIG_DATA, "NGA_forecast_SSP1_2050"))
    ap.add_argument("--roads_dir", default=os.path.join(FIG_DATA, "roads"))
    ap.add_argument("--coast", default=os.path.join(FIG_DATA, "naturalearth", "ne_10m_admin_0_countries_BGD_NGA.shp"))
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    land = gpd.read_file(os.path.join(a.roads_dir, "nga_dist.shp")).to_crs(DST).geometry.union_all()
    outline = land
    if a.coast and os.path.exists(a.coast):
        ne = gpd.read_file(a.coast)
        outline = land.intersection(ne[ne["ADM0_A3"] == "NGA"].to_crs(DST).geometry.union_all().buffer(1000))
    polys = []
    for g in getattr(outline, "geoms", [outline]):
        polys += [g] if g.geom_type == "Polygon" else list(getattr(g, "geoms", []))
    xl0, xr0, yb0, yt0 = GRID_EXTENT
    G = Grid((xl0, yb0, xr0, yt0))
    mask = rasterize([(land, 1)], out_shape=(G.ny, G.nx), transform=G.T, fill=0, dtype=np.uint8).astype(bool)

    d = a.drivers_dir
    urban = G.warp(os.path.join(d, "base", "NGA_urban.tif"), Resampling.max, nodata=0)
    roaddens = G.warp(os.path.join(d, "base", "NGA_road_density_total.tif"), Resampling.average, nodata=0)
    rd = np.where(np.isfinite(roaddens) & mask, roaddens, 0.0)
    wgt = ndimage.gaussian_filter((np.isfinite(roaddens) & mask).astype(float), SMOOTH_KM * 1000 / CELL)
    roaddens = np.where(mask, ndimage.gaussian_filter(rd, SMOOTH_KM * 1000 / CELL) / np.maximum(wgt, 1e-6), np.nan)
    pop = G.warp(os.path.join(d, "SSP1", "2020", "NGA_pop_density_total_1km.tif"), Resampling.average,
                 valid=lambda x: x > 0)
    disturb = G.warp(os.path.join(d, "base", "NGA_dist_urban.tif"), Resampling.average, nodata=0)

    with rasterio.open(os.path.join(d, "base", "NGA_urban.tif")) as src:
        u = src.read(1) > 0
        Tu, crs_u = src.transform, src.crs
    lab, n = ndimage.label(u, structure=np.ones((3, 3)))
    sizes = ndimage.sum(u, lab, range(1, n + 1))
    top = np.argsort(-sizes)[:MAX_CENTERS] + 1
    cents = ndimage.center_of_mass(u, lab, top)
    tfu = Transformer.from_crs(crs_u, DST, always_xy=True)
    centers = [(tfu.transform(Tu.c + (c + 0.5) * Tu.a, Tu.f + (r + 0.5) * Tu.e), np.sqrt(s)) for (r, c), s in
               zip(cents, sizes[top - 1])]

    urb_rgba = np.zeros((G.ny, G.nx, 4))
    urb_rgba[mask] = (1, 1, 1, 1)
    urb_rgba[mask & np.isfinite(urban)] = matplotlib.colors.to_rgba("#D55E00")
    layers = [urb_rgba,
              rgba_ramp(roaddens, mask, "Blues", lo=2, hi=98, c1=0.75),
              rgba_ramp(pop, mask, "Oranges", log=True, c1=0.70),
              rgba_ramp(disturb, mask, "Purples", lo=1, hi=99.5, log=True)]
    lb = land.bounds
    c0s, c1s = int((lb[0] - G.ext[0]) // CELL), int(np.ceil((lb[2] - G.ext[0]) / CELL))
    r0s, r1s = int((G.ext[3] - lb[3]) // CELL), int(np.ceil((G.ext[3] - lb[1]) / CELL))
    stack_layers = [L[r0s:r1s, c0s:c1s] for L in layers]
    stack_ext = (G.ext[0] + c0s * CELL, G.ext[0] + c1s * CELL, G.ext[3] - r1s * CELL, G.ext[3] - r0s * CELL)
    labels = ["Urban\ncenters", "Road\ndensity", "Population\ndensity", "Proximity to\nurban centers"]

    allr = G.warp(os.path.join(a.run_dir, "NGA_SSP1_since2015_2050_all.tif"), Resampling.max, band=5, nodata=0)
    upg = G.warp(os.path.join(a.run_dir, "NGA_SSP1_since2015_2050_upgraded.tif"), Resampling.max, band=5, nodata=0)
    with rasterio.open(os.path.join(a.run_dir, "NGA_SSP1_since2015_2050_upgraded.tif")) as src:
        up5 = src.read(5)
        To, crs_o, res = src.transform, src.crs, src.res[0]
    rr, cc = np.nonzero(up5)
    xs, ys = Transformer.from_crs(crs_o, DST, always_xy=True).transform(To.c + (cc + 0.5) * To.a,
                                                                         To.f + (rr + 0.5) * To.e)
    boxes, grid_geom = original_boxes((np.asarray(xs), np.asarray(ys)))
    rows = []
    for d in boxes:
        rows.append(dict(location=d["n"], grid_rows=f"{d['rows'][0]}-{d['rows'][1]}",
                         grid_cols=f"{d['cols'][0]}-{d['cols'][1]}", box_color=d["color"],
                         upgraded_pixels=d["px"], upgraded_km=round(d["px"] * res / 1000, 1)))
        print(f"location {d['n']}: rows {d['rows'][0]}-{d['rows'][1]}, cols {d['cols'][0]}-{d['cols'][1]}, "
              f"{d['px']:,} upgraded px ({d['px'] * res / 1000:,.0f} km)")
    print(f"total upgraded: {len(rr):,} px; inside boxes: {sum(r['upgraded_pixels'] for r in rows):,}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "Fig4_upgrade_locations.csv"), index=False)

    W, Hh = 180.0, 80.0
    fig = plt.figure(figsize=(W / 25.4, Hh / 25.4))
    page = fig.add_axes([0, 0, 1, 1])
    page.set_xlim(0, W)
    page.set_ylim(Hh, 0)
    page.axis("off")

    draw_stack(page, STACK_X, STACK_Y, STACK_W, STACK_H, stack_layers, labels)

    w = STACK_W
    eu = np.array([0.40, -0.31]) * w
    ev = np.array([0.60, 0.29]) * w
    A = np.array([STACK_X, STACK_Y + abs(eu[1])])
    gx0, gx1, gy0, gy1 = stack_ext
    drawn = []
    for (x, y), rpx in centers:
        uu, vv = (x - gx0) / (gx1 - gx0), (gy1 - y) / (gy1 - gy0)
        p = A + eu + uu * ev - vv * eu
        r = 0.8 + 0.010 * rpx
        if any(np.hypot(*(p - q)) < r + rq + 0.2 for q, rq in drawn):
            continue
        drawn.append((p, r))
        page.add_patch(Circle(p, r, fill=False, ec="black", lw=0.5, zorder=3))

    mw = 50.0
    tf = Transformer.from_crs(DST, "EPSG:4326", always_xy=True)
    xl, xr, yb, yt = G.ext
    mh = mw * (yt - yb) / (xr - xl)
    axes = []
    for k, x0 in enumerate([68.0, 122.0]):
        ax = fig.add_axes([x0 / W, (Hh - 13.0 - mh) / Hh, mw / W, mh / Hh])
        axes.append(ax)
        ax.set_xlim(xl, xr)
        ax.set_ylim(yb, yt)
        ax.set_aspect("equal")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax_b, ax_c = axes
    ax_b.imshow(layers[1], extent=G.ext, interpolation="nearest", zorder=1)
    road_img = np.zeros((G.ny, G.nx, 4))
    road_img[mask] = (1, 1, 1, 1)
    road_img[np.isfinite(allr) & mask] = matplotlib.colors.to_rgba(C_ROAD)
    up_mask = ndimage.binary_dilation(np.isfinite(upg), iterations=1) & mask
    road_img[up_mask] = matplotlib.colors.to_rgba(C_UPG)
    ax_c.imshow(road_img, extent=G.ext, interpolation="nearest", zorder=1)
    for ax, key in zip(axes, ("pos_b", "pos_c")):
        for p in polys:
            xx, yy = p.exterior.xy
            ax.plot(xx, yy, color=C_EDGE, lw=0.4, zorder=4)
        draw_grid_and_boxes(ax, grid_geom, boxes, key)
    degree_ticks(ax_b, tf, label_y=True)
    degree_ticks(ax_c, tf, label_y=True)
    ax_c.yaxis.tick_right()
    ax_c.spines["left"].set_visible(False)
    ax_c.spines["right"].set_visible(True)
    L = 200000
    sx, sy = xr - 0.04 * (xr - xl) - L, yb + 0.04 * (yt - yb)
    ax_c.add_patch(Rectangle((sx, sy), L, 0.008 * (yt - yb), color="black", lw=0, zorder=9))
    ax_c.text(sx + L / 2, sy + 0.02 * (yt - yb), "200 km", ha="center", va="bottom", fontsize=FS, zorder=9,
              bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

    for x, letter, t in [(1.0, "a", "Road upgrade location determinants"),
                         (61.0, "b", "Upgrade location and size identification"),
                         (122.0, "c", "Upgrade length identification")]:
        page.text(x, 5.0, letter, ha="left", va="center", fontsize=FS_LETTER, fontweight="bold")
        page.text(x + 5.0, 5.2, t, ha="left", va="center", fontsize=FS_TITLE, fontweight="bold")
    for xa, ya in ((58.0, 52.0), (118.4, 40.0)):
        page.add_patch(FancyArrow(xa, ya, 3.2, 0, width=1.8, head_width=3.8, head_length=2.0,
                                  length_includes_head=True, color="#BDBDBD", lw=0, zorder=5))
    k0 = 13.0 + mh + 6.6
    cax = fig.add_axes([(68.0 + 6) / W, (Hh - k0 - 0.8) / Hh, (mw - 24) / W, 1.6 / Hh])
    cax.imshow(matplotlib.colormaps["Blues"](np.linspace(0.08, 0.75, 256))[None, :, :], aspect="auto")
    cax.set_xticks([0, 255], ["Low", "High"])
    cax.set_yticks([])
    cax.tick_params(length=0, pad=1)
    for sp in cax.spines.values():
        sp.set_linewidth(0.4)
    page.text(68.0 + mw - 16.5, k0, "Road density", ha="left", va="center", fontsize=FS)
    kx = 122.0
    for i, d in enumerate(boxes):
        page.add_patch(Rectangle((kx + i * 3.6, k0 - 1.0), 2.6, 2.0, fill=False, ec=d["color"], lw=BOX_LW))
    page.text(kx + 4 * 3.6 + 0.6, k0, "Upgrade locations 1\u20134", ha="left", va="center", fontsize=FS)
    for x, face, lab in [(kx, C_ROAD, "Road network, 2050"), (kx + 27, C_UPG, "Upgraded roads")]:
        page.add_patch(Rectangle((x, k0 + 3.6), 3.0, 2.0, facecolor=face, edgecolor="none"))
        page.text(x + 4.0, k0 + 4.6, lab, ha="left", va="center", fontsize=FS)

    base = os.path.join(a.out, "Fig4_upgrade_locations")
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
