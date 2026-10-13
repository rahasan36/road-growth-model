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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DATA = os.path.join(REPO, "data", "figures")
MM = 1 / 25.4
PT = 8.0
DISPLAY_M = 2000
DST = "ESRI:102953"
RES = [(30, "30 m grid"), (100, "100 m grid"), (300, "300 m grid"),
       (500, "500 m grid"), (1000, "1,000 m grid")]
C_URB = "#d55e00"
C_EDGE = "#7f7f7f"

_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Arial", "Liberation Sans", "DejaVu Sans"] if f in _avail), "DejaVu Sans")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": PT,
    "xtick.labelsize": PT, "ytick.labelsize": PT, "legend.fontsize": PT,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic",
    "mathtext.bf": FONT + ":bold",
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "axes.edgecolor": "black", "text.color": "black",
    "xtick.color": "black", "ytick.color": "black",
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})


def to_display(arr, T, src_crs, ext):
    left, right, bottom, top = ext
    nx = int(np.ceil((right - left) / DISPLAY_M))
    ny = int(np.ceil((top - bottom) / DISPLAY_M))
    r, c = np.nonzero(arr > 0)
    x = T.c + (c + 0.5) * T.a
    y = T.f + (r + 0.5) * T.e
    x, y = Transformer.from_crs(src_crs, DST, always_xy=True).transform(x, y)
    gi = ((top - y) // DISPLAY_M).astype(int)
    gj = ((x - left) // DISPLAY_M).astype(int)
    ok = (gi >= 0) & (gi < ny) & (gj >= 0) & (gj < nx)
    out = np.zeros((ny, nx), dtype=np.uint8)
    out[gi[ok], gj[ok]] = 1
    return out, (left, left + nx * DISPLAY_M, top - ny * DISPLAY_M, top)


def degree_ticks(ax, tf, step=1.0, label_x=True, label_y=True):
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
    ax.set_xticks(np.interp(lv, lon[o], xs[o]), [f"{v:g}\u00b0E" for v in lv] if label_x else [])
    o = np.argsort(lat)
    ax.set_yticks(np.interp(pv, lat[o], ys[o]), [f"{v:g}\u00b0N" for v in pv] if label_y else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_dir", default=os.path.join(FIG_DATA, "sensitivity", "urban_clusters"))
    ap.add_argument("--roads_dir", default=os.path.join(FIG_DATA, "roads"))
    ap.add_argument("--coast", default=os.path.join(FIG_DATA, "naturalearth", "ne_10m_admin_0_countries_BGD_NGA.shp"))
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    met = pd.read_csv(os.path.join(a.cluster_dir, "BGD_cluster_metrics.csv")).set_index("resolution_m")
    agr = pd.read_csv(os.path.join(a.cluster_dir, "BGD_cluster_agreement.csv")).set_index("resolution_m")

    land = gpd.read_file(os.path.join(a.roads_dir, "bgd_dist.shp")).to_crs(DST).geometry.union_all()
    if a.coast and os.path.exists(a.coast):
        ne = gpd.read_file(a.coast)
        land = land.intersection(ne[ne["ADM0_A3"] == "BGD"].to_crs(DST).geometry.union_all().buffer(1000))
    polys = []
    for g in getattr(land, "geoms", [land]):
        polys += [g] if g.geom_type == "Polygon" else list(getattr(g, "geoms", []))
    lx0, ly0, lx1, ly1 = land.bounds
    m = 0.02 * (lx1 - lx0)
    xl, xr, yb, yt = lx0 - m, lx1 + m, ly0 - m, ly1 + m
    tf = Transformer.from_crs(DST, "EPSG:4326", always_xy=True)

    maps, rows = [], []
    for res, lab in RES:
        with rasterio.open(os.path.join(a.cluster_dir, f"BGD_clusters_{res}m.tif")) as src:
            arr, T, crs, bounds = src.read(1), src.transform, src.crs, tuple(src.bounds)
        n_px = int((arr > 0).sum())
        n_cl = int(len(np.unique(arr[arr > 0])))
        assert n_px == int(met.loc[res, "n_clustered_px"]) and n_cl == int(met.loc[res, "n_clusters"])
        disp, ext = to_display(arr, T, crs, (xl, xr, yb, yt))
        maps.append((res, lab, disp, ext, n_cl, n_px))
        rows.append(dict(grid=lab, epsilon_px=met.loc[res, "epsilon_px"], minpts=int(met.loc[res, "minpts"]),
                         n_clusters=n_cl, n_clustered_px=n_px,
                         jaccard_vs_300m=agr.loc[res, "jaccard_vs_300m"],
                         centroid_match_3km_pct=agr.loc[res, "centroid_match_3km"]))
        print(f"{lab}: {n_cl} clusters, {n_px:,} clustered pixels")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "SFig14_urban_cluster_sensitivity_counts.csv"), index=False)

    l, b, r, t = bounds
    ring = np.vstack([np.column_stack([np.linspace(l, r, 50), np.full(50, b)]),
                      np.column_stack([np.full(50, r), np.linspace(b, t, 50)]),
                      np.column_stack([np.linspace(r, l, 50), np.full(50, t)]),
                      np.column_stack([np.full(50, l), np.linspace(t, b, 50)])])
    rx, ry = Transformer.from_crs(crs, DST, always_xy=True).transform(ring[:, 0], ring[:, 1])

    W = 180.0
    left, gap, right = 11.0, 3.0, 1.0
    mw = (W - left - right - 2 * gap) / 3
    mh = mw * (yt - yb) / (xr - xl)
    top, tick_h, vgap = 6.0, 6.0, 6.0
    H = top + 2 * mh + vgap + tick_h
    fig = plt.figure(figsize=(W * MM, H * MM))
    print(f"figure {W:.0f} x {H:.0f} mm")
    cmap = ListedColormap(["white", C_URB])
    for k, (res, lab, disp, ext, n_cl, n_px) in enumerate(maps):
        row, col = divmod(k, 3)
        y = H - top - (row + 1) * mh - row * vgap
        ax = fig.add_axes([(left + col * (mw + gap)) / W, y / H, mw / W, mh / H])
        ax.imshow(np.ma.masked_equal(disp, 0), cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                  extent=ext, zorder=2)
        for p in polys:
            xx, yy = p.exterior.xy
            ax.plot(xx, yy, color=C_EDGE, lw=0.4, zorder=1)
        ax.plot(rx, ry, color="black", lw=0.6, ls=(0, (3, 2)), zorder=3)
        ax.set_xlim(xl, xr)
        ax.set_ylim(yb, yt)
        ax.set_aspect("equal")
        bottom_of_col = (row == 1) or (col == 2)
        degree_ticks(ax, tf, label_x=bottom_of_col, label_y=(col == 0))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(0, 1.015, "abcde"[k], transform=ax.transAxes, ha="left", va="bottom", fontsize=PT, fontweight="bold")
        ax.text(1, 1.015, lab, transform=ax.transAxes, ha="right", va="bottom", fontsize=PT, fontweight="bold")
        eps = met.loc[res, "epsilon_px"]
        eps_s = f"{eps:.0f}" if float(eps).is_integer() else f"{eps:.1f}"
        ax.text(0.97, 0.97,
                f"$\\varepsilon$ = {eps_s} px, MinPts = {int(met.loc[res, 'minpts'])}\n"
                f"{n_cl} clusters\n{n_px:,} pixels",
                transform=ax.transAxes, ha="right", va="top", fontsize=PT, linespacing=1.3, zorder=6)
        L = 100000
        sx, sy = xl + 0.06 * (xr - xl), yb + 0.08 * (yt - yb)
        ax.add_patch(Rectangle((sx, sy), L, 0.008 * (yt - yb), color="black", lw=0, zorder=6))
        ax.text(sx + L / 2, sy + 0.018 * (yt - yb), "100 km", ha="center", va="bottom", fontsize=PT, zorder=6)

    y = H - top - 2 * mh - vgap
    lax = fig.add_axes([(left + 2 * (mw + gap)) / W, y / H, mw / W, mh / H])
    lax.set_axis_off()
    handles = [Patch(facecolor=C_URB, edgecolor="none", label="Urban cluster"),
               Line2D([0], [0], color=C_EDGE, lw=0.8, label="Bangladesh"),
               Line2D([0], [0], color="black", lw=0.8, ls=(0, (3, 2)), label="Analysis area")]
    lax.legend(handles=handles, loc="center left", frameon=False, handlelength=1.6,
               handleheight=0.8, handletextpad=0.5, labelspacing=0.5)

    base = os.path.join(a.out, "SFig14_urban_cluster_sensitivity")
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
