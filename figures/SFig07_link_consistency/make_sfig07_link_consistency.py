import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.transforms
import pickle
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from shapely import get_coordinates
from pyproj import Transformer

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

LOCATIONS = [
    ("bgd", "Dhaka", 17, "Dhaka"),
    ("bgd", "Chattogram", 9, "Chattogram"),
    ("nga", "Abuja Municipal", 11, "Abuja"),
]
COUNTRY = {"bgd": "Bangladesh", "nga": "Nigeria"}
SSPS = [("ssp1", "SSP1"), ("ssp3", "SSP3"), ("ssp5", "SSP5")]

C_CONSISTENT = "#0072b2"
C_SPURIOUS = "#e69f00"
C_MISSING = "#000000"
C_LAND = "#efefef"
C_EDGE = "#8c8c8c"
LW = 0.35

_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Arial", "Liberation Sans", "DejaVu Sans"] if f in _avail), "DejaVu Sans")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT],
    "font.size": PT,
    "legend.fontsize": PT,
    "text.color": "black",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def classify(obs, pred, boundary, buffer_m):
    obs = obs.iloc[obs.sindex.query(boundary, predicate="intersects")]
    pred = pred.iloc[pred.sindex.query(boundary, predicate="intersects")]
    obs_c = gpd.clip(obs, boundary)
    pred_c = gpd.clip(pred, boundary)
    obs_buf = obs_c.geometry.buffer(buffer_m).union_all() if len(obs_c) else None
    pred_buf = pred_c.geometry.buffer(buffer_m).union_all() if len(pred_c) else None
    if obs_buf is not None and len(pred_c):
        hit = pred_c.intersects(obs_buf)
        consistent, spurious = pred_c[hit], pred_c[~hit]
    else:
        consistent, spurious = pred_c.iloc[0:0], pred_c
    missing = obs_c[~obs_c.intersects(pred_buf)] if (pred_buf is not None and len(obs_c)) else obs_c
    return consistent, missing, spurious


def segments(gdf):
    out = []
    for g in gdf.geometry:
        if g is None or g.is_empty:
            continue
        parts = g.geoms if g.geom_type.startswith("Multi") else [g]
        for p in parts:
            if p.geom_type == "LineString":
                out.append(np.asarray(p.coords)[:, :2])
    return out


def draw_boundary(ax, geom):
    parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for p in parts:
        x, y = p.exterior.xy
        ax.fill(x, y, color=C_LAND, lw=0, zorder=0)
        ax.plot(x, y, color=C_EDGE, lw=0.4, zorder=5)


def ink_in_box(ax, pts, bb_disp):
    if len(pts) == 0:
        return 0
    d = ax.transData.transform(pts)
    return int(np.sum((d[:, 0] > bb_disp.x0) & (d[:, 0] < bb_disp.x1) &
                      (d[:, 1] > bb_disp.y0) & (d[:, 1] < bb_disp.y1)))


def best_corner(ax, handles, pts, exclude=()):
    r = ax.figure.canvas.get_renderer()
    scores = []
    for loc in ["upper right", "upper left", "lower left", "lower right", "center left", "center right"]:
        if loc in exclude:
            continue
        lg = ax.legend(handles=handles, loc=loc, frameon=False, borderaxespad=0.2,
                       handlelength=1.0, handletextpad=0.3, labelspacing=0.2)
        ax.figure.canvas.draw()
        scores.append((ink_in_box(ax, pts, lg.get_window_extent(r)), loc))
        lg.remove()
    return min(scores)[1]


def nice_length(km):
    for v in [1, 2, 5, 10, 15, 20, 25, 50, 100]:
        if v >= km:
            return v
    return 100


def scale_bar(ax, pts, exclude):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    km = nice_length(0.15 * w / 1000)
    L = km * 1000
    bar_h = 0.006 * h
    r = ax.figure.canvas.get_renderer()
    best = None
    for corner in ["lower left", "lower right", "upper right", "upper left"]:
        if corner in exclude:
            continue
        sx = x0 + 0.03 * w if "left" in corner else x1 - 0.03 * w - L
        sy = y0 + 0.03 * h if "lower" in corner else y1 - 0.09 * h
        box = matplotlib.transforms.Bbox.from_extents(
            *ax.transData.transform((sx - 0.01 * w, sy - 0.01 * h)),
            *ax.transData.transform((sx + L + 0.01 * w, sy + 0.07 * h)))
        n = ink_in_box(ax, pts, box)
        if best is None or n < best[0]:
            best = (n, sx, sy)
    _, sx, sy = best
    ax.add_patch(Rectangle((sx, sy), L, bar_h, color="black", lw=0, zorder=7))
    ax.text(sx + L / 2, sy + 2.5 * bar_h, f"{km} km", ha="center", va="bottom",
            fontsize=PT, zorder=7)


def degree_ticks(ax, crs, label_x=True, label_y=True):
    tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = np.linspace(x0, x1, 400)
    lon, _ = tf.transform(xs, np.full_like(xs, y0))
    ys = np.linspace(y0, y1, 400)
    _, lat = tf.transform(np.full_like(ys, x0), ys)

    def pick(lo, hi):
        for step in (0.05, 0.1, 0.2, 0.5, 1.0):
            v = np.arange(np.ceil(lo / step) * step, hi, step)
            v = v[(v > lo + 0.04 * (hi - lo)) & (v < hi - 0.04 * (hi - lo))]
            if len(v) <= 3:
                return v, step
        return v, step

    def fmt(v, step, hemi):
        d = 0 if step >= 1 else (1 if step >= 0.1 else 2)
        return f"{v:.{d}f}\u00b0{hemi}"

    lv, ls = pick(lon.min(), lon.max())
    order = np.argsort(lon)
    ax.set_xticks(np.interp(lv, lon[order], xs[order]),
                  [fmt(v, ls, "E") for v in lv] if label_x else [])
    pv, ps = pick(lat.min(), lat.max())
    order = np.argsort(lat)
    ax.set_yticks(np.interp(pv, lat[order], ys[order]),
                  [fmt(v, ps, "N") for v in pv] if label_y else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=os.path.join(FIG_DATA, "SFig07_district_extracts"))
    ap.add_argument("--dist_dir", default=os.path.join(FIG_DATA, "roads"),
                    help="folder with bgd_dist.shp and nga_dist.shp")
    ap.add_argument("--year", type=int, default=2050)
    ap.add_argument("--buffer", type=float, default=50.0, help="match buffer in metres")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--cache", default=None,
                    help="optional .pkl path: reuse a saved classification to re-draw quickly")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.cache and os.path.exists(a.cache):
        with open(a.cache, "rb") as f:
            results, rows = pickle.load(f)
        print("classification read from", a.cache)
    else:
        results, rows = classify_all(a)
        if a.cache:
            with open(a.cache, "wb") as f:
                pickle.dump((results, rows), f)
    for row in rows:
        print(f"{row['city']:<11} {row['scenario']}: consistent {row['consistent']:>6}  "
              f"missing {row['missing']:>5}  spurious {row['spurious']:>5}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "SFig07_link_consistency_counts.csv"), index=False)
    draw(a, results)


def classify_all(a):
    data = {}
    for cc in sorted({l[0] for l in LOCATIONS}):
        obs = read_vector(os.path.join(a.input_dir, f"{cc}_roads_2015.shp"))
        adm = read_vector(os.path.join(a.dist_dir, f"{cc}_dist.shp"))
        if adm.crs != obs.crs:
            adm = adm.to_crs(obs.crs)
        preds = {}
        for s, _ in SSPS:
            p = read_vector(os.path.join(a.input_dir, f"{cc}_roads_{a.year}_{s}.shp"))
            preds[s] = p.to_crs(obs.crs) if p.crs != obs.crs else p
        data[cc] = (obs, adm, preds)

    results, rows = {}, []
    for cc, name, idx, disp in LOCATIONS:
        obs, adm, preds = data[cc]
        sel = adm[adm["adm2_name"] == name]
        if len(sel) != 1 or sel.index[0] != idx:
            raise SystemExit(f"{name}: expected one district at index {idx}, found {sel.index.tolist()}")
        geom = sel.geometry.iloc[0]
        for s, slab in SSPS:
            con, mis, spu = classify(obs, preds[s], geom, a.buffer)
            results[(disp, s)] = (geom, con, mis, spu)
            rows.append(dict(city=disp, district=name, year=a.year, scenario=slab,
                             buffer_m=a.buffer, consistent=len(con), missing=len(mis),
                             spurious=len(spu)))
    return results, rows


def draw(a, results):

    ext = {}
    for cc, name, idx, disp in LOCATIONS:
        x0, y0, x1, y1 = results[(disp, "ssp1")][0].bounds
        m = 0.02 * max(x1 - x0, y1 - y0)
        ext[disp] = (x0 - m, x1 + m, y0 - m, y1 + m)
    aspects = [(e[1] - e[0]) / (e[3] - e[2]) for e in ext.values()]
    W = 180.0
    left_mm, right_mm, gap_mm = 15.0, 1.0, 10.0
    top_mm, bottom_mm, vgap_mm = 6.0, 5.0, 4.0
    usable = W - left_mm - right_mm - 2 * gap_mm
    H_map = usable / sum(aspects)
    widths = [H_map * r for r in aspects]
    H = top_mm + 3 * H_map + 2 * vgap_mm + bottom_mm
    fig = plt.figure(figsize=(W * MM, H * MM))
    print(f"figure {W:.0f} x {H:.0f} mm, map height {H_map:.1f} mm")

    handles_proto = [(C_CONSISTENT, "Consistent"), (C_MISSING, "Missing"), (C_SPURIOUS, "Spurious")]
    letters = iter("abcdefghi")
    for r, (s, slab) in enumerate(SSPS):
        x = left_mm
        for c, (cc, name, idx, disp) in enumerate(LOCATIONS):
            y = H - top_mm - (r + 1) * H_map - r * vgap_mm
            ax = fig.add_axes([x / W, y / H, widths[c] / W, H_map / H])
            x += widths[c] + gap_mm
            geom, con, mis, spu = results[(disp, s)]
            draw_boundary(ax, geom)
            for gdf, col, z in [(con, C_CONSISTENT, 2), (spu, C_SPURIOUS, 3), (mis, C_MISSING, 4)]:
                segs = segments(gdf)
                if segs:
                    ax.add_collection(LineCollection(segs, colors=col, linewidths=LW,
                                                     zorder=z, capstyle="round"))
            e = ext[disp]
            ax.set_xlim(e[0], e[1])
            ax.set_ylim(e[2], e[3])
            ax.set_aspect("equal")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_linewidth(0.5)
            ax.tick_params(length=2.0, width=0.5, labelsize=PT, pad=1.5)
            degree_ticks(ax, con.crs, label_x=(r == len(SSPS) - 1), label_y=True)
            ax.text(0.0, 1.01, next(letters), transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=PT, fontweight="bold", zorder=8)
            if r == 0:
                ax.text(0.5, 1.0, f"{disp}, {COUNTRY[cc]}", transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=PT, fontweight="bold")
            if c == 0:
                fig.text(2.0 / W, (y + H_map / 2) / H, slab, ha="left", va="center",
                         rotation=90, fontsize=PT, fontweight="bold")
            handles = [Line2D([0], [0], color=col, lw=1.2,
                              label=f"{lab} ({n:,})")
                       for (col, lab), n in zip(handles_proto, (len(con), len(mis), len(spu)))]
            pts = np.vstack([get_coordinates(g.geometry.values) for g in (con, mis, spu) if len(g)]
                            or [np.empty((0, 2))])
            loc = best_corner(ax, handles, pts)
            ax.legend(handles=handles, loc=loc, frameon=False, borderaxespad=0.2,
                      handlelength=1.0, handletextpad=0.3, labelspacing=0.2)
            if r == len(SSPS) - 1:
                scale_bar(ax, pts, exclude=(loc,))

    base = os.path.join(a.out, "SFig07_link_consistency")
    fig.savefig(base + ".png", dpi=600)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".svg")
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
