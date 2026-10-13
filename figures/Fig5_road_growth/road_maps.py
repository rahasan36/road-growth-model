import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from pyproj import CRS as ProjCRS, Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import road_lines as RL

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DATA = os.path.join(REPO, "data", "figures")
NE = os.path.join(FIG_DATA, "naturalearth", "ne_10m_admin_0_countries_BGD_NGA_BRA.shp")
DIST = os.path.join(FIG_DATA, "roads")
MM = 1 / 25.4
DPI = 600
FS, FS_LETTER = 8.0, 14.0
COUNTRIES = ("BGD", "NGA", "BRA")
NAMES = {"BGD": "Bangladesh", "NGA": "Nigeria", "BRA": "Brazil"}
CRS = {"BGD": "ESRI:102953", "NGA": "ESRI:102022", "BRA": "ESRI:102033"}
WIN = {"BGD": [("Dhaka", 90.41, 23.81, 0, 5, 14), ("Khulna", 89.55, 22.82, -3, -2, 14), ("Chattogram", 91.80, 22.36, 3, 4, 14)],
       "NGA": [("Lagos", 3.38, 6.52, 4, 9, 16), ("Kano", 8.52, 12.00, -1, -5, 16), ("Abuja", 7.49, 9.06, -4, 0, 16)],
       "BRA": [("São Paulo", -46.63, -23.55, -16, 18, 30), ("Brasília", -47.88, -15.79, 14, 12, 20),
               ("Manaus", -60.02, -3.12, -5, -9, 30)]}
WIN_COL = ("#009E73", "#332288", "#882255")
WIN_SCALE_KM = {"BGD": 2, "NGA": 2, "BRA": 5}
SCALE_KM = {"BGD": 100, "NGA": 200, "BRA": 500}
PLACE = {"BGD": dict(scale="ur", legend="lower left"),
         "NGA": dict(scale="ll", legend="lower right"),
         "BRA": dict(scale="lr", legend="lower left")}
PAGE_W, LEFT, RIGHT, GAP, BAND, XLAB = 183.0, 10.0, 1.0, 5.0, 6.0, 5.0
MAP_H, NAT_W = 52.0, 46.0
WIN_W = (PAGE_W - LEFT - RIGHT - NAT_W - 3 * GAP) / 3
WIN_ASPECT = MAP_H / WIN_W
GREY = {1: "#6E6E6E", 2: "#8C8C8C", 3: "#ADADAD", 4: "#C9C9C9"}
C_NEW, C_UPG, C_EDGE = "#D55E00", "#0072B2", "#4D4D4D"
LW_NAT = dict(base={1: 0.35, 2: 0.25, 3: 0.15, 4: 0.10}, new={1: 0.45, 2: 0.32, 3: 0.22, 4: 0.18},
              up={1: 0.40, 2: 0.30, 3: 0.20, 4: 0.16})
LW_WIN = dict(base={1: 1.10, 2: 0.80, 3: 0.55, 4: 0.40}, new={1: 1.30, 2: 1.05, 3: 0.85, 4: 0.70},
              up={1: 1.20, 2: 0.95, 3: 0.75, 4: 0.60})

_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Arial", "Liberation Sans", "DejaVu Sans") if f in _fonts), "DejaVu Sans")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "text.color": "black", "xtick.color": "black", "ytick.color": "black",
    "pdf.fonttype": 42, "agg.path.chunksize": 20000, "svg.fonttype": "none", "savefig.facecolor": "white",
})


def country_outline(D, dst):
    ne = gpd.read_file(NE)
    ne = ne[ne["ADM0_A3"] == D].to_crs(dst).geometry.union_all()
    p = os.path.join(DIST, f"{D.lower()}_dist.shp")
    if os.path.exists(p):
        return gpd.read_file(p).to_crs(dst).geometry.union_all().intersection(ne.buffer(1000))
    parts = list(shapely.get_parts(ne))
    main = max(parts, key=lambda q: q.area)
    return shapely.union_all([q for q in parts if q.distance(main) < 300_000])


def read_roads_2015(folder, D, dst):
    parts = sorted(glob.glob(os.path.join(folder, f"{D}_roads_2015.part*.fgb.zip")),
                   key=lambda q: int(re.search(r"part(\d+)\.fgb\.zip$", q).group(1)))
    if not parts:
        raise SystemExit(f"no {D}_roads_2015.part*.fgb.zip in {folder}")
    dfs = [pyogrio.read_dataframe("/vsizip/" + q, columns=["k"]) for q in parts]
    df = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs=dfs[0].crs)
    g = df.geometry.values
    if not ProjCRS.from_user_input(df.crs).equals(ProjCRS.from_user_input(dst)):
        tf = Transformer.from_crs(df.crs, dst, always_xy=True)
        g = shapely.transform(g, lambda xy: np.column_stack(tf.transform(xy[:, 0], xy[:, 1])))
    k = df.k.values.astype(int)
    n = len(k)
    return dict(g=g, k=k, k0=k, exist=np.ones(n, bool), up=np.zeros(n, bool), km=shapely.length(g) / 1000)


def read_cells(path, dst):
    with rasterio.open(path) as r:
        a = r.read(5)
        T, crs, cell_km = r.transform, r.crs, r.res[0] / 1000.0
    rr, cc = np.nonzero(a)
    k = a[rr, cc]
    xs, ys = rasterio.transform.xy(T, rr, cc)
    px, py = Transformer.from_crs(crs, dst, always_xy=True).transform(np.asarray(xs), np.asarray(ys))
    return dict(x=np.asarray(px), y=np.asarray(py), k=k.astype(np.uint8), cell_km=cell_km)


def cells_in(C, box=None):
    m = np.ones(len(C["x"]), bool)
    if box is not None:
        m &= (C["x"] >= box[0]) & (C["x"] < box[2]) & (C["y"] >= box[1]) & (C["y"] < box[3])
    return m.sum() * C["cell_km"]


def forecast_lines(new_tif, up_tif, dst, EX, EXk, save=None):
    with rasterio.open(new_tif) as r:
        a, T, crs = r.read(5), r.transform, r.crs
    gN, kN, stN = RL.new_road_lines(a, T, crs, dst, existing=EX)
    with rasterio.open(up_tif) as r:
        a, T, crs = r.read(5), r.transform, r.crs
    gU, kU, kO, stU = RL.upgraded_on_lines(EX, EXk, a, T, crs, dst)
    print("  new road lines", stN)
    print("  upgraded road lines", stU)
    if save:
        gpd.GeoDataFrame(dict(k=kN), geometry=gN, crs=dst).to_file(save + "_new_lines.fgb", driver="FlatGeobuf")
        gpd.GeoDataFrame(dict(k=kU, k_old=kO), geometry=gU, crs=dst).to_file(save + "_upgraded_lines.fgb", driver="FlatGeobuf")
    g = np.concatenate([gN, gU])
    nN, nU = len(kN), len(kU)
    return dict(g=g, k=np.r_[kN, kU], k0=np.r_[kN, kO], exist=np.r_[np.zeros(nN, bool), np.ones(nU, bool)],
                up=np.r_[np.zeros(nN, bool), np.ones(nU, bool)], km=shapely.length(g) / 1000)


def clip_to(L, box):
    clip = shapely.box(*box)
    G = L["g"]
    sel = shapely.intersects(G, clip)
    sub = {k: v[sel] for k, v in L.items() if isinstance(v, np.ndarray) and v.shape[:1] == G.shape and k not in ("g", "gs")}
    return shapely.intersection(G[sel], clip), sub


def prepare(L, wins, tol):
    L["gs"] = shapely.simplify(L["g"], tol)
    L["win"] = {nm: clip_to(L, wb) for nm, wb, _ in wins}
    del L["g"]
    return L


def windows(D, dst):
    tf = Transformer.from_crs("EPSG:4326", dst, always_xy=True)
    out = []
    for nm, lon, lat, dx, dy, wkm in WIN[D]:
        x, y = tf.transform(lon, lat)
        x, y = x + dx * 1e3, y + dy * 1e3
        hw = wkm * 500.0
        hh = hw * WIN_ASPECT
        out.append((nm, (x - hw, y - hh, x + hw, y + hh), wkm))
    return out


def base_layers(L, w):
    return [(L["exist"] & (L["k0"] == c), GREY[c], w["base"][c], 2 + (4 - c) * 0.1) for c in (4, 3, 2, 1)]


def change_layers(F, w):
    out = [(F["up"] & (F["k"] == c), C_UPG, w["up"][c], 4 + (4 - c) * 0.1) for c in (4, 3, 2, 1)]
    out += [(~F["exist"] & (F["k"] == c), C_NEW, w["new"][c], 5 + (4 - c) * 0.1) for c in (4, 3, 2, 1)]
    return out


def draw_lines(ax, geoms, color, lw, z):
    geoms = geoms[~shapely.is_empty(geoms)]
    if len(geoms) == 0:
        return
    xy, idx = shapely.get_coordinates(geoms, return_index=True)
    if len(xy) < 2:
        return
    codes = np.full(len(xy), MplPath.LINETO, np.uint8)
    codes[np.r_[0, np.flatnonzero(np.diff(idx)) + 1]] = MplPath.MOVETO
    ax.add_patch(PathPatch(MplPath(xy, codes), fill=False, edgecolor=color, linewidth=lw, zorder=z,
                           capstyle="round", joinstyle="round"))


def draw_panel(ax, G, layers):
    f = ax.figure
    pos = ax.get_position()
    g = Figure(figsize=(pos.width * f.get_figwidth(), pos.height * f.get_figheight()), dpi=DPI)
    FigureCanvasAgg(g)
    g.patch.set_alpha(0)
    a2 = g.add_axes([0, 0, 1, 1])
    a2.set_xlim(ax.get_xlim())
    a2.set_ylim(ax.get_ylim())
    a2.axis("off")
    a2.patch.set_alpha(0)
    for m, col, lw, z in layers:
        draw_lines(a2, G[m], col, lw, z)
    g.canvas.draw()
    img = np.asarray(g.canvas.buffer_rgba()).copy()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ax.imshow(img, extent=(x0, x1, y0, y1), interpolation="none", aspect="auto", zorder=3)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def draw_outline(ax, poly, lw):
    for p in shapely.get_parts(poly):
        for ring in [p.exterior] + list(p.interiors):
            x, y = np.asarray(ring.coords).T
            ax.plot(x, y, color=C_EDGE, lw=lw, zorder=1.5, solid_joinstyle="round")


def degree_ticks(ax, tf, lon_step, lat_step):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = np.linspace(x0, x1, 400)
    lon, _ = tf.transform(xs, np.full_like(xs, y0))
    ys = np.linspace(y0, y1, 400)
    _, lat = tf.transform(np.full_like(ys, x0), ys)

    def vals(v, step):
        o = np.arange(np.ceil(v.min() / step) * step, v.max(), step)
        return o[(o > v.min() + 0.04 * np.ptp(v)) & (o < v.max() - 0.04 * np.ptp(v))]

    def fmt(v, pos, neg):
        return "0°" if abs(v) < 1e-9 else f"{abs(v):g}°{pos if v > 0 else neg}"

    lv, pv = vals(lon, lon_step), vals(lat, lat_step)
    o = np.argsort(lon)
    ax.set_xticks(np.interp(lv, lon[o], xs[o]), [fmt(v, "E", "W") for v in lv])
    o = np.argsort(lat)
    ax.set_yticks(np.interp(pv, lat[o], ys[o]), [fmt(v, "N", "S") for v in pv])


def window_ticks(ax, tf, step, nd):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = np.linspace(x0, x1, 200)
    lon, _ = tf.transform(xs, np.full_like(xs, y0))
    ys = np.linspace(y0, y1, 200)
    _, lat = tf.transform(np.full_like(ys, x0), ys)

    def vals(v):
        o = np.round(np.arange(np.ceil(v.min() / step) * step, v.max(), step), 6)
        return o[(o > v.min() + 0.08 * np.ptp(v)) & (o < v.max() - 0.08 * np.ptp(v))]

    def fmt(v, pos, neg):
        return f"{abs(v):.{nd}f}°{pos if v > 0 else neg}"

    lv, pv = vals(lon), vals(lat)
    o = np.argsort(lon)
    ax.set_xticks(np.interp(lv, lon[o], xs[o]), [fmt(v, "E", "W") for v in lv])
    o = np.argsort(lat)
    ax.set_yticks(np.interp(pv, lat[o], ys[o]), [fmt(v, "N", "S") for v in pv])
    ax.tick_params(axis="both", length=1.8, width=0.5, pad=1.2)
    for t in ax.get_yticklabels():
        t.set_rotation(90)
        t.set_va("center")


def scale_bar(ax, km, where):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    L = km * 1000.0
    sx = x0 + 0.06 * (x1 - x0) if where in ("ll", "ul") else x1 - 0.06 * (x1 - x0) - L
    sy = y0 + 0.05 * (y1 - y0) if where in ("ll", "lr") else y1 - 0.16 * (y1 - y0)
    ax.add_patch(Rectangle((sx, sy), L, 0.012 * (y1 - y0), color="black", lw=0, zorder=11))
    ax.text(sx + L / 2, sy + 0.022 * (y1 - y0), f"{km:g} km", ha="center", va="bottom", fontsize=FS, zorder=11,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.9))
