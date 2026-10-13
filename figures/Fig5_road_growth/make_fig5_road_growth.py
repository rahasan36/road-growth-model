import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd
import shapely
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import road_maps as F5
from pyproj import Transformer

plt = F5.plt
ORDER = ("BGD", "BRA", "NGA")
SSPS = (1, 3, 5)
PAGE_W, LEFT, RIGHT, GAP, IGAP = 183.0, 6.0, 0.5, 2.2, 0.7
EX_W = 30.0
BLK_W = (PAGE_W - LEFT - RIGHT - EX_W - 3 * GAP) / 3
INS_W = (BLK_W - 2 * IGAP) / 3
MAP_H = 34.0
HEAD, BAND, TLBL, BLBL, ROW_GAP = 0.5, 6.2, 3.4, 3.4, 1.6
LAT_W, LON_H, MAP_H_MAX = 4.6, 4.4, 56.0
CLASSES = (1, 2, 3, 4)
CLASS_NAME = {1: "Main", 2: "Secondary", 3: "Tertiary", 4: "Local"}
C_CLASS = {1: "#D55E00", 2: "#0072B2", 3: "#E69F00", 4: "#CC79A7"}
LWC_MAP = {1: 0.40, 2: 0.29, 3: 0.19, 4: 0.15}
LWC_INS = {1: 0.98, 2: 0.77, 3: 0.60, 4: 0.49}
LW_INS = {k: {c: v * 0.7 for c, v in d.items()} for k, d in F5.LW_WIN.items()}
LW_MAP = {k: {c: v * 0.8 for c, v in d.items()} for k, d in F5.LW_NAT.items()}
INS_SCALE_KM = {"BGD": 5, "NGA": 5, "BRA": 10}
LEG = {"BGD": "lower left", "BRA": "lower left", "NGA": "lower left"}
SC = {"BGD": "ur", "BRA": "ur", "NGA": "ur"}
PAD_BOTTOM = {"BGD": 0.30, "BRA": 0.55, "NGA": 0.60}
STEP = {"BGD": (2, 2), "BRA": (20, 10), "NGA": (4, 4)}
STEP_TALL = {"BGD": (2, 2), "BRA": (20, 20), "NGA": (4, 4)}
PAD_S6 = {"BGD": 0.55, "BRA": 0.75, "NGA": 0.85}
LABEL_SIDE = {"BGD": ("r", "l", "bl"), "BRA": ("b", "b", "b"), "NGA": ("br", "b", "b")}


def extent(poly, asp, pad_bottom=0.0):
    x0, y0, x1, y1 = poly.bounds
    m = 0.02 * max(x1 - x0, y1 - y0)
    y0 -= pad_bottom * (y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0 + 2 * m, y1 - y0 + 2 * m
    w, h = max(w, h / asp), max(h, w * asp)
    return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2


def load(D, a):
    print(F5.NAMES[D], flush=True)
    dst = F5.CRS[D]
    poly = F5.country_outline(D, dst)
    x0, y0, x1, y1 = poly.bounds
    tol = max(x1 - x0, y1 - y0) / 3000.0
    F5.WIN_ASPECT = 1.0
    wins = F5.windows(D, dst)
    L0 = F5.read_roads_2015(a.roads, D, dst)
    FS = {}
    for s in SSPS:
        stem = os.path.join(a.forecast, f"{D}_SSP{s}_since2015_2050")
        F = F5.prepare(F5.forecast_lines(stem + "_new.tif", stem + "_upgraded.tif", dst, L0["g"], L0["k0"]), wins, tol)
        cells = dict(new=F5.read_cells(stem + "_new.tif", dst), up=F5.read_cells(stem + "_upgraded.tif", dst))
        FS[s] = (F, cells)
        gc.collect()
    L0 = F5.prepare(L0, wins, tol)
    return dict(poly=poly, L0=L0, FS=FS, wins=wins, tf=Transformer.from_crs(dst, "EPSG:4326", always_xy=True))


def frame(ax, lw=0.6, col="black"):
    for s in ax.spines.values():
        s.set_linewidth(lw)
        s.set_edgecolor(col)


def boxes(ax, wins, xr):
    for n, (nm, wb, wkm) in enumerate(wins):
        cx, cy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2
        hw = max((wb[2] - wb[0]) / 2, 0.022 * xr)
        for ec, lw, z in (("white", 2.4, 10), (F5.WIN_COL[n], 1.2, 10.1)):
            ax.add_patch(Rectangle((cx - hw, cy - hw), 2 * hw, 2 * hw, fill=False, ec=ec, lw=lw, zorder=z))


def map_axes(f, H, x, y, w, h, ext):
    ax = f.add_axes([x / PAGE_W, 1 - (y + h) / H, w / PAGE_W, h / H])
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def scale_bar(ax, km, sx, sy):
    y0, y1 = ax.get_ylim()
    L = km * 1000.0
    ax.add_patch(Rectangle((sx, sy), L, 0.01 * (y1 - y0), color="black", lw=0, zorder=11))
    ax.text(sx + L / 2, sy + 0.018 * (y1 - y0), f"{km:g} km", ha="center", va="bottom", fontsize=F5.FS, zorder=11,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.9))


def text_mm_height(ax):
    y0, y1 = ax.get_ylim()
    h_mm = ax.get_position().height * ax.figure.get_figheight() * 25.4
    return 3.4 / h_mm * (y1 - y0)


def scale_above_legend(ax, lg, km, poly):
    r = ax.figure.canvas.get_renderer()
    lb = lg.get_window_extent(r).transformed(ax.transData.inverted())
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w_mm = ax.get_position().width * ax.figure.get_figwidth() * 25.4
    half = 5.5 / w_mm * (x1 - x0)
    sx = max(lb.x0 + 0.05 * (x1 - x0), x0 + half - km * 500.0)
    sy = lb.y1 + 0.015 * (y1 - y0)
    box = shapely.box(sx, sy, sx + km * 1000.0, sy + 0.018 * (y1 - y0) + text_mm_height(ax))
    scale_bar(ax, km, sx, sy)
    return box.intersects(poly)


def scale_in_corner(ax, km, poly):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    L = km * 1000.0
    th = 0.018 * (y1 - y0) + text_mm_height(ax)
    half = 5.5 / (ax.get_position().width * ax.figure.get_figwidth() * 25.4) * (x1 - x0)
    for sx in (max(x0 + 0.05 * (x1 - x0), x0 + half - L / 2), min(x1 - 0.05 * (x1 - x0) - L, x1 - half - L / 2)):
        for sy in (y0 + 0.04 * (y1 - y0), y1 - 0.04 * (y1 - y0) - th):
            if not shapely.box(sx, sy, sx + L, sy + th).buffer(0.01 * (x1 - x0)).intersects(poly):
                scale_bar(ax, km, sx, sy)
                return True
    return False


def corner_labels(page, ax, tf, x, y, w, h, tick_mm=1.0):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ys = np.linspace(y0, y1, 400)
    _, lat = tf.transform(np.full_like(ys, x0), ys)
    xs = np.linspace(x0, x1, 400)
    lon, _ = tf.transform(xs, np.full_like(xs, y0))
    lv = np.floor(lat.max() * 100) / 100
    if lv >= lat.max() - 1e-9:
        lv -= 0.01
    gv = np.floor(lon.max() * 100) / 100
    if gv >= lon.max() - 1e-9:
        gv -= 0.01
    o = np.argsort(lat)
    yt = float(np.interp(lv, lat[o], ys[o]))
    o = np.argsort(lon)
    xt = float(np.interp(gv, lon[o], xs[o]))
    dx, dy = tick_mm / w * (x1 - x0), tick_mm / h * (y1 - y0)
    kw = dict(color="black", lw=0.6, solid_capstyle="butt", zorder=14, clip_on=False)
    ax.plot([x0, x0 + dx], [yt, yt], **kw)
    ax.plot([xt, xt], [y0, y0 + dy], **kw)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    page.text(x, y - 0.9, f"{abs(lv):.2f}°{'N' if lv > 0 else 'S'}", fontsize=F5.FS, ha="left", va="baseline")
    px = x + (xt - x0) / (x1 - x0) * w
    page.text(px, y + h + 0.5, f"{abs(gv):.2f}°{'E' if gv > 0 else 'W'}", fontsize=F5.FS, ha="right", va="top")


def letter_at(page, x, y, ch):
    page.text(x, y, ch, fontsize=F5.FS_LETTER, fontweight="bold", va="baseline", ha="left")


def title_at(page, x, y, t):
    page.text(x, y, t, fontsize=F5.FS, fontweight="bold", va="baseline", ha="right")


def geometry(D, ticks):
    if not ticks:
        return dict(MH=MAP_H, MX=0.0, MW=BLK_W, XL=0.0)
    x0, y0, x1, y1 = F5.country_outline(D, F5.CRS[D]).bounds
    m = 0.02 * max(x1 - x0, y1 - y0)
    MW = BLK_W - LAT_W
    MH = min(MAP_H_MAX, MW * (y1 - y0 + 2 * m) / (x1 - x0 + 2 * m))
    return dict(MH=MH, MX=LAT_W, MW=MW, XL=LON_H)


def row_height(g):
    return BAND + g["MH"] + g["XL"] + TLBL + INS_W + BLBL + ROW_GAP


def lat_ticks(ax, tf, D, step=None):
    F5.degree_ticks(ax, tf, *(step or STEP[D]))
    ax.tick_params(axis="both", length=1.8, width=0.5, pad=1.2)
    for t in ax.get_yticklabels():
        t.set_rotation(90)
        t.set_va("center")


def fc_layers(F, fig, lw_map, lw_cls):
    if fig == "fig5":
        return F5.change_layers(F, lw_map)
    return [(~F["exist"] & (F["k"] == c), C_CLASS[c], lw_cls[c], 5 + (4 - c) * 0.1) for c in (4, 3, 2, 1)]


def legend_handles(fig):
    h = [Line2D([], [], color=F5.GREY[2], lw=1.2, label="2015 roads")]
    if fig == "fig5":
        return h + [Line2D([], [], color=F5.C_NEW, lw=1.6, label="New roads"),
                    Line2D([], [], color=F5.C_UPG, lw=1.6, label="Upgraded roads")]
    return h + [Line2D([], [], color=C_CLASS[c], lw=1.6, label=CLASS_NAME[c]) for c in CLASSES]


def lengths(cells, fig, box=None):
    if fig == "fig5":
        return dict(new_km=round(F5.cells_in(cells["new"], box), 1), upgraded_km=round(F5.cells_in(cells["up"], box), 1))
    C = cells["new"]
    m = np.ones(len(C["x"]), bool)
    if box is not None:
        m &= (C["x"] >= box[0]) & (C["x"] < box[2]) & (C["y"] >= box[1]) & (C["y"] < box[3])
    return {f"new_{CLASS_NAME[c].lower()}_km": round(float((m & (C["k"] == c)).sum() * C["cell_km"]), 1) for c in CLASSES}


def build_row(f, H, page, y, D, d, letter, rows, g, a):
    top = y + BAND
    EX_H = g["MH"] + g["XL"] + TLBL + INS_W
    ex = extent(d["poly"], EX_H / EX_W, (PAD_S6 if a.figure == "sfig6" and not a.map_ticks else PAD_BOTTOM)[D])
    ax = map_axes(f, H, LEFT, top, EX_W, EX_H, ex)
    frame(ax)
    F5.draw_panel(ax, d["L0"]["gs"], F5.base_layers(d["L0"], LW_MAP))
    F5.draw_outline(ax, d["poly"], 0.5)
    lat_ticks(ax, d["tf"], D, STEP_TALL[D] if a.map_ticks else None)
    boxes(ax, d["wins"], ex[1] - ex[0])
    for n, (nm, wb, wkm) in enumerate(d["wins"]):
        hw = max((wb[2] - wb[0]) / 2, 0.022 * (ex[1] - ex[0]))
        cx, cy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2
        side, off = LABEL_SIDE[D][n], hw + 0.012 * (ex[1] - ex[0])
        if side == "b":
            xy, ha, va = (cx, cy - off), "center", "top"
        elif side == "bl":
            xy, ha, va = (cx + hw, cy - off), "right", "top"
        elif side == "br":
            xy, ha, va = (cx - hw, cy - off), "left", "top"
        else:
            xy, ha, va = (cx + (off if side == "r" else -off), cy), ("left" if side == "r" else "right"), "center"
        ax.text(*xy, nm, ha=ha, va=va, fontsize=F5.FS, zorder=12,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.25, alpha=0.85))
    ax.text(0.96, 0.985, F5.NAMES[D], transform=ax.transAxes, ha="right", va="top", fontsize=F5.FS, zorder=12,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.9))
    lg = ax.legend(handles=legend_handles(a.figure), loc=LEG[D], fontsize=F5.FS, frameon=True, framealpha=0.92,
                   edgecolor="none", handlelength=1.2, handletextpad=0.4, borderpad=0.3, labelspacing=0.25,
                   borderaxespad=0.3)
    lg.set_zorder(13)
    if scale_above_legend(ax, lg, F5.SCALE_KM[D], d["poly"]):
        print(f"  {D}: existing-map scale bar touches the country")
    letter_at(page, 0.3, top - 1.7, chr(letter))
    title_at(page, LEFT + EX_W, top - 1.7, "Existing (2015)")
    letter += 1
    km0 = float(d["L0"]["km"].sum())
    se = extent(d["poly"], g["MH"] / g["MW"])
    for j, s in enumerate(SSPS):
        F, cells = d["FS"][s]
        x0 = LEFT + EX_W + GAP + j * (BLK_W + GAP)
        axs = map_axes(f, H, x0 + g["MX"], top, g["MW"], g["MH"], se)
        for nm_sp, sp in axs.spines.items():
            sp.set_visible(a.map_ticks and nm_sp in ("left", "bottom"))
            sp.set_linewidth(0.5)
        if a.map_ticks:
            lat_ticks(axs, d["tf"], D)
        F5.draw_panel(axs, d["L0"]["gs"], F5.base_layers(d["L0"], LW_MAP))
        F5.draw_panel(axs, F["gs"], fc_layers(F, a.figure, LW_MAP, LWC_MAP))
        F5.draw_outline(axs, d["poly"], 0.5)
        boxes(axs, d["wins"], se[1] - se[0])
        if j == 0 and not scale_in_corner(axs, F5.SCALE_KM[D], d["poly"]):
            print(f"  {D}: no free corner for the forecast-map scale bar")
        letter_at(page, x0 - 0.3, top - 1.7, chr(letter))
        title_at(page, x0 + BLK_W, top - 1.7, f"SSP{s} (2050)")
        letter += 1
        rows.append(dict(country=F5.NAMES[D], ssp=f"SSP{s}", panel="National", width_km=None,
                         roads_2015_km=round(km0, 1), **lengths(cells, a.figure)))
        itop = top + g["MH"] + g["XL"] + TLBL
        for n, (nm, wb, wkm) in enumerate(d["wins"]):
            xi = x0 + n * (INS_W + IGAP)
            axw = map_axes(f, H, xi, itop, INS_W, INS_W, (wb[0], wb[2], wb[1], wb[3]))
            frame(axw, 1.0, F5.WIN_COL[n])
            Gc, sub = d["L0"]["win"][nm]
            F5.draw_panel(axw, Gc, F5.base_layers(sub, LW_INS))
            Gc, sub = F["win"][nm]
            F5.draw_panel(axw, Gc, fc_layers(sub, a.figure, LW_INS, LWC_INS))
            F5.draw_outline(axw, d["poly"].intersection(shapely.box(*wb)), 0.5)
            corner_labels(page, axw, d["tf"], xi, itop, INS_W, INS_W)
            if j == 0 and n == 0:
                km = INS_SCALE_KM[D]
                sx, sy = wb[0] + 0.13 * (wb[2] - wb[0]), wb[1] + 0.07 * (wb[3] - wb[1])
                axw.add_patch(Rectangle((sx, sy), km * 1000.0, 0.025 * (wb[3] - wb[1]), color="black", lw=0, zorder=11))
                axw.text(sx + km * 500.0, sy + 0.045 * (wb[3] - wb[1]), f"{km} km", ha="center", va="bottom",
                         fontsize=F5.FS, zorder=11, bbox=dict(facecolor="white", edgecolor="none", pad=0.2, alpha=0.9))
            rows.append(dict(country=F5.NAMES[D], ssp=f"SSP{s}", panel=nm, width_km=wkm, roads_2015_km=None,
                             **lengths(cells, a.figure, wb)))
    return letter


NAME = {"fig5": "Fig5_road_growth", "sfig6": "SFig06_new_roads_by_class"}
TITLE = {"fig5": "Figure 5: new and upgraded roads to 2050 in Bangladesh, Brazil and Nigeria under SSP1, SSP3 and SSP5.",
         "sfig6": "Supplementary Fig. 6: new roads to 2050 by class in Bangladesh, Brazil and Nigeria under SSP1, SSP3 and SSP5."}


def save(f, a, rows):
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, NAME[a.figure])
    stem, n = base, 0
    while any(os.path.exists(f"{stem}{e}") for e in (".png", ".pdf", ".svg", "_lengths.csv")):
        n += 1
        stem = f"{base}_r{n}"
    for e in ("png", "pdf", "svg"):
        f.savefig(f"{stem}.{e}", dpi=F5.DPI)
    if F5.FONT != "Arial":
        t = open(stem + ".svg", encoding="utf-8").read()
        t = t.replace(f"font-family: '{F5.FONT}', sans-serif", f"font-family: Arial, '{F5.FONT}', sans-serif").replace(
            f"font-family: '{F5.FONT}'", f"font-family: Arial, '{F5.FONT}', sans-serif")
        open(stem + ".svg", "w", encoding="utf-8").write(t)
    pd.DataFrame(rows).to_csv(stem + "_lengths.csv", index=False)
    return stem


def main(figure="fig5", out=HERE):
    ap = argparse.ArgumentParser(description=TITLE[figure])
    ap.add_argument("--forecast", default=os.path.join(F5.FIG_DATA, "fig5_forecast"),
                    help="folder with <ISO3>_SSP<s>_since2015_2050_new.tif and _upgraded.tif")
    ap.add_argument("--roads", default=os.path.join(F5.FIG_DATA, "fig5_roads_2015"),
                    help="folder with <ISO3>_roads_2015.part<n>.fgb.zip")
    ap.add_argument("--no_lat_lon", action="store_true", help="draw the forecast maps without latitude and longitude")
    ap.add_argument("--out", default=out)
    a = ap.parse_args()
    a.figure, a.map_ticks = figure, not a.no_lat_lon
    geo = {D: geometry(D, a.map_ticks) for D in ORDER}
    H = HEAD + sum(row_height(geo[D]) for D in ORDER)
    f = plt.figure(figsize=(PAGE_W * F5.MM, H * F5.MM))
    page = f.add_axes([0, 0, 1, 1])
    page.set_xlim(0, PAGE_W)
    page.set_ylim(H, 0)
    page.axis("off")
    page.set_zorder(-1)
    y, letter, rows = HEAD, ord("a"), []
    for D in ORDER:
        d = load(D, a)
        letter = build_row(f, H, page, y, D, d, letter, rows, geo[D], a)
        del d
        gc.collect()
        y += row_height(geo[D])
    stem = save(f, a, rows)
    plt.close(f)
    print(f"Wrote {stem}.png, .pdf, .svg and _lengths.csv ({PAGE_W:.0f} x {H:.0f} mm)")


if __name__ == "__main__":
    main()
