import argparse
import os
import sys

import numpy as np
import pandas as pd
import shapely
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Fig5_road_growth"))
import make_fig5_road_growth as F5

plt = F5.plt
CLASSES = (1, 2, 3, 4)
CLASS_NAME = {1: "Main", 2: "Secondary", 3: "Tertiary", 4: "Local"}
C_CLASS = {1: "#D55E00", 2: "#0072B2", 3: "#E69F00", 4: "#CC79A7"}
LW_NAT = {1: 0.50, 2: 0.36, 3: 0.24, 4: 0.18}
LW_WIN = {1: 1.40, 2: 1.10, 3: 0.85, 4: 0.70}
PLACE = {"BGD": dict(scale="ur", legend="lower left", ncol=2, pad_bottom=0.20),
         "NGA": dict(scale="ll", legend="lower right", ncol=1, pad_bottom=0.0),
         "BRA": dict(scale="lr", legend="lower left", ncol=1, pad_bottom=0.0)}


def widen(ext, pad_bottom):
    xl, xr, yb, yt = ext
    dh = pad_bottom * (yt - yb)
    dw = dh * (xr - xl) / (yt - yb)
    return xl - dw / 2, xr + dw / 2, yb - dh, yt


def class_layers(F, lw):
    return [(~F["exist"] & (F["k"] == c), C_CLASS[c], lw[c], 5 + (4 - c) * 0.1) for c in (4, 3, 2, 1)]


def class_km(d, box=None):
    if d["cells"] is None:
        return None
    C = d["cells"]["new"]
    m = np.ones(len(C["x"]), bool)
    if box is not None:
        m &= (C["x"] >= box[0]) & (C["x"] < box[2]) & (C["y"] >= box[1]) & (C["y"] < box[3])
    return {c: float((m & (C["k"] == c)).sum() * C["cell_km"]) for c in CLASSES}


def build(a, data):
    W, FS, MM = F5.PAGE_W, F5.FS, F5.MM
    H = len(data) * (F5.BAND + F5.MAP_H + F5.XLAB) + 1.0
    f = plt.figure(figsize=(W * MM, H * MM))
    page = f.add_axes([0, 0, 1, 1])
    page.set_xlim(0, W)
    page.set_ylim(H, 0)
    page.axis("off")
    page.set_zorder(-1)
    y, letter, rows = 0.0, ord("a"), []
    for D, d in data.items():
        xl, xr, yb, yt = widen(d["ext"], PLACE[D]["pad_bottom"])
        top = y + F5.BAND
        ax = f.add_axes([F5.LEFT / W, 1 - (top + F5.MAP_H) / H, F5.NAT_W / W, F5.MAP_H / H])
        ax.set_xlim(xl, xr)
        ax.set_ylim(yb, yt)
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_linewidth(0.6)
        F5.draw_panel(ax, d["L0"]["gs"], F5.base_layers(d["L0"], F5.LW_NAT))
        if d["F"] is not None:
            F5.draw_panel(ax, d["F"]["gs"], class_layers(d["F"], LW_NAT))
        F5.draw_outline(ax, d["poly"], 0.5)
        F5.degree_ticks(ax, d["tf"], d["step"][0] * (2 if PLACE[D]["pad_bottom"] else 1), d["step"][1])
        ax.text(0.97, 0.97, F5.NAMES[D], transform=ax.transAxes, ha="right", va="top", fontsize=FS, zorder=12,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.9))
        F5.scale_bar(ax, F5.SCALE_KM[D], PLACE[D]["scale"])
        km0 = float(d["L0"]["km"].sum())
        tot = class_km(d)
        h = [Line2D([], [], color=F5.GREY[2], lw=1.2, label="2015 roads")]
        if tot is not None:
            h += [Line2D([], [], color=C_CLASS[c], lw=1.6, label=CLASS_NAME[c]) for c in CLASSES]
        else:
            h += [Line2D([], [], color="white", lw=0, label="No forecast")]
        lg = ax.legend(handles=h, loc=PLACE[D]["legend"], fontsize=FS, frameon=True, framealpha=0.92, edgecolor="none",
                       handlelength=1.2, handletextpad=0.4, borderpad=0.3, labelspacing=0.25, borderaxespad=0.3,
                       ncol=PLACE[D]["ncol"], columnspacing=0.9)
        lg.set_zorder(13)
        page.text(F5.LEFT - 7.5, y + 3.2, chr(letter), fontsize=F5.FS_LETTER, fontweight="bold", va="center", ha="left")
        letter += 1
        for n, (nm, wb, wkm) in enumerate(d["wins"]):
            cx, cy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2
            mn = 0.022 * (xr - xl)
            hw, hh = max((wb[2] - wb[0]) / 2, mn), max((wb[3] - wb[1]) / 2, mn * F5.WIN_ASPECT)
            for ec, lw, z in (("white", 2.9, 10), (F5.WIN_COL[n], 1.5, 10.1)):
                ax.add_patch(Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh, fill=False, ec=ec, lw=lw, zorder=z))
        for n, (nm, wb, wkm) in enumerate(d["wins"]):
            x0 = F5.LEFT + F5.NAT_W + F5.GAP + n * (F5.WIN_W + F5.GAP)
            axw = f.add_axes([x0 / W, 1 - (top + F5.MAP_H) / H, F5.WIN_W / W, F5.MAP_H / H])
            axw.set_xlim(wb[0], wb[2])
            axw.set_ylim(wb[1], wb[3])
            axw.set_aspect("equal")
            for s in axw.spines.values():
                s.set_linewidth(1.1)
                s.set_edgecolor(F5.WIN_COL[n])
            F5.window_ticks(axw, d["tf"], 0.05 if wkm <= 16 else 0.1, 2 if wkm <= 16 else 1)
            Gc, sub = d["L0"]["win"][nm]
            F5.draw_panel(axw, Gc, F5.base_layers(sub, F5.LW_WIN))
            if d["F"] is not None:
                Gc, sub = d["F"]["win"][nm]
                F5.draw_panel(axw, Gc, class_layers(sub, LW_WIN))
            F5.draw_outline(axw, d["poly"].intersection(shapely.box(*wb)), 0.6)
            page.text(x0 + F5.WIN_W, y + 3.6, nm, fontsize=FS, ha="right", va="center")
            km = F5.WIN_SCALE_KM[D]
            sx, sy = wb[0] + 0.06 * (wb[2] - wb[0]), wb[1] + 0.035 * (wb[3] - wb[1])
            axw.add_patch(Rectangle((sx, sy), km * 1000.0, 0.012 * (wb[3] - wb[1]), color="black", lw=0, zorder=11))
            axw.text(sx + km * 500.0, sy + 0.02 * (wb[3] - wb[1]), f"{km} km", ha="center", va="bottom", fontsize=FS,
                     zorder=11, bbox=dict(facecolor="white", edgecolor="none", pad=0.3, alpha=0.9))
            wl = class_km(d, wb)
            if wl is not None:
                axw.text(0.96, 0.035, f"New {sum(wl.values()):,.0f} km", transform=axw.transAxes, fontsize=FS,
                         ha="right", va="bottom", zorder=12, bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.9))
            page.text(x0 - 0.2, y + 3.2, chr(letter), fontsize=F5.FS_LETTER, fontweight="bold", va="center", ha="left")
            letter += 1
            row = dict(country=F5.NAMES[D], panel=nm, width_km=wkm, height_km=round((wb[3] - wb[1]) / 1000, 1), roads_2015_km=None)
            row.update({f"new_{CLASS_NAME[c].lower()}_km": None if wl is None else round(wl[c], 1) for c in CLASSES})
            row["new_km"] = None if wl is None else round(sum(wl.values()), 1)
            rows.append(row)
        row = dict(country=F5.NAMES[D], panel="National", width_km=None, height_km=None, roads_2015_km=round(km0, 1))
        row.update({f"new_{CLASS_NAME[c].lower()}_km": None if tot is None else round(tot[c], 1) for c in CLASSES})
        row["new_km"] = None if tot is None else round(sum(tot.values()), 1)
        rows.append(row)
        y = top + F5.MAP_H + F5.XLAB
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, "SFig06_new_roads_by_class" if a.ssp == 1 else f"SFig06_new_roads_by_class_SSP{a.ssp}")
    stem, n = base, 0
    while any(os.path.exists(f"{stem}{e}") for e in (".png", ".pdf", ".svg", "_lengths.csv")):
        n += 1
        stem = f"{base}_r{n}"
    for e in ("png", "pdf", "svg"):
        f.savefig(f"{stem}.{e}", dpi=F5.DPI)
    plt.close(f)
    if F5.FONT != "Arial":
        s = open(stem + ".svg", encoding="utf-8").read()
        s = s.replace(f"font-family: '{F5.FONT}', sans-serif", f"font-family: Arial, '{F5.FONT}', sans-serif").replace(
            f"font-family: '{F5.FONT}'", f"font-family: Arial, '{F5.FONT}', sans-serif")
        open(stem + ".svg", "w", encoding="utf-8").write(s)
    pd.DataFrame(rows).to_csv(stem + "_lengths.csv", index=False)
    print(f"Wrote {stem}.png, .pdf, .svg and _lengths.csv ({W:.0f} x {H:.0f} mm)")


def main():
    ap = argparse.ArgumentParser(description="Figure S6: new roads to 2050 by class in Bangladesh, Nigeria and Brazil.")
    ap.add_argument("--ssp", type=int, default=1, choices=[1, 3, 5])
    ap.add_argument("--forecast", default=os.path.join(F5.FIG_DATA, "fig5_forecast"), help="folder with <ISO3>_SSP<s>_since2015_2050_new.tif and _upgraded.tif")
    ap.add_argument("--roads", default=os.path.join(F5.FIG_DATA, "fig5_roads_2015"), help="folder with <ISO3>_roads_2015.part<n>.fgb.zip")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--save_lines", action="store_true", help="also write the traced road lines (FlatGeobuf)")
    a = ap.parse_args()
    data = {D: F5.load(D, a) for D in F5.COUNTRIES}
    build(a, data)


if __name__ == "__main__":
    main()
