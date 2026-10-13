import argparse
import os

import matplotlib
matplotlib.use("Agg")
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
C_TP, C_FP, C_FN = "#0072b2", "#d55e00", "#000000"
C_EDGE = "#7f7f7f"

REGIONS = [
    ("a", "Padma Bridge corridor", 23.46, 90.26, 20),
    ("b", "Rangpur\u2013Dinajpur", 26.04, 89.55, 25),
    ("c", "Chittagong Hill Tracts", 23.50, 92.20, 30),
]

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


def degree_ticks(ax, tf, step=0.1):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = np.linspace(x0, x1, 300)
    lon, _ = tf.transform(xs, np.full_like(xs, y0))
    ys = np.linspace(y0, y1, 300)
    _, lat = tf.transform(np.full_like(ys, x0), ys)
    for xstep in (step, 2 * step, 5 * step):
        lv = np.arange(np.ceil(lon.min() / xstep) * xstep, lon.max(), xstep)
        lv = lv[(lv > lon.min() + 0.05 * np.ptp(lon)) & (lv < lon.max() - 0.05 * np.ptp(lon))]
        if len(lv) <= 3:
            break
    pv = np.arange(np.ceil(lat.min() / step) * step, lat.max(), step)
    pv = pv[(pv > lat.min() + 0.05 * np.ptp(lat)) & (pv < lat.max() - 0.05 * np.ptp(lat))]
    o = np.argsort(lon)
    ax.set_xticks(np.interp(lv, lon[o], xs[o]), [f"{v:.1f}\u00b0E" for v in lv])
    o = np.argsort(lat)
    ax.set_yticks(np.interp(pv, lat[o], ys[o]), [f"{v:.1f}\u00b0N" for v in pv])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpfpfn", default=os.path.join(FIG_DATA, "validation", "BGD_spatial_TPFPFN_300m_20260607.tif"))
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with rasterio.open(a.tpfpfn) as src:
        cls, T, crs, (H0, W0) = src.read(1), src.transform, src.crs, src.shape
    px = abs(T.a)
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    panels, rows = [], []
    for letter, name, lat, lon, rad_km in REGIONS:
        x, y = to_proj.transform(lon, lat)
        c, r = ~T * (x, y)
        c, r = int(c), int(r)
        R = int(rad_km * 1000 / px)
        r0, r1, c0, c1 = max(0, r - R), min(H0, r + R), max(0, c - R), min(W0, c + R)
        crop = cls[r0:r1, c0:c1]
        ext = (T.c + c0 * T.a, T.c + c1 * T.a, T.f + r1 * T.e, T.f + r0 * T.e)
        tp, fp, fn = (int((crop == k).sum()) for k in (1, 2, 3))
        print(f"{name}: TP {tp:,}  commission {fp:,}  omission {fn:,}")
        rows.append(dict(panel=letter, region=name, centre_lat=lat, centre_lon=lon, radius_km=rad_km,
                         true_positive=tp, commission=fp, omission=fn))
        panels.append((letter, name, crop, ext, tp, fp, fn))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "SFig11_case_studies_counts.csv"), index=False)

    W = 180.0
    left, gap, right = 13.0, 13.5, 1.0
    mw = (W - left - right - 2 * gap) / 3
    top, tick_h, leg_h = 6.0, 7.0, 13.0
    H = top + mw + tick_h + leg_h
    fig = plt.figure(figsize=(W * MM, H * MM))
    print(f"figure {W:.0f} x {H:.0f} mm")
    cmap = ListedColormap(["white", C_TP, C_FP, C_FN])
    for k, (letter, name, crop, ext, tp, fp, fn) in enumerate(panels):
        ax = fig.add_axes([(left + k * (mw + gap)) / W, (tick_h + leg_h) / H, mw / W, mw / H])
        ax.imshow(crop, cmap=cmap, vmin=0, vmax=3, interpolation="nearest", extent=ext, zorder=1)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        degree_ticks(ax, to_geo)
        ax.text(0, 1.015, letter, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=PT, fontweight="bold")
        ax.text(1, 1.015, name, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=PT, fontweight="bold")
        L = 10000
        sx = ext[1] - 0.04 * (ext[1] - ext[0]) - L
        sy = ext[2] + 0.05 * (ext[3] - ext[2])
        span = ext[3] - ext[2]
        ax.add_patch(Rectangle((sx - 0.02 * span, sy - 0.025 * span), L + 0.04 * span, 0.11 * span,
                               facecolor="white", edgecolor="none", alpha=0.9, zorder=6))
        ax.add_patch(Rectangle((sx, sy), L, 0.012 * span, color="black", lw=0, zorder=7))
        ax.text(sx + L / 2, sy + 0.025 * span, "10 km", ha="center", va="bottom", fontsize=PT, zorder=7)
        handles = [Patch(facecolor=C_TP, label=f"True positive ({tp:,})"),
                   Patch(facecolor=C_FP, label=f"Commission ({fp:,})"),
                   Patch(facecolor=C_FN, label=f"Omission ({fn:,})")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.11), frameon=False,
                  handlelength=1.0, handleheight=0.8, handletextpad=0.4, labelspacing=0.2,
                  borderaxespad=0.0)

    base = os.path.join(a.out, "SFig11_case_studies")
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
