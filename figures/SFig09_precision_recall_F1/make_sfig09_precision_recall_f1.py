import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DATA = os.path.join(REPO, "data", "figures")
MM = 1 / 25.4
PT = 8.0

BUFFERS = [0, 300, 600, 900]
COUNTRIES = [("BGD", "Bangladesh"), ("BRA", "Brazil"), ("NGA", "Nigeria")]
METRICS = ["Precision", "Recall", "F1"]
SERIES = [("forecast", "GRIP-based forecast"), ("osm15", "OSM 2015 baseline")]

DATA = {
    "Bangladesh": {
        "forecast": {"Precision": [0.802, 0.980, 0.992, 0.996],
                     "Recall":    [0.651, 0.897, 0.938, 0.956],
                     "F1":        [0.719, 0.937, 0.964, 0.975]},
        "osm15":    {"Precision": [0.465, 0.700, 0.793, 0.858],
                     "Recall":    [0.563, 0.831, 0.901, 0.932],
                     "F1":        [0.509, 0.760, 0.843, 0.893]}},
    "Brazil": {
        "forecast": {"Precision": [0.676, 0.972, 0.986, 0.989],
                     "Recall":    [0.165, 0.339, 0.412, 0.475],
                     "F1":        [0.265, 0.503, 0.581, 0.642]},
        "osm15":    {"Precision": [0.410, 0.680, 0.791, 0.847],
                     "Recall":    [0.109, 0.188, 0.232, 0.266],
                     "F1":        [0.172, 0.295, 0.359, 0.405]}},
    "Nigeria": {
        "forecast": {"Precision": [0.659, 0.943, 0.966, 0.978],
                     "Recall":    [0.133, 0.272, 0.337, 0.400],
                     "F1":        [0.222, 0.422, 0.500, 0.567]},
        "osm15":    {"Precision": [0.385, 0.611, 0.717, 0.785],
                     "Recall":    [0.083, 0.138, 0.170, 0.198],
                     "F1":        [0.137, 0.225, 0.275, 0.316]}},
}

STYLE = {
    "forecast": dict(color="#0072b2", ls="-", marker="o"),
    "osm15":    dict(color="#d55e00", ls="--", marker="s"),
}

_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Arial", "Liberation Sans", "DejaVu Sans"] if f in _avail), "DejaVu Sans")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": PT,
    "axes.labelsize": PT, "xtick.labelsize": PT, "ytick.labelsize": PT, "legend.fontsize": PT,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": "black", "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})


def read_metrics(metrics_dir):
    out = {}
    for code, name in COUNTRIES:
        out[name] = {}
        for key, suffix in [("forecast", ""), ("osm15", "_OSM15")]:
            f = os.path.join(metrics_dir, f"{code}_extended_metrics{suffix}_20260612.csv")
            d = pd.read_csv(f)
            out[name][key] = {m: d[m].astype(float).tolist() for m in METRICS}
    return out


def fmt(v):
    return f"{v:.3f}" if v >= 0.995 else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", default=os.path.join(FIG_DATA, "validation", "metrics"),
                    help="folder with <country>_extended_metrics*_20260612.csv; pass \"\" to use the values in this script")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = read_metrics(a.metrics_dir) if a.metrics_dir else DATA

    rows = []
    for _, cn in COUNTRIES:
        for key, lab in SERIES:
            for m in METRICS:
                for b, v in zip(BUFFERS, data[cn][key][m]):
                    rows.append(dict(country=cn, series=lab, metric=m, buffer_m=b, value=v))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "SFig09_source_data.csv"), index=False)

    fig, axes = plt.subplots(3, 3, figsize=(180 * MM, 150 * MM), sharex=True, sharey=True,
                             gridspec_kw=dict(left=0.085, right=0.99, bottom=0.075, top=0.925,
                                              wspace=0.10, hspace=0.20))
    letters = iter("abcdefghi")
    for r, (_, cn) in enumerate(COUNTRIES):
        for c, m in enumerate(METRICS):
            ax = axes[r, c]
            for key, lab in SERIES:
                st = STYLE[key]
                y = data[cn][key][m]
                ax.plot(BUFFERS, y, color=st["color"], ls=st["ls"], lw=1.2,
                        marker=st["marker"], ms=4.0, mfc="white", mew=1.0, mec=st["color"],
                        zorder=3, label=lab)
                for b, v in zip(BUFFERS, y):
                    up = key == "forecast"
                    ax.annotate(fmt(v), (b, v), textcoords="offset points",
                                xytext=(0, 4 if up else -4), ha="center",
                                va="bottom" if up else "top", fontsize=PT, zorder=4)
            ax.set_xlim(-90, 990)
            ax.set_xticks(BUFFERS)
            ax.set_ylim(-0.10, 1.12)
            ax.set_yticks(np.arange(0, 1.01, 0.2), [f"{t:.1f}" for t in np.arange(0, 1.01, 0.2)])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            off = ScaledTranslation(0, 3 / 72, fig.dpi_scale_trans)
            ax.text(0, 1, next(letters), transform=ax.transAxes + off, ha="left", va="bottom",
                    fontsize=PT, fontweight="bold")
            if r == 0:
                ax.text(0.5, 1, m, transform=ax.transAxes + off, ha="center", va="bottom",
                        fontsize=PT, fontweight="bold")
            if c == 0:
                ax.set_ylabel(cn, fontweight="bold")
            if r == 2:
                ax.set_xlabel("Buffer distance (m)")

    handles = [Line2D([0], [0], color=STYLE[k]["color"], ls=STYLE[k]["ls"], lw=1.2,
                      marker=STYLE[k]["marker"], ms=4.0, mfc="white", mew=1.0,
                      mec=STYLE[k]["color"], label=lab) for k, lab in SERIES]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.995), handlelength=2.6, columnspacing=2.0)

    base = os.path.join(a.out, "SFig09_precision_recall_F1")
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
