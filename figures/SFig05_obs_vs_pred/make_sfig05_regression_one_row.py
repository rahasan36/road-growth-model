import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1 / 25.4

_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Arial", "Liberation Sans", "DejaVu Sans"] if f in _avail), "DejaVu Sans")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT],
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": FONT + ":italic",
    "mathtext.bf": FONT + ":bold",
    "mathtext.default": "it",
    "svg.fonttype": "none",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.edgecolor": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})
SHOW_GRID = False
POINT_COLOR = "black"
POINT_ALPHA = 0.35
LINE_COLOR = "#d62728"
LIMS = (-0.4, 7.1)
TICKS = range(0, 8)

CLASSES = [
    ("all", "All roads", "Road_Length", True),
    ("main", "Main roads", "Main", False),
    ("secondary", "Secondary roads", "Secondary", False),
    ("tertiary", "Tertiary roads", "Tertiary", False),
    ("local", "Local roads", "Local", False),
]


def design(d, log_urban):
    X = pd.DataFrame({
        "log10_AREA2": np.log10(d["AREA2"]),
        "log10_Density": np.log10(d["Density"]),
        "log10_GDPC": np.log10(d["GDPC"]),
        "OECD_Membe": d["OECD_Membe"].astype(float),
    })
    if log_urban:
        X["log10_UR_PROP"] = np.log10(d["UR_PROP"])
        X["log10_UrbAmt2020"] = np.log10(d["UrbAmt2020"] + 1)
    else:
        X["UR_PROP"] = d["UR_PROP"].astype(float)
        X["UrbAmt2020"] = d["UrbAmt2020"].astype(float)
    return X


def ols(y, X):
    ok = np.isfinite(y) & np.all(np.isfinite(X.values), axis=1)
    yy = y[ok]
    A = np.column_stack([np.ones(ok.sum()), X.values[ok]])
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    fit = A @ beta
    resid = yy - fit
    n, k = A.shape
    sse = resid @ resid
    sst = ((yy - yy.mean()) ** 2).sum()
    r2 = 1 - sse / sst
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    sigma2 = sse / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(A.T @ A)))
    return dict(ok=ok, beta=beta, se=se, fit=fit, obs=yy, n=n, r2=r2, adj=adj,
                names=["Intercept"] + list(X.columns))


def fit_all(d):
    out = {}
    for key, label, col, log_urban in CLASSES:
        y = np.log10(d[col].astype(float) + 1).values
        out[key] = ols(y, design(d, log_urban))
        out[key]["label"] = label
        out[key]["country"] = d["CNTRY_NAME"].values[out[key]["ok"]]
    return out


def panel(ax, m, letter, xlabel, ylabel, label_top):
    ax.plot((0, 7), (0, 7), color=LINE_COLOR, lw=1.0, zorder=1)
    ax.scatter(m["obs"], m["fit"], s=7, color=POINT_COLOR, lw=0, alpha=POINT_ALPHA, zorder=2)
    ax.set_xlim(LIMS)
    ax.set_ylim(LIMS)
    ax.set_xticks(list(TICKS))
    ax.set_yticks(list(TICKS))
    ax.set_aspect("equal")
    if SHOW_GRID:
        ax.grid(color="0.9", lw=0.5, zorder=0)
    ax.set_xlabel(xlabel if xlabel else "")
    ax.set_ylabel(ylabel if ylabel else "")
    if not xlabel:
        ax.tick_params(labelbottom=True)
    ax.text(0.04, 0.96, f"{m['label']}\n$R$\u00b2 = {m['r2']:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    from matplotlib.transforms import ScaledTranslation
    off = ScaledTranslation(0 / 72, 2 / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, letter, transform=ax.transAxes + off, ha="left", va="bottom",
            fontsize=8, fontweight="bold")
    if label_top > 0:
        res = np.abs(m["obs"] - m["fit"])
        for i in np.argsort(res)[::-1][:label_top]:
            ax.annotate(m["country"][i], (m["obs"][i], m["fit"][i]),
                        xytext=(3, -3), textcoords="offset points",
                        fontsize=8, ha="left", va="top")


def make_figure(models, out_base, label_top):
    fig, axes = plt.subplots(1, 5, figsize=(183 * MM, 46 * MM), sharey=True,
                             gridspec_kw=dict(left=0.06, right=0.995, bottom=0.21,
                                              top=0.93, wspace=0.10))
    keys = ["all", "main", "secondary", "tertiary", "local"]
    for i, (ax, key, letter) in enumerate(zip(axes, keys, "abcde")):
        panel(ax, models[key], letter, "Observed",
              "Predicted" if i == 0 else None, label_top)
        if i > 0:
            ax.tick_params(labelleft=False)
    fig.savefig(out_base + ".png", dpi=600)
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".svg")
    if FONT != "Arial":
        with open(out_base + ".svg", encoding="utf-8") as f:
            svg = f.read()
        svg = svg.replace("font-family: 'Liberation Sans', sans-serif", "font-family: Arial, 'Liberation Sans', sans-serif")
        svg = svg.replace("font-family: 'Liberation Sans'", "font-family: Arial, 'Liberation Sans', sans-serif")
        with open(out_base + ".svg", "w", encoding="utf-8") as f:
            f.write(svg)
    plt.close(fig)


def write_tables(models, out_dir):
    rows = []
    for key, m in models.items():
        for c, o, p in zip(m["country"], m["obs"], m["fit"]):
            rows.append(dict(road_class=m["label"], country=c,
                             observed_log10=o, predicted_log10=p, residual=o - p))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "SFig05_fitted_values_one_row.csv"), index=False)

    rows = []
    for key, m in models.items():
        for name, b, s in zip(m["names"], m["beta"], m["se"]):
            rows.append(dict(road_class=m["label"], term=name, estimate=b, std_error=s,
                             n=m["n"], r2=m["r2"], adj_r2=m["adj"]))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "SFig05_model_summary_one_row.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "RoadForecast_urban.xlsx"))
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--label-top", type=int, default=0,
                    help="label the N countries with the largest residual in each panel")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = pd.read_excel(a.data)
    models = fit_all(d)
    for m in models.values():
        print(f"{m['label']:<16} n={m['n']:<4d} R2={m['r2']:.4f}  adj R2={m['adj']:.4f}")
    write_tables(models, a.out)
    make_figure(models, os.path.join(a.out, "SFig05_obs_vs_pred_one_row"), a.label_top)
    print("Wrote figure and tables to", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
