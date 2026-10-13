import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.transforms import Bbox, ScaledTranslation

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1 / 25.4

_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Arial", "Liberation Sans", "DejaVu Sans"] if f in _avail), "DejaVu Sans")


def set_style(pt):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT],
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": FONT + ":italic",
        "mathtext.bf": FONT + ":bold",
        "font.size": pt,
        "axes.labelsize": pt,
        "xtick.labelsize": pt,
        "ytick.labelsize": pt,
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
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


POINT_COLOR = "black"
POINT_ALPHA = 0.35
LABELLED_ALPHA = 0.9
LINE_COLOR = "#d62728"
LEADER_COLOR = "0.45"
XLIM = (0.6, 7.2)

CLASSES = [
    ("all", "All roads", "Road_Length", True),
    ("main", "Main roads", "Main", False),
    ("secondary", "Secondary roads", "Secondary", False),
    ("tertiary", "Tertiary roads", "Tertiary", False),
    ("local", "Local roads", "Local", False),
]

DISPLAY = {
    "Tanzania, United Republic of": "Tanzania",
    "Bahamas, The": "Bahamas",
    "Gambia, The": "Gambia",
    "Pacific Islands (Palau)": "Palau",
}


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
    sse = ((yy - fit) ** 2).sum()
    sst = ((yy - yy.mean()) ** 2).sum()
    return dict(ok=ok, fit=fit, obs=yy, resid=yy - fit, n=len(yy), r2=1 - sse / sst)


def fit_all(d):
    out = {}
    for key, label, col, log_urban in CLASSES:
        y = np.log10(d[col].astype(float) + 1).values
        m = ols(y, design(d, log_urban))
        m["label"] = label
        names = d["CNTRY_NAME"].astype(str).values[m["ok"]]
        m["country"] = np.array([DISPLAY.get(c, c) for c in names])
        out[key] = m
    return out


def _overlaps(b, boxes):
    for o in boxes:
        if b.x0 < o.x1 and b.x1 > o.x0 and b.y0 < o.y1 and b.y1 > o.y0:
            return True
    return False


def _seg_hits(p, q, boxes):
    for t in np.linspace(0.15, 0.85, 8):
        x = p[0] + t * (q[0] - p[0])
        y = p[1] + t * (q[1] - p[1])
        for o in boxes:
            if o.x0 <= x <= o.x1 and o.y0 <= y <= o.y1:
                return True
    return False


def _cross(p1, p2, q1, q2):
    def o(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = o(q1, q2, p1), o(q1, q2, p2)
    d3, d4 = o(p1, p2, q1), o(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def place_labels(ax, xs, ys, names, order, fontsize, max_labels, keep_out):
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    to_disp = ax.transData.transform
    axbox = ax.get_window_extent(r)
    pad = 2 * fig.dpi / 72
    inner = Bbox.from_extents(axbox.x0 + pad, axbox.y0 + pad, axbox.x1 - pad, axbox.y1 - pad)
    pts = to_disp(np.column_stack([xs, ys]))
    pr = 2.0 * fig.dpi / 72
    point_boxes = [Bbox.from_extents(x - pr, y - pr, x + pr, y + pr) for x, y in pts]
    placed = list(keep_out)
    chosen = []
    unit = fig.dpi / 72
    radii = [2, 4, 7, 11, 16, 22]
    leaders = []
    angles = np.deg2rad([0, 180, 30, 150, 330, 210, 60, 120, 300, 240, 90, 270,
                         15, 165, 345, 195, 45, 135, 315, 225, 75, 105, 285, 255])
    for i in order:
        if len(chosen) >= max_labels:
            break
        t = ax.text(0, 0, names[i], fontsize=fontsize, transform=None)
        bb = t.get_window_extent(r)
        t.remove()
        w, h = bb.width, bb.height
        px, py = pts[i]
        others = point_boxes
        done = False
        for rad in radii:
            for a in angles:
                dx, dy = np.cos(a), np.sin(a)
                cx = px + dx * (rad * unit + w / 2)
                cy = py + dy * (rad * unit + h / 2)
                b = Bbox.from_extents(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if b.x0 < inner.x0 or b.x1 > inner.x1 or b.y0 < inner.y0 or b.y1 > inner.y1:
                    continue
                if _overlaps(b, placed) or _overlaps(b, others):
                    continue
                far = rad > 4
                ex = min(max(px, b.x0), b.x1)
                ey = min(max(py, b.y0), b.y1)
                if far and (_seg_hits((px, py), (ex, ey), placed) or
                            any(_cross((px, py), (ex, ey), u, v) for u, v in leaders)):
                    continue
                if any(_seg_hits(u, v, [b]) for u, v in leaders):
                    continue
                placed.append(Bbox.from_extents(b.x0 - unit, b.y0 - unit, b.x1 + unit, b.y1 + unit))
                inv = ax.transData.inverted()
                lx, ly = inv.transform((cx, cy))
                ax.text(lx, ly, names[i], fontsize=fontsize, ha="center", va="center", zorder=5)
                if far:
                    leaders.append(((px, py), (ex, ey)))
                    lx2, ly2 = inv.transform((ex, ey))
                    ax.plot([xs[i], lx2], [ys[i], ly2], color=LEADER_COLOR, lw=0.4, zorder=3)
                chosen.append(i)
                done = True
                break
            if done:
                break
    return chosen


def pick_key_corner(ax, xs, ys, text, fontsize):
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    best = None
    for (hx, hy, ha, va) in [(0.03, 0.97, "left", "top"), (0.97, 0.97, "right", "top"),
                             (0.03, 0.03, "left", "bottom"), (0.97, 0.03, "right", "bottom")]:
        t = ax.text(hx, hy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize)
        bb = t.get_window_extent(r)
        t.remove()
        grow = 3 * fig.dpi / 72
        n = np.sum((pts[:, 0] > bb.x0 - grow) & (pts[:, 0] < bb.x1 + grow) &
                   (pts[:, 1] > bb.y0 - grow) & (pts[:, 1] < bb.y1 + grow))
        if best is None or n < best[0]:
            best = (n, hx, hy, ha, va)
    _, hx, hy, ha, va = best
    t = ax.text(hx, hy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize, zorder=6)
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    g = 2 * fig.dpi / 72
    return Bbox.from_extents(bb.x0 - g, bb.y0 - g, bb.x1 + g, bb.y1 + g)


def base_panel(ax, m, letter, ylabel, xlabel, pt):
    ax.axhline(0, color=LINE_COLOR, lw=1.0, zorder=1)
    ax.scatter(m["fit"], m["resid"], s=7, color=POINT_COLOR, lw=0, alpha=POINT_ALPHA, zorder=2)
    ax.set_xlim(XLIM)
    lo, hi = m["resid"].min(), m["resid"].max()
    span = hi - lo
    ax.set_ylim(lo - 0.10 * span, hi + 0.10 * span)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    off = ScaledTranslation(0, 2 / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, letter, transform=ax.transAxes + off, ha="left", va="bottom",
            fontsize=pt, fontweight="bold")
    ax.text(1, 1, m["label"], transform=ax.transAxes + off, ha="right", va="bottom",
            fontsize=pt, fontweight="bold")


def make_figure(models, out_base, pt, max_labels):
    fig = plt.figure(figsize=(180 * MM, 128 * MM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.36, 0.64],
                  left=0.075, right=0.995, bottom=0.075, top=0.965, wspace=0.13)
    ax_a = fig.add_subplot(gs[0, 0])
    sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 1], wspace=0.22, hspace=0.20)
    axes = [fig.add_subplot(sub[i, j]) for i in range(2) for j in range(2)]
    panels = [(ax_a, "all", "a", "Residual", "Predicted")]
    for ax, key, letter, (rr, cc) in zip(axes, ["main", "secondary", "tertiary", "local"],
                                         "bcde", [(0, 0), (0, 1), (1, 0), (1, 1)]):
        panels.append((ax, key, letter, "Residual" if cc == 0 else None,
                       "Predicted" if rr == 1 else None))
    for ax, key, letter, yl, xl in panels:
        base_panel(ax, models[key], letter, yl, xl, pt)

    labelled = []
    for ax, key, letter, yl, xl in panels:
        m = models[key]
        sd = m["resid"].std(ddof=1)
        order = [i for i in np.argsort(-np.abs(m["resid"])) if abs(m["resid"][i]) > sd]
        cap = max_labels * (2 if key == "all" else 1)
        idx = place_labels(ax, m["fit"], m["resid"], m["country"], order, pt, cap, [])
        ax.scatter(m["fit"][idx], m["resid"][idx], s=7, color=POINT_COLOR, lw=0,
                   alpha=LABELLED_ALPHA, zorder=4)
        for i in idx:
            labelled.append(dict(road_class=m["label"], country=m["country"][i],
                                 predicted_log10=m["fit"][i], residual=m["resid"][i]))
        print(f"  {m['label']:<16} labelled {len(idx)} of {m['n']}")

    fig.savefig(out_base + ".png", dpi=600)
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".svg")
    if FONT != "Arial":
        with open(out_base + ".svg", encoding="utf-8") as f:
            svg = f.read()
        svg = svg.replace(f"font-family: '{FONT}', sans-serif", f"font-family: Arial, '{FONT}', sans-serif")
        svg = svg.replace(f"font-family: '{FONT}'", f"font-family: Arial, '{FONT}', sans-serif")
        with open(out_base + ".svg", "w", encoding="utf-8") as f:
            f.write(svg)
    plt.close(fig)
    return labelled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "RoadForecast_urban.xlsx"))
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--font-pt", type=float, default=7.0,
                    help="text size in pt at 180 mm (Nature artwork guide: 5-7 pt)")
    ap.add_argument("--max-labels", type=int, default=25,
                    help="most call-outs per class panel (panel a gets twice this)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    set_style(a.font_pt)

    d = pd.read_excel(a.data)
    models = fit_all(d)
    rows = []
    for m in models.values():
        for c, p, o, r in zip(m["country"], m["fit"], m["obs"], m["resid"]):
            rows.append(dict(road_class=m["label"], country=c, predicted_log10=p,
                             observed_log10=o, residual=r))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "SFig04_residuals.csv"), index=False)
    labelled = make_figure(models, os.path.join(a.out, "SFig04_residuals"), a.font_pt, a.max_labels)
    pd.DataFrame(labelled).to_csv(os.path.join(a.out, "SFig04_labelled_countries.csv"), index=False)
    print("Wrote figure and tables to", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
