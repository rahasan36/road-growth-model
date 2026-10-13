#!/usr/bin/env python3
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

WIDTH_MM = 150.0
FS_BOX = 8.0
FS_LAB = 7.0
FS_LETTER = 8.0

PX_W, PX_H = 900, 878
_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Arial", "Liberation Sans", "DejaVu Sans"] if f in _avail), "DejaVu Sans")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT], "text.color": "#1a1a1a",
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})

W_IN = WIDTH_MM / 25.4
H_IN = W_IN * PX_H / PX_W
fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, PX_W)
ax.set_ylim(PX_H, 0)
ax.set_aspect("equal")
ax.axis("off")

PT = W_IN * 72 / PX_W
LW = 1.4 * PT
LW_ARROW = 1.1 * PT
EDGE = "#333333"
ARROW = "#555555"
ORANGE = EDGE

FILL_A, FILL_B, FILL_C = "white", "white", "white"
FILL_URBAN, FILL_REL, FILL_DIAMOND = "#AAD9F4", "#E5BCD3", "#66C5AB"
DIAMOND_EDGE = EDGE


def text(x, y, s, fs=FS_BOX, weight="normal", bg=None, ls=1.35, **kw):
    t = ax.text(x, y, s, fontsize=fs, fontweight=weight, ha="center", va="center",
                linespacing=ls, zorder=10, **kw)
    if bg:
        t.set_bbox(dict(facecolor=bg, edgecolor="none", pad=1.0))
    return t


def arrow(p0, p1, z=4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                 color=ARROW, lw=LW_ARROW, shrinkA=0, shrinkB=0, zorder=z))


def line(xs, ys, z=3):
    ax.plot(xs, ys, color=ARROW, lw=LW_ARROW, zorder=z, solid_joinstyle="miter")


def process(x0, y0, x1, y1, fill, edge=EDGE, inset=17, rounding=4):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={rounding}",
                                facecolor=fill, edgecolor=edge, lw=LW, zorder=5))
    for x in (x0 + inset, x1 - inset):
        ax.plot([x, x], [y0, y1], color=edge, lw=LW, zorder=6)


def box(x0, y0, x1, y1, fill, rounding=4):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={rounding}",
                                facecolor=fill, edgecolor=EDGE, lw=LW, zorder=5))


def hexagon(cx, cy, w=178, h=130, tip=35):
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    pts = [(x0, cy), (x0 + tip, y0), (x1 - tip, y0), (x1, cy), (x1 - tip, y1), (x0 + tip, y1)]
    ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=ORANGE, lw=LW,
                         joinstyle="round", zorder=5))


def parallelogram(xl, y0, y1, w=150, slant=30):
    pts = [(xl + slant, y0), (xl + slant + w, y0), (xl + w, y1), (xl, y1)]
    ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=EDGE, lw=LW,
                         joinstyle="miter", zorder=5))


def document(x0, y0, x1, y1, z):
    xs = np.linspace(x0, x1, 60)
    wave = y1 - 8 * np.sin((xs - x0) / (x1 - x0) * 2 * np.pi)
    verts = [(x0, y0), (x1, y0)] + list(zip(xs[::-1], wave[::-1]))
    ax.add_patch(Polygon(verts, closed=True, facecolor="white", edgecolor=EDGE, lw=LW, zorder=z))


for k, (dx, dy) in enumerate([(-14, -16), (-7, -8), (0, 0)]):
    document(50 + dx, 45 + dy, 215 + dx, 152 + dy, z=5 + k)
text(132, 100, "Spatial driver\nmaps")

cx, cw, ctop, cbot, eh = 337, 158, 35, 165, 22
ax.add_patch(Rectangle((cx - cw / 2, ctop), cw, cbot - ctop, facecolor="white", edgecolor="none", zorder=5))
ax.plot([cx - cw / 2] * 2, [ctop, cbot], color=EDGE, lw=LW, zorder=6)
ax.plot([cx + cw / 2] * 2, [ctop, cbot], color=EDGE, lw=LW, zorder=6)
ax.add_patch(Ellipse((cx, cbot), cw, eh, facecolor="white", edgecolor="none", zorder=5.5))
ax.add_patch(Ellipse((cx, ctop), cw, eh, facecolor="white", edgecolor=EDGE, lw=LW, zorder=6))
from matplotlib.patches import Arc
ax.add_patch(Arc((cx, cbot), cw, eh, theta1=0, theta2=180, color=EDGE, lw=LW, zorder=6))
text(cx, 113, "Global input\nvariables")

arrow((215, 100), (cx - cw / 2, 100))

process(697, 30, 877, 163, "white", edge=ORANGE, inset=20)
text(787, 97, "Future road\nlength forecast")
arrow((697, 98), (cx + cw / 2, 98))
text(560, 98, "Future road length", fs=FS_LAB, weight="bold", bg="white", color="#444444")

ax.add_patch(FancyBboxPatch((222, 210), 223, 595, boxstyle="round,pad=0,rounding_size=8",
                            facecolor="none", edgecolor="black", lw=2.4 * PT,
                            linestyle=(0, (6, 3.5)), zorder=4))

rows = {"A": (248, 383), "B": (440, 575), "C": (646, 780)}
fills = {"A": FILL_A, "B": FILL_B, "C": FILL_C}
labels = {"A": "Radial\npattern-based\nroad growth",
          "B": "Road\nupgradation",
          "C": "New road\nnetwork\ndevelopment"}
for key, (y0, y1) in rows.items():
    process(248, y0, 428, y1, fills[key])
    text(338, (y0 + y1) / 2, labels[key])
    ax.text(251, y0 - 3, key, fontsize=FS_LETTER, fontweight="bold", ha="left", va="bottom", zorder=10)

arrow((cx, cbot + eh / 2), (cx, 248))
text(cx, 195, "Spatial distribution", fs=FS_LAB, weight="bold", bg="white", color="#444444")
arrow((cx, 383), (cx, 440))
text(cx, 410, "Updated existing roads\n+ pattern-based growth", fs=6.0, weight="bold", bg="white",
     color="#444444")
arrow((cx, 575), (cx, 646))
text(cx, 601, "Total forecasted length \u2212\nlength of (pattern-based\ngrowth + upgraded roads)",
     fs=6.0, weight="bold", bg="white", color="#444444", ls=1.2)

hex_text = {"A": "Existing radial\npattern\nidentification",
            "B": "Upgrade\nlocation & amount\ndetermination",
            "C": "Growth amount &\ndirection"}
hex_h = {"A": 135, "B": 132, "C": 122}
for key, (y0, y1) in rows.items():
    cy = (y0 + y1) / 2
    hexagon(111, cy, h=hex_h[key])
    text(111, cy, hex_text[key])
    arrow((200, cy), (248, cy))

par_text = {"A": "New radial\npattern-based\nroad growth",
            "B": "Upgraded roads\n(new category)",
            "C": "New\nhierarchical\nlinear growth"}
for key, (y0, y1) in rows.items():
    cy = (y0 + y1) / 2
    parallelogram(475, y0 + 2, y1 + 2)
    text(565, cy, par_text[key])
    arrow((428, cy), (490, cy))

yA, yB, yC = [(a + b) / 2 for a, b in rows.values()]
line([640, 668, 668], [yA, yA, yB])
line([640, 668, 668], [yC, yC, yB])
arrow((640, yB), (697, yB))

dcx, dcy, dw, dh = 787, 507, 92, 68
ax.add_patch(Polygon([(dcx - dw, dcy), (dcx, dcy - dh), (dcx + dw, dcy), (dcx, dcy + dh)], closed=True,
                     facecolor=FILL_DIAMOND, edgecolor=DIAMOND_EDGE, lw=LW, joinstyle="miter", zorder=5))
text(dcx, dcy + 2, "Final\nforecasted road\nnetwork")

box(697, 248, 877, 383, FILL_URBAN)
text(787, 316, "Urban land change\nmodel")
arrow((dcx, dcy - dh), (dcx, 383))
text(dcx, 412, "Integration", fs=FS_LAB, weight="bold", bg="white", color="#444444")
arrow((dcx, 248), (dcx, 163))
text(dcx, 207, "Future urban area forecast", fs=FS_LAB, weight="bold", bg="white", color="#444444")

box(697, 646, 877, 780, FILL_REL)
text(787, 713, "Reliability\nassessment")
arrow((dcx, dcy + dh), (dcx, 646))
text(dcx, 605, "Centrality measure\nStructural & topological\ncharacteristics", fs=6.5,
     bg="white", color="#333333")

key = [(FILL_URBAN, "External to the road growth model"),
       (FILL_DIAMOND, "Road growth model output"),
       (FILL_REL, "Post-processing")]
ky = 843
widths = [ax.text(0, 0, lab, fontsize=FS_LAB).get_window_extent(fig.canvas.get_renderer()).width
          for _, lab in key]
for t in list(ax.texts)[-len(key):]:
    t.remove()
px_per_disp = PX_W / fig.bbox.width
sw, gap_txt, gap_item = 26, 7, 34
total = sum(sw + gap_txt + w * px_per_disp for w in widths) + gap_item * (len(key) - 1)
x = (PX_W - total) / 2
for (col, lab), wd in zip(key, widths):
    ax.add_patch(Rectangle((x, ky - 9), sw, 18, facecolor=col, edgecolor=EDGE, lw=LW, zorder=5))
    ax.text(x + sw + gap_txt, ky, lab, fontsize=FS_LAB, ha="left", va="center", zorder=10)
    x += sw + gap_txt + wd * px_per_disp + gap_item

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig1_framework")
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{out}.{ext}", dpi=600, facecolor="white")
if FONT != "Arial":
    with open(out + ".svg", encoding="utf-8") as f:
        svg = f.read()
    svg = svg.replace(f"font-family: '{FONT}', sans-serif", f"font-family: Arial, '{FONT}', sans-serif")
    svg = svg.replace(f"font-family: '{FONT}'", f"font-family: Arial, '{FONT}', sans-serif")
    with open(out + ".svg", "w", encoding="utf-8") as f:
        f.write(svg)
print(f"Figure {WIDTH_MM:.0f} x {WIDTH_MM * PX_H / PX_W:.0f} mm. Saved: {out}.{{png,pdf,svg}}")
