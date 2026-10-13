#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, PathPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path
from matplotlib import font_manager
import matplotlib.patheffects as pe

SANS = next((f for f in ["Arial", "Liberation Sans", "DejaVu Sans"]
             if f in {x.name for x in font_manager.fontManager.ttflist}), "DejaVu Sans")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [SANS],
    "mathtext.fontset": "custom",
    "mathtext.rm": SANS,
    "mathtext.it": f"{SANS}:italic",
    "mathtext.bf": f"{SANS}:bold",
    "mathtext.sf": SANS,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "text.color": "black",
})
FS = 8.0
LETTER_FS = 14.0

FIG_W_IN = 180 / 25.4
A_X, A_Y = (20, 1766), (52, 1080)
B_X, B_Y = (20, 1530), (20, 930)
GAP_IN = 0.12
B_SCALE = 0.86

ka = FIG_W_IN / (A_X[1] - A_X[0])
kb = B_SCALE * FIG_W_IN / (B_X[1] - B_X[0])
ha_in = ka * (A_Y[1] - A_Y[0])
hb_in = kb * (B_Y[1] - B_Y[0])
FIG_H_IN = ha_in + GAP_IN + hb_in

fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
ax_a = fig.add_axes([0, (hb_in + GAP_IN) / FIG_H_IN, 1, ha_in / FIG_H_IN])
ax_b = fig.add_axes([(1 - B_SCALE) / 2, 0, B_SCALE, hb_in / FIG_H_IN])
for ax, (x0, x1), (y0, y1) in [(ax_a, A_X, A_Y), (ax_b, B_X, B_Y)]:
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.axis("off")

VERMILLION, ORANGE, YELLOW, BLUE, SKY, BLACK, GREY = (
    "#D55E00", "#E69F00", "#F0E442", "#0072B2", "#56B4E9", "#000000", "#8C8C8C")
SEQ = ["#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5"]


def draw_pattern(ax, k):
    PT = k * 72
    STYLE = {
        "o1":   (SEQ[0], "#434343", 11.0, 1.5),
        "o2":   (SEQ[1], "#434343", 11.0, 1.5),
        "o3":   (SEQ[2], "#434343", 11.0, 1.5),
        "o4":   (SEQ[3], "#08306B", 11.0, 1.5),
        "rad":  (BLACK, BLACK, 11.0, 1.5),
        "yel":  (YELLOW, "#6C4F00", 11.0, 1.5),
        "used": (VERMILLION, BLACK, 13.5, 3.0),
        "skip": ("#BDBDBD", VERMILLION, 13.5, 3.0),
    }
    NODES = [
        ("o1", 445.6, 195.4), ("o1", 353.8, 216.9), ("o1", 1055.8, 285.4), ("o1", 191.6, 411.9),
        ("o1", 1518.7, 437.7), ("o1", 224.2, 511.4), ("o1", 1591.0, 590.8), ("o1", 191.6, 744.5),
        ("o1", 392.8, 812.5), ("o1", 602.5, 892.1), ("o1", 1593.0, 914.9),
        ("o2", 578.5, 129.4), ("o2", 970.3, 158.3), ("o2", 347.7, 367.3), ("o2", 828.0, 395.0),
        ("o2", 1124.6, 469.7), ("o2", 315.7, 642.6), ("o2", 1416.3, 819.7),
        ("o3", 718.4, 124.6), ("o3", 843.7, 159.5), ("o3", 966.7, 390.8), ("o3", 1224.0, 404.1),
        ("o3", 583.4, 437.8), ("o3", 310.3, 488.4), ("o3", 1050.4, 560.7), ("o3", 467.5, 568.0),
        ("o3", 950.5, 625.1), ("o3", 484.5, 705.3), ("o3", 877.5, 725.2), ("o3", 603.8, 782.4),
        ("o3", 976.3, 834.3),
        ("o4", 623.8, 224.5), ("o4", 655.1, 328.3), ("o4", 1292.2, 588.3), ("o4", 719.6, 645.6),
        ("rad", 723.2, 433.7), ("rad", 793.0, 546.3), ("rad", 601.4, 569.2), ("rad", 1327.8, 665.3),
        ("rad", 566.5, 676.5), ("rad", 1212.5, 677.5), ("rad", 1073.4, 735.5), ("rad", 1276.3, 823.6),
        ("used", 668.3, 561.3), ("used", 1298.7, 737.3),
        ("skip", 469.5, 462.5), ("skip", 934.7, 514.4), ("skip", 1077.0, 654.0), ("skip", 1049.3, 839.6),
        ("yel", 762.9, 219.7), ("yel", 511.5, 296.4),
        ("o4", 1080.0, 447.0),
    ]
    EDGES = [
        (1, 50), (0, 50), (11, 50), (50, 31), (50, 13), (13, 3), (50, 22), (11, 31), (31, 18),
        (31, 32), (18, 19), (18, 49), (19, 49), (19, 12), (49, 32), (49, 35), (49, 24), (32, 22),
        (32, 35), (23, 35), (5, 23), (23, 25), (25, 37), (22, 37), (37, 36), (35, 37), (35, 36),
        (35, 14), (14, 20), (36, 51), (36, 26), (36, 28), (36, 34), (20, 2), (20, 51), (51, 21),
        (51, 15), (12, 21), (21, 33), (33, 4), (33, 38), (33, 40), (38, 6), (38, 40), (38, 42),
        (38, 17), (40, 41), (40, 42), (26, 40), (24, 41), (24, 38), (26, 41), (15, 41), (41, 42),
        (41, 30), (30, 42), (42, 17), (42, 10), (30, 28), (25, 39), (37, 39), (39, 34), (39, 27),
        (39, 29), (27, 16), (16, 7), (27, 8), (34, 29), (34, 28), (29, 9),
    ]
    RADII = [((670.1, 565.5), 274.8), ((1298.2, 736.6), 274.8)]
    BOX_FILL, BOX_EDGE = "#F8F8F8", "#D9D9D9"

    def node(x, y, kind, r=None, lw=None, z=5):
        face, edge, r0, lw0 = STYLE[kind]
        ax.add_patch(Circle((x, y), r or r0, facecolor=face, edgecolor=edge,
                            lw=(lw or lw0) * PT, zorder=z))

    for i, j in EDGES:
        (_, x0, y0), (_, x1, y1) = NODES[i], NODES[j]
        ax.plot([x0, x1], [y0, y1], color=GREY, lw=1.0 * PT, zorder=1, solid_capstyle="round")
    for (cx, cy), r in RADII:
        ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=VERMILLION,
                            lw=2.5 * PT, linestyle=(0, (6, 3.5)), zorder=2))
    for kind, x, y in NODES:
        node(x, y, kind)

    def box(x0, y0, x1, y1):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="square,pad=0",
                                    facecolor=BOX_FILL, edgecolor=BOX_EDGE, lw=1.5 * PT, zorder=6))

    def text(x, base, s):
        ax.text(x, base, s, fontsize=FS, ha="left", va="baseline", zorder=8)

    box(1085.5, 69.5, 1725.0, 258.5)
    node(1124.5, 98.5, "used", r=17.0, lw=3.5, z=7)
    node(1124.5, 158.0, "skip", r=17.0, lw=3.5, z=7)
    ax.add_patch(Circle((1125.6, 221.5), 22.5, facecolor="none", edgecolor=VERMILLION,
                        lw=2.5 * PT, linestyle=(0, (5, 3)), zorder=7))
    text(1176, 109, "Urban center used in pattern detection")
    text(1176, 171, "Urban center skipped due to proximity")
    text(1176, 233, "Radius around urban center")

    box(29.5, 874.5, 440.0, 1069.5)
    for y, kind in zip([899.5, 946.5, 993.5, 1039.5], ["o1", "o2", "o3", "o4"]):
        node(66.5, y, kind, r=14.5, lw=1.6, z=7)
    for base, s in zip([910, 957, 1004, 1050],
                       ["First-order connectivity", "Second-order connectivity",
                        "Third-order connectivity", "Fourth-order connectivity"]):
        text(118, base, s)

    box(683.0, 954.5, 990.0, 1069.5)
    node(715.5, 978.5, "yel", r=14.5, lw=1.6, z=7)
    node(715.5, 1037.5, "rad", r=14.5, lw=1.6, z=7)
    text(752, 989, "Skipped radial")
    text(752, 1048, "Identified as radial")


def draw_radial(ax, k):
    PT = k * 72
    MS = PT / 0.72
    HATCH = 1.0 / (k * 100)
    plt.rcParams["hatch.linewidth"] = 0.6 * (k * 100)
    CONNECTED, NONCONN, NEW = BLUE, VERMILLION, YELLOW
    OUTSIDE_LINK = "#F4B183"
    DASH_BLUE, NAVY, PURPLE, BROWN = "#2F528F", "#002060", "#CC79A7", "#C55A11"
    BAND_HATCH = "#9ECAE1"
    OUTER_FILL, OUTER_EDGE = "#F2F2F2", BLACK
    LEGEND_FILL, LEGEND_EDGE = "#F8F8F8", "#D9D9D9"
    C = np.array([475.0, 474.8])

    def hatch(ch, n):
        return ch * max(1, int(round(n * HATCH)))

    def circle_path(cx, cy, r, n=720, reverse=False):
        t = np.linspace(0, 2 * np.pi, n)
        if reverse:
            t = t[::-1]
        v = np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])
        return v, [Path.MOVETO] + [Path.LINETO] * (n - 2) + [Path.CLOSEPOLY]

    def arrow(p0, p1, color, lw_px, style="-|>", ls="-", head=9, shrinkB=0, z=6):
        ax.annotate("", xy=p1, xytext=p0, zorder=z,
                    arrowprops=dict(arrowstyle=f"{style},head_length={head*0.11:.2f},head_width={head*0.045:.2f}",
                                    color=color, lw=lw_px * PT, linestyle=ls, mutation_scale=10 * MS,
                                    shrinkA=0, shrinkB=shrinkB * MS))

    def node(xy, r, face, edge=None, lw_px=1.2, z=8):
        ax.add_patch(Circle(xy, r, facecolor=face, edgecolor=edge or face, lw=lw_px * PT, zorder=z))

    def label(xy, s, rot=0, **kw):
        ax.text(*xy, s, fontsize=FS, rotation=rot, ha="center", va="center",
                rotation_mode="anchor", zorder=12,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")], **kw)

    ax.add_patch(Circle((475.0, 475.0), 440, facecolor=OUTER_FILL, edgecolor=OUTER_EDGE, lw=2.6 * PT, zorder=1))
    vo, co = circle_path(475.5, 468.0, 320)
    vi, ci = circle_path(475.5, 476.0, 240, reverse=True)
    ax.add_patch(PathPatch(Path(np.vstack([vo, vi]), co + ci), facecolor="none",
                           edgecolor=BAND_HATCH, lw=0, hatch=hatch("x", 8), zorder=2))
    ax.add_patch(Circle((475.5, 476.0), 240, facecolor="white", edgecolor="none", zorder=2.5))
    for (cx, cy, r) in [(475.5, 468.0, 320), (475.5, 476.0, 240)]:
        ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=DASH_BLUE,
                            lw=2.6 * PT, linestyle=(0, (7, 3)), zorder=3))
    ax.add_patch(Circle((478.5, 471.5), 280, facecolor="none", edgecolor=NAVY, lw=2.6 * PT, zorder=3.5))

    new_node = np.array([724.5, 596.5])
    u = (new_node - C) / np.linalg.norm(new_node - C)
    a0, half, Lc = np.arctan2(u[1], u[0]), np.radians(11.0), 240.0
    rho, cap_c = Lc * np.sin(half), C + Lc * u
    p_up = C + Lc * np.cos(half) * np.array([np.cos(a0 - half), np.sin(a0 - half)])
    p_lo = C + Lc * np.cos(half) * np.array([np.cos(a0 + half), np.sin(a0 + half)])
    ang_up = np.arctan2(*(p_up - cap_c)[::-1])
    ang_lo = np.arctan2(*(p_lo - cap_c)[::-1])
    if ang_lo < ang_up:
        ang_lo += 2 * np.pi
    cap = np.linspace(ang_up, ang_lo, 60)
    cone = np.vstack([C, p_up, np.column_stack([cap_c[0] + rho * np.cos(cap),
                                                cap_c[1] + rho * np.sin(cap)]), p_lo, C])
    ax.add_patch(Polygon(cone, closed=True, facecolor="white", edgecolor="#C9C9C9",
                         hatch=hatch("+", 4), lw=0, zorder=4))
    ax.add_patch(Polygon(cone, closed=True, facecolor="none", edgecolor=BROWN,
                         lw=2.0 * PT, linestyle=(0, (4, 2.5)), zorder=4.5))
    ax.plot(*np.column_stack([C, new_node]), color=BLACK, lw=2.0 * PT,
            linestyle=(0, (7, 4)), zorder=5)
    ax.add_patch(Arc(C, 132, 132, theta1=-79.5, theta2=122.0, color=BLACK,
                     lw=2.0 * PT, linestyle=(0, (1, 1.6)), zorder=5))

    connected = [(529.0, 168.7), (313.1, 280.8), (343.6, 704.0)]
    outside = [(772.8, 277.0), (611.1, 497.1), (723.6, 817.5)]
    for p in connected:
        arrow(C, p, CONNECTED, 2.0, head=9, shrinkB=7)
    for p in outside:
        arrow(C, p, OUTSIDE_LINK, 2.0, head=9, shrinkB=10)
    arrow((475.0, 486.0), (475.0, 914.0), PURPLE, 2.2, style="<|-|>", head=8, ls=(0, (4, 3)))
    arrow((476.0, 470.0), (200.0, 470.0), DASH_BLUE, 2.2, style="<|-|>", head=8, ls=(0, (4, 3)))
    arrow((200.0, 510.0), (237.0, 510.0), DASH_BLUE, 2.0, style="<|-|>", head=6.5)
    arrow((757.0, 481.0), (796.0, 481.0), DASH_BLUE, 2.0, style="<|-|>", head=6.5)
    d = np.radians(136.0)
    arrow(C + 279 * np.array([np.cos(d), np.sin(d)]), C + 320 * np.array([np.cos(d), np.sin(d)]),
          DASH_BLUE, 2.0, style="<|-|>", head=6.5)

    for p in connected:
        node(p, 8.3, CONNECTED)
    for p in [(185.3, 410.9), (322.7, 414.0)]:
        node(p, 8.3, NONCONN)
    for p in outside:
        node(p, 8.3, "white", BLACK, lw_px=1.6)
    node(new_node, 7.5, NEW, BLACK, lw_px=1.6)
    node(C, 7.0, BLACK)

    label((467.0, 447.5), "C", fontweight="bold")
    label((519.5, 470.5), r"$\theta$")
    label((334.0, 456.0), r"$r_{\mathrm{avg}}$")
    label((222.0, 530.5), r"$\Delta r$")
    label((781.0, 458.0), r"$\Delta r$")
    label((252.0, 670.0), r"$\Delta r$", rot=44)
    label((562.0, 511.5), "10°", rot=-25)
    label((664.5, 549.5), r"$l$")
    label((379.5, 603.5), r"$r_i$")
    label((487.5, 854.0), r"$r$")
    for xy in [(783.0, 257.0), (616.0, 475.5), (732.5, 794.5)]:
        label(xy, r"$x_i$")

    ax.add_patch(FancyBboxPatch((1025, 227), 487, 513, boxstyle="round,pad=0,rounding_size=8",
                                facecolor=LEGEND_FILL, edgecolor=LEGEND_EDGE, lw=1.5 * PT, zorder=1))
    SX, TX = 1077.0, 1118.0
    rows = [262.0, 324.5, 386.5, 463.5, 559.0, 636.0, 700.0]
    node((SX, rows[0]), 11.0, NONCONN)
    node((SX, rows[1]), 11.0, CONNECTED)
    node((SX - 1.5, rows[2]), 10.0, NEW, BLACK, lw_px=1.6)
    node((SX, rows[3]), 10.0, "white", BLACK, lw_px=1.6)
    ax.add_patch(Circle((SX + 1.5, rows[4]), 14.5, facecolor="none", edgecolor=NAVY, lw=2.6 * PT, zorder=8))
    ax.add_patch(Circle((SX + 3, rows[5]), 14.0, facecolor="none", edgecolor=BAND_HATCH, lw=0,
                        hatch=hatch("x", 8), zorder=8))
    ax.add_patch(Circle((SX + 3, rows[5]), 7.0, facecolor=LEGEND_FILL, edgecolor="none", zorder=8.2))
    for r, lw in [(7.0, 1.4), (14.0, 1.6)]:
        ax.add_patch(Circle((SX + 3, rows[5]), r, facecolor="none", edgecolor=DASH_BLUE, lw=lw * PT,
                            linestyle=(0, (2.5, 1.5)), zorder=8.4))
    ax.add_patch(Circle((SX + 3, rows[6]), 14.5, facecolor=OUTER_FILL, edgecolor=OUTER_EDGE, lw=2.6 * PT, zorder=8))

    def ltext(y, s):
        ax.text(TX, y, s, fontsize=FS, ha="left", va="center", zorder=12)

    ltext(rows[0], "Non-connected nodes")
    ltext(rows[1], "Connected nodes")
    ltext(rows[2], "New node")
    ltext(rows[3] - 13, "Connected nodes beyond the")
    ltext(rows[3] + 17, r"average radius ($x_i$)")
    ltext(rows[4], r"Average radius, $r_{\mathrm{avg}} = \Sigma_{i=1}^{N}\, r_i\,/\,N$")
    ltext(rows[5], r"Accepted range, $r_{\mathrm{avg}} \pm \Delta r$")
    ltext(rows[6], r"Initial search radius ($r$)")


draw_pattern(ax_a, ka)
draw_radial(ax_b, kb)

ax_a.text(A_X[0] + 8, A_Y[0] + 10, "a", fontsize=LETTER_FS, fontweight="bold",
          ha="left", va="top", zorder=20)
fig.text(ka * (A_X[0] + 8 - A_X[0]) / FIG_W_IN, hb_in / FIG_H_IN - 0.002, "b", fontsize=LETTER_FS,
         fontweight="bold", ha="left", va="top", zorder=20)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig3_pattern_radial_schematic")
for ext in ("png", "svg", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=600, facecolor="white")
if SANS != "Arial":
    with open(out + ".svg", encoding="utf-8") as f:
        svg = f.read()
    svg = svg.replace(f"font-family: '{SANS}', sans-serif", f"font-family: Arial, '{SANS}', sans-serif")
    svg = svg.replace(f"font-family: '{SANS}'", f"font-family: Arial, '{SANS}', sans-serif")
    with open(out + ".svg", "w", encoding="utf-8") as f:
        f.write(svg)
print(f"Figure {FIG_W_IN * 25.4:.0f} x {FIG_H_IN * 25.4:.0f} mm. Saved: {out}.{{png,svg,pdf}}")
