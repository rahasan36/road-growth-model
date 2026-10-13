#!/usr/bin/env python3
import argparse
import base64
import io
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from PIL import Image
from matplotlib.transforms import Affine2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG_DATA = os.path.join(REPO, "data", "figures")
WIDTH_MM = 180.0
FS = 6.5
X0, X1, Y0, Y1 = -45, 945, -4, 680

_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Arial", "Liberation Sans", "DejaVu Sans"] if f in _avail), "DejaVu Sans")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": FS,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.bf": f"{FONT}:bold",
    "mathtext.it": f"{FONT}:italic", "text.color": "black",
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})

EDGE, ARROW = "#333333", "#555555"
STAGE = {
    "input":   ("#D9D9D9", "Input data"),
    "prep":    ("#9AD2F2", "Pre-processing"),
    "pattern": ("#FFFFFF", "Pattern analysis"),
    "growth":  ("#E69F00", "Road growth"),
    "output":  ("#009E73", "Model output"),
}

SVG_SCALE = 3260.75 / 900
THUMBS = {
    "bw": ((1421.49, -470, 343.36, 444)),
    "aY": ((2039.79, 260, 346.14, 444)),
    "aZ": ((2039.79, 800, 343.36, 444)),
    "ba": ((1391.49, 1542.2, 403.36, 381.85)),
    "bz": ((780, 1502.2, 344.68, 468.16)),
    "aX": ((160, 800, 344.68, 444)),
    "aW": ((160, 260, 344.68, 444)),
    "by": ((784.58, -480, 335.52, 464)),
}


def thumb_extent(key):
    x, y, w, h = THUMBS[key]
    x, y = (x + 300) / SVG_SCALE, (y + 480) / SVG_SCALE
    return x, y, w / SVG_SCALE, h / SVG_SCALE


def load_thumbs(svg_path, extract_to):
    out = {}
    if svg_path and os.path.exists(svg_path):
        s = open(svg_path, encoding="utf-8").read()
        for m in re.finditer(r"<image([^>]*)>", s):
            iid = re.search(r'id="([^"]+)"', m.group(1))
            data = re.search(r'href="data:image/\w+;base64,([^"]+)"', m.group(1))
            if iid and data and iid.group(1) in THUMBS:
                out[iid.group(1)] = Image.open(io.BytesIO(base64.b64decode(data.group(1)))).convert("RGBA")
        if extract_to:
            os.makedirs(extract_to, exist_ok=True)
            for k, im in out.items():
                im.save(os.path.join(extract_to, f"{k}.png"))
    elif extract_to and os.path.isdir(extract_to):
        for k in THUMBS:
            p = os.path.join(extract_to, f"{k}.png")
            if os.path.exists(p):
                out[k] = Image.open(p).convert("RGBA")
    missing = set(THUMBS) - set(out)
    if missing:
        raise SystemExit(f"thumbnails missing: {sorted(missing)}")
    return out


def build_layers(drivers_dir, roads_dir, cell=500.0):
    import geopandas as gpd
    import rasterio
    from matplotlib import cm, colors
    from rasterio.features import rasterize
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import from_origin
    from scipy.ndimage import maximum_filter

    dst_crs = "ESRI:102953"
    land = gpd.read_file(os.path.join(roads_dir, "bgd_dist.shp")).to_crs(dst_crs).geometry.union_all()
    x0, y0, x1, y1 = land.bounds
    nx, ny = int(np.ceil((x1 - x0) / cell)), int(np.ceil((y1 - y0) / cell))
    T = from_origin(x0, y1, cell, cell)
    mask = rasterize([(land, 1)], out_shape=(ny, nx), transform=T, fill=0, dtype=np.uint8).astype(bool)

    def warp(path, how):
        with rasterio.open(path) as src:
            a = src.read(1).astype(np.float32)
            a[a == (src.nodata if src.nodata is not None else -9999)] = np.nan
            if src.nodata == 0:
                a[a == 0] = np.nan
            out = np.full((ny, nx), np.nan, dtype=np.float32)
            reproject(a, out, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                      dst_transform=T, dst_crs=dst_crs, dst_nodata=np.nan, resampling=how)
        return out

    def ramp(a, cmap, lo=2, hi=98, log=False):
        v = np.log10(a) if log else a.copy()
        ok = np.isfinite(v) & mask
        q0, q1 = np.nanpercentile(v[ok], [lo, hi])
        if q1 <= q0:
            q1 = np.nanmax(v[ok])
        n = np.clip((v - q0) / (q1 - q0), 0, 1)
        rgba = matplotlib.colormaps[cmap](0.08 + 0.62 * np.nan_to_num(n))
        rgba[~ok] = (0, 0, 0, 0)
        rgba[mask & ~np.isfinite(v)] = (1, 1, 1, 1)
        return rgba

    def binary(a, color):
        rgba = np.zeros((ny, nx, 4))
        rgba[mask] = (1, 1, 1, 1)
        rgba[mask & np.isfinite(a) & (a > 0)] = colors.to_rgba(color)
        return rgba

    d = drivers_dir
    L = {
        "urban": binary(warp(os.path.join(d, "BGD_urban.tif"), Resampling.max), "#D55E00"),
        "roads": binary(warp(os.path.join(roads_dir, "BGD_all_final_roads_2030_ssp1.tif"), Resampling.max)
                        .__array__() * np.nan_to_num(1.0), "#333333"),
        "pop": ramp(warp(os.path.join(d, "BGD_pop_density_total.tif"), Resampling.average), "Oranges", log=True),
        "roaddens": ramp(warp(os.path.join(d, "BGD_road_density_total.tif"), Resampling.average), "Blues"),
        "distroads": ramp(warp(os.path.join(d, "BGD_dist_roads.tif"), Resampling.average), "Purples", lo=5, hi=95, log=True),
        "disturban": ramp(warp(os.path.join(d, "BGD_dist_urban.tif"), Resampling.average), "Greys", lo=5, hi=99.5, log=True),
        "height": binary(maximum_filter(np.nan_to_num(warp(os.path.join(d, "BGD_height.tif"), Resampling.max)), 5)
                         .astype(np.float32), "#0072B2"),
    }
    with rasterio.open(os.path.join(roads_dir, "BGD_all_final_roads_2030_ssp1.tif")) as src:
        r = src.read(1).astype(np.float32)
        r = np.where((r >= 1) & (r <= 10), 1.0, np.nan)
        out = np.full((ny, nx), np.nan, dtype=np.float32)
        reproject(r, out, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                  dst_transform=T, dst_crs=dst_crs, dst_nodata=np.nan, resampling=Resampling.max)
    L["roads"] = binary(out, "#4D4D4D")
    return L


def draw_stack(ax, x0, y0, w, h, layers, lw):
    eu = np.array([0.482, -0.063]) * w
    ev = np.array([0.518, 0.389]) * w
    n = len(layers)
    span = h - (abs(eu[1]) + ev[1])
    sp = span / max(1, n - 1)
    for k in range(n - 1, -1, -1):
        A = np.array([x0, y0 + abs(eu[1]) + k * sp])
        quad = [A, A + eu, A + eu + ev, A + ev]
        z = 2 + (n - k) * 0.01
        ax.add_patch(Polygon(quad, closed=True, facecolor="white", edgecolor="none", zorder=z))
        img = layers[k]
        tr = Affine2D(np.array([[eu[0], ev[0], A[0]], [eu[1], ev[1], A[1]], [0, 0, 1]])) + ax.transData
        im = ax.imshow(img, extent=(0, 1, 1, 0), origin="upper", interpolation="nearest",
                       transform=tr, zorder=z + 0.001)
        ax.add_patch(Polygon(quad, closed=True, facecolor="none", edgecolor="#333333", lw=lw,
                             joinstyle="miter", zorder=z + 0.002))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", default=None,
                    help="original flowchart SVG; without it the thumbnails are read from --thumbs")
    ap.add_argument("--thumbs", default=os.path.join(HERE, "thumbnails"),
                    help="folder to save the extracted thumbnails to, or to read them from")
    ap.add_argument("--drivers_dir", default=os.path.join(FIG_DATA, "BGD_drivers_osm15"))
    ap.add_argument("--roads_dir", default=os.path.join(FIG_DATA, "roads"))
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    thumbs = load_thumbs(a.svg, a.thumbs)

    W_IN = WIDTH_MM / 25.4
    H_IN = W_IN * (Y1 - Y0) / (X1 - X0)
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y1, Y0)
    ax.set_aspect("equal")
    ax.axis("off")
    PT = W_IN * 72 / (X1 - X0)
    LW = 0.8
    LWA = 0.7

    def text(x, y, s, ha="center", va="center", **kw):
        return ax.text(x, y, s, fontsize=FS, ha=ha, va=va, linespacing=1.3, zorder=12, **kw)

    def note(x, y, lines, ha):
        text(x, y, "\n".join(lines), ha=ha, va="top")

    def arrow(p0, p1, rad=0.0, head=True):
        ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-|>" if head else "-", mutation_scale=7,
                                     color=ARROW, lw=LWA, shrinkA=0, shrinkB=0, zorder=3))

    def curve(p0, c, p1, head=True):
        verts = [p0, c, p1]
        path = Path(verts, [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=ARROW, lw=LWA, zorder=3))
        if head:
            t = np.array(p1) - np.array(c)
            t = t / np.linalg.norm(t)
            ax.add_patch(FancyArrowPatch(tuple(np.array(p1) - 6 * t), p1, arrowstyle="-|>",
                                         mutation_scale=7, color=ARROW, lw=LWA, shrinkA=0,
                                         shrinkB=0, zorder=3))

    def line(xs, ys):
        ax.plot(xs, ys, color=ARROW, lw=LWA, zorder=3)

    def number(x, y, n, stage):
        face = STAGE[stage][0]
        ax.add_patch(Circle((x, y), 9.5, facecolor=face, edgecolor=EDGE, lw=LW, zorder=8))
        text(x, y + 0.5, str(n), fontweight="bold",
             color="white" if stage == "output" else "black")

    def hexagon(x0, x1, y0, y1, stage, s, tip=12):
        cy = (y0 + y1) / 2
        pts = [(x0, cy), (x0 + tip, y0), (x1 - tip, y0), (x1, cy), (x1 - tip, y1), (x0 + tip, y1)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=STAGE[stage][0], edgecolor=EDGE, lw=LW, zorder=6))
        text((x0 + x1) / 2, cy, s)

    def stadium(x0, x1, y0, y1, stage, s):
        h = y1 - y0
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                    facecolor=STAGE[stage][0], edgecolor=EDGE, lw=LW, zorder=6))
        text((x0 + x1) / 2, (y0 + y1) / 2, s)

    def documents(x0, x1, y0, y1, stage, s):
        for k, d in enumerate([8, 4, 0]):
            xs = np.linspace(x0 + d, x1 + d, 50)
            wave = y1 - d - 6 * np.sin((xs - x0 - d) / (x1 - x0) * 2 * np.pi)
            verts = [(x0 + d, y0 - d), (x1 + d, y0 - d)] + list(zip(xs[::-1], wave[::-1]))
            ax.add_patch(Polygon(verts, closed=True, facecolor=STAGE[stage][0], edgecolor=EDGE,
                                 lw=LW, zorder=6 + k * 0.1))
        text((x0 + x1) / 2, (y0 + y1) / 2 - 2, s)

    L = build_layers(a.drivers_dir, a.roads_dir)
    for key, names in [("bw", ["urban", "roads", "pop", "roaddens", "distroads", "disturban"]),
                       ("aZ", ["distroads", "disturban", "height", "pop"])]:
        x, y, w, h = thumb_extent(key)
        draw_stack(ax, x, y, w, h, [L[n] for n in names], lw=0.5)
    for k, im in thumbs.items():
        if k in ("bw", "aZ"):
            continue
        x, y, w, h = thumb_extent(k)
        ax.imshow(np.asarray(im), extent=(x, x + w, y + h, y), interpolation="lanczos", zorder=2)

    arrow((393, 66), (472, 66))
    curve((574, 86), (693, 86), (693, 202))
    arrow((693, 327), (693, 352))
    curve((690, 478), (690, 612), (580, 612))
    arrow((466, 612), (395, 612))
    curve((298, 600), (175, 600), (175, 478))
    arrow((175, 353), (175, 328))
    curve((175, 203), (175, 66), (298, 66), head=False)

    line([523, 523], [125, 146])
    documents(463, 578, 155, 207, "input", "Input data\nprocessing")
    number(523, 136, 1, "input")

    hexagon(492, 615, 242, 292, "prep", "Urban cluster\ndetection")
    number(630, 248, 2, "prep")

    hexagon(492, 615, 383, 441, "prep", "Suitability &\ngrowth direction\ndetermination")
    number(630, 398, 3, "prep")

    line([523, 523], [516, 560])
    hexagon(462, 584, 470, 516, "pattern", "Pattern detection")
    number(540, 532, 4, "pattern")

    line([346, 346], [520, 549])
    hexagon(278, 414, 459, 521, "pattern", "New origin node\ndetection\nAngular constraints")
    number(362, 533, 5, "pattern")

    stadium(250, 385, 382, 438, "growth", "Radial pattern-based\ngrowth")
    number(236, 394, 6, "growth")
    line([222, 250], [410, 410])

    stadium(250, 385, 240, 292, "growth", "Hierarchical linear\ngrowth")
    number(236, 248, 7, "growth")
    line([222, 250], [266, 266])

    line([346, 346], [128, 154])
    dcx, dcy, dw, dh = 346, 180, 52, 27
    ax.add_patch(Polygon([(dcx - dw, dcy), (dcx, dcy - dh), (dcx + dw, dcy), (dcx, dcy + dh)], closed=True,
                         facecolor=STAGE["output"][0], edgecolor=EDGE, lw=LW, zorder=6))
    text(dcx, dcy, "Final road\nnetwork", color="white")
    number(328, 139, 8, "output")

    note(294, 10, ["Spatial validation", "Intersection accuracy", "Error diagnosis",
                   "Topological & structural metrics"], "right")
    note(576, 14, ["Input driver maps", "Road network", "Urban extent"], "left")
    note(745, 243, [r"$\bf{D}$ensity-$\bf{B}$ased $\bf{S}$patial",
                    r"$\bf{C}$lustering of $\bf{A}$pplications",
                    r"with $\bf{N}$oise (DBSCAN)", "algorithm"], "left")
    note(745, 388, ["\u201cAttraction values\u201d as a", "factor of distance, urban",
                    "area size, building height", "& total population"], "left")
    note(581, 622, ["Connectivity", "Centrality measures", "Dijkstra\u2019s algorithm"], "left")
    note(294, 622, ["Circuity", "Existing pattern filter", "New pattern origin"], "right")
    note(122, 398, ["Preferential attachment", "Pattern-based growth"], "right")
    note(122, 244, ["Road upgradation", "New road development", "Road hierarchy"], "right")

    lx, ly = -38, 560
    for i, (face, lab) in enumerate(STAGE.values()):
        y = ly + i * 20
        ax.add_patch(Rectangle((lx, y - 6), 18, 12, facecolor=face, edgecolor=EDGE, lw=LW, zorder=6))
        text(lx + 25, y, lab, ha="left")

    base = os.path.join(a.out, "Fig2_allocation_steps")
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
    print(f"Figure {WIDTH_MM:.0f} x {WIDTH_MM * (Y1 - Y0) / (X1 - X0):.0f} mm. Saved: {base}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
