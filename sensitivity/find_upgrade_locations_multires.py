import os
DATA = os.environ.get("ROAD_DATA", "data")
import os, sys, warnings, gc
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject as rasterio_reproject
from rasterio.windows import from_bounds as window_from_bounds
from scipy.ndimage import label as ndlabel, binary_dilation, generate_binary_structure
from scipy.spatial import cKDTree
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_growth_per_type(year: int, scenario: str, country: str = "BGD") -> List[float]:
    d = pd.read_csv(os.path.join(DATA, "road_demand_published.csv"))
    r = d[(d.country == country) & (d.ssp == scenario) & (d.year == year)]
    if len(r) != 1:
        raise ValueError(f"No road demand for {country} {scenario} {year}")
    r = r.iloc[0]
    return [r.main_km / 2, r.main_km / 2, r.secondary_km, r.tertiary_km, r.local_km]


CITIES = {
    "Dhaka":      (90.4125, 23.8103),
    "Chittagong": (91.7832, 22.3569),
}
CROP_RADIUS_M = 200_000


@dataclass
class Config:
    base_dir: str = DATA
    country: str = "BGD"
    scenario: int = 1

    forecast_year: int = 2030
    forecast_scenario: str = "SSP5"

    target_resolutions: List[int] = field(default_factory=lambda: [1000, 300, 100])
    native_res: int = 300

    max_radius: int = 10
    min_upgrade_per_intersect: int = 5
    upgrade_percent: List[float] = field(default_factory=lambda: [0.80, 0.75, 0.70, 0.50, 0.0])

    seed: int = 42
    output_dir: str = ""

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = os.path.join(self.base_dir, "upgrade_sensitivity_output")

    @property
    def input_dir(self) -> str:
        return os.path.join(self.base_dir, "model_inputs", self.country, f"SSP{self.scenario}", "2030")


def compute_crop_bbox(crs) -> Tuple[float, float, float, float]:
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    xs, ys = [], []
    for name, (lon, lat) in CITIES.items():
        cx, cy = t.transform(lon, lat)
        xs += [cx - CROP_RADIUS_M, cx + CROP_RADIUS_M]
        ys += [cy - CROP_RADIUS_M, cy + CROP_RADIUS_M]
    return min(xs), min(ys), max(xs), max(ys)


def crop_raster_to_bbox(data: np.ndarray, profile: dict,
                        bbox: Tuple[float,float,float,float]) -> Tuple[np.ndarray, dict]:
    transform = profile["transform"]
    xmin, ymin, xmax, ymax = bbox

    rb = rasterio.transform.array_bounds(data.shape[0], data.shape[1], transform)
    xmin = max(xmin, rb[0]); ymin = max(ymin, rb[1])
    xmax = min(xmax, rb[2]); ymax = min(ymax, rb[3])

    if xmin >= xmax or ymin >= ymax:
        return np.zeros((1,1), dtype=data.dtype), profile

    inv = ~transform
    col_min, row_max = [int(v) for v in inv * (xmin, ymin)]
    col_max, row_min = [int(v) for v in inv * (xmax, ymax)]

    row_min = max(0, row_min); row_max = min(data.shape[0], row_max)
    col_min = max(0, col_min); col_max = min(data.shape[1], col_max)

    cropped = data[row_min:row_max, col_min:col_max].copy()

    new_transform = rasterio.transform.from_origin(
        transform.c + col_min * transform.a,
        transform.f + row_min * transform.e,
        abs(transform.a), abs(transform.e),
    )
    new_profile = profile.copy()
    new_profile.update(height=cropped.shape[0], width=cropped.shape[1], transform=new_transform)
    return cropped, new_profile


def load_raster(path: str) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        profile = src.profile.copy()
    return data, profile


def resample_raster(data: np.ndarray, src_profile: dict, target_res: float,
                    method: Resampling = Resampling.nearest) -> Tuple[np.ndarray, dict]:
    src_res_x = src_profile["transform"].a
    src_res_y = abs(src_profile["transform"].e)
    new_height = max(1, int(round(data.shape[0] * src_res_y / target_res)))
    new_width  = max(1, int(round(data.shape[1] * src_res_x / target_res)))

    bounds = rasterio.transform.array_bounds(data.shape[0], data.shape[1], src_profile["transform"])
    new_transform = from_bounds(*bounds, new_width, new_height)

    dst_profile = src_profile.copy()
    dst_profile.update(width=new_width, height=new_height, transform=new_transform)

    src_arr = data.astype(np.float64).reshape(1, data.shape[0], data.shape[1])
    out = np.empty((1, new_height, new_width), dtype=np.float64)
    rasterio_reproject(
        source=src_arr, destination=out,
        src_transform=src_profile["transform"], src_crs=src_profile["crs"],
        dst_transform=new_transform, dst_crs=src_profile["crs"],
        resampling=method,
    )
    return out[0], dst_profile


def save_raster(path: str, data: np.ndarray, profile: dict):
    p = profile.copy()
    p.update(dtype="float32", count=1, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(data.astype(np.float32), 1)


def find_intersection_pixels(A1: np.ndarray, A2: np.ndarray) -> np.ndarray:
    struct = generate_binary_structure(2, 2)
    A1_dilated = binary_dilation(A1 > 0, structure=struct, iterations=1)
    rows, cols = np.where(A1_dilated & (A2 > 0))
    if len(rows) == 0:
        return np.empty((0, 2), dtype=int)
    return np.column_stack((rows, cols))


def find_upgrade_locations(
    A1, A2, road_density_map, max_radius, upgrade_length,
    min_upgrade_per_intersect, rng,
) -> Tuple[np.ndarray, np.ndarray]:
    N_r, N_c = A1.shape

    intersect_pts = find_intersection_pixels(A1, A2)
    if len(intersect_pts) == 0:
        return np.empty((0, 2), dtype=int), A2.copy()

    labels, _ = ndlabel((A2 > 0).astype(np.int32), structure=np.ones((3, 3)))
    seg_ids = labels[intersect_pts[:, 0], intersect_pts[:, 1]]
    valid = seg_ids > 0
    intersect_pts = intersect_pts[valid]
    seg_ids = seg_ids[valid]
    if len(intersect_pts) == 0:
        return np.empty((0, 2), dtype=int), A2.copy()

    density_vals = road_density_map[intersect_pts[:, 0], intersect_pts[:, 1]]
    max_d = density_vals.max()
    upgrade_radius = (max_radius * density_vals / max_d) if max_d > 0 else np.full(len(density_vals), max_radius)

    unique_segs = np.unique(seg_ids)
    seg_trees, seg_coords = {}, {}
    for seg in unique_segs:
        r, c = np.where(labels == seg)
        coords = np.column_stack((r, c))
        seg_coords[seg] = coords
        seg_trees[seg] = cKDTree(coords)

    A2_upgrade = A2.copy()
    upgraded_list = []
    upgrade_left = upgrade_length

    for idx in rng.permutation(len(seg_ids)):
        if upgrade_left <= 0:
            break
        seg = seg_ids[idx]
        indices = seg_trees[seg].query_ball_point(intersect_pts[idx].astype(float), r=upgrade_radius[idx])
        if not indices:
            continue
        cand = seg_coords[seg][indices]
        new_count = np.sum(A2_upgrade[cand[:, 0], cand[:, 1]] != -1)
        if new_count >= min_upgrade_per_intersect:
            upgraded_list.append(cand)
            A2_upgrade[cand[:, 0], cand[:, 1]] = -1
            upgrade_left = upgrade_length - int(np.sum(A2_upgrade == -1))

    if upgraded_list:
        all_up = np.vstack(upgraded_list)
        _, ui = np.unique(all_up[:, 0] * N_c + all_up[:, 1], return_index=True)
        return all_up[ui], A2_upgrade
    return np.empty((0, 2), dtype=int), A2_upgrade


def compute_upgrade_metrics(to_be_upgraded, A2_upgrade, n_intersect_pts, resolution_m, road_type):
    n = len(to_be_upgraded)
    km = n * resolution_m / 1000.0

    seg_lengths = []
    if n > 0:
        mask = (A2_upgrade == -1).astype(np.int32)
        lbl, nlbl = ndlabel(mask, structure=np.ones((3, 3)))
        for i in range(1, nlbl + 1):
            seg_lengths.append(int(np.sum(lbl == i)))

    mean_s = float(np.mean(seg_lengths)) if seg_lengths else 0
    std_s  = float(np.std(seg_lengths)) if seg_lengths else 0
    cr = float(to_be_upgraded[:, 0].mean()) if n > 0 else np.nan
    cc = float(to_be_upgraded[:, 1].mean()) if n > 0 else np.nan

    return {
        "road_type": road_type, "resolution_m": resolution_m,
        "n_intersect_pts": n_intersect_pts, "n_upgraded_px": n,
        "upgraded_km": round(km, 3), "n_segments": len(seg_lengths),
        "mean_seg_len_px": round(mean_s, 2), "std_seg_len_px": round(std_s, 2),
        "mean_seg_len_km": round(mean_s * resolution_m / 1000, 4),
        "std_seg_len_km": round(std_s * resolution_m / 1000, 4),
        "centroid_row_px": round(cr, 2) if not np.isnan(cr) else None,
        "centroid_col_px": round(cc, 2) if not np.isnan(cc) else None,
    }


def pixel_to_map(row, col, transform):
    return (transform.c + col * transform.a + row * transform.b,
            transform.f + col * transform.d + row * transform.e)


LAYER_NAMES_CAT  = ["road_types", "urban"]
LAYER_NAMES_CONT = ["road_density_total"]


def load_and_crop_layers(cfg: Config, bbox) -> dict:
    prefix = os.path.join(cfg.input_dir, f"{cfg.country}_")
    layers = {}

    all_names = LAYER_NAMES_CAT + LAYER_NAMES_CONT
    for name in all_names:
        path = prefix + name + ".tif"
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found")
            continue
        data, profile = load_raster(path)
        data, profile = crop_raster_to_bbox(data, profile, bbox)
        is_cat = name in LAYER_NAMES_CAT
        layers[name] = {"data": data, "profile": profile, "categorical": is_cat}
        print(f"    {name}: {data.shape}")

    return layers


def resample_all_layers(layers: dict, target_res: float) -> dict:
    resampled = {}
    rt = layers["road_types"]
    rt_data, rt_profile = resample_raster(rt["data"], rt["profile"], target_res, Resampling.nearest)
    resampled["road_types"] = {"data": rt_data, "profile": rt_profile}
    target_shape = rt_data.shape

    for name, layer in layers.items():
        if name == "road_types":
            continue
        method = Resampling.nearest if layer["categorical"] else Resampling.bilinear
        d, p = resample_raster(layer["data"], layer["profile"], target_res, method)
        if d.shape != target_shape:
            aligned = np.zeros(target_shape, dtype=np.float64)
            mr, mc = min(d.shape[0], target_shape[0]), min(d.shape[1], target_shape[1])
            aligned[:mr, :mc] = d[:mr, :mc]
            d = aligned
        p.update(height=target_shape[0], width=target_shape[1], transform=rt_profile["transform"])
        resampled[name] = {"data": d, "profile": p}
    return resampled


def run_upgrade_analysis(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    growth_km = get_growth_per_type(cfg.forecast_year, cfg.forecast_scenario, cfg.country)

    print("=" * 72)
    print("Road Upgrade Location Sensitivity — Bangladesh (BGD)")
    print("=" * 72)
    print(f"Input       : {cfg.input_dir}")
    print(f"Output      : {cfg.output_dir}")
    print(f"Resolutions : {cfg.target_resolutions} m")
    print(f"Forecast    : {cfg.forecast_year} {cfg.forecast_scenario}")
    print(f"Growth km   : {[round(g,1) for g in growth_km]}")
    print(f"Crop cities : {list(CITIES.keys())}, radius={CROP_RADIUS_M/1000:.0f} km")
    print()

    ref_path = os.path.join(cfg.input_dir, f"{cfg.country}_road_types.tif")
    with rasterio.open(ref_path) as src:
        ref_crs = src.crs
    bbox = compute_crop_bbox(ref_crs)
    print(f"Crop bbox (Mollweide): x=[{bbox[0]:.0f}, {bbox[2]:.0f}], y=[{bbox[1]:.0f}, {bbox[3]:.0f}]")

    print("Loading & cropping layers …")
    layers = load_and_crop_layers(cfg, bbox)
    print(f"  {len(layers)} layers loaded.\n")

    all_metrics: List[dict] = []

    for res in cfg.target_resolutions:
        print("-" * 72)
        print(f"RESOLUTION: {res} m")
        print("-" * 72)

        if res == cfg.native_res:
            resampled = {k: {"data": v["data"].copy(), "profile": v["profile"].copy()}
                         for k, v in layers.items()}
            print("  Native 300 m (no resampling).")
        else:
            print(f"  Resampling to {res} m …")
            resampled = resample_all_layers(layers, res)
            print(f"  Grid: {resampled['road_types']['data'].shape}")

        rt_data = resampled["road_types"]["data"]
        rt_prof = resampled["road_types"]["profile"]

        rd_map = resampled.get("road_density_total", {}).get("data")
        if rd_map is None:
            print("  ERROR: no road_density_total, skipping.")
            continue
        if rd_map.shape != rt_data.shape:
            a = np.zeros(rt_data.shape, dtype=np.float64)
            mr, mc = min(rd_map.shape[0], rt_data.shape[0]), min(rd_map.shape[1], rt_data.shape[1])
            a[:mr, :mc] = rd_map[:mr, :mc]
            rd_map = a

        unique_types = sorted([int(v) for v in np.unique(rt_data) if v > 0])
        print(f"  Road types: {unique_types}")

        lin_scale = cfg.native_res / res
        area_scale = lin_scale ** 2

        full_road_px_300 = {1: 12271, 2: 33527, 3: 77296, 4: 472386, 5: 33237}
        crop_fractions = {}
        for t in unique_types:
            cropped_px = int(np.sum(rt_data == t))
            native_px = full_road_px_300.get(t, 1)
            crop_fractions[t] = min(1.0, cropped_px * (res / cfg.native_res)**2 / max(native_px, 1))

        predicted_px = [round(km * 1000 / res) for km in growth_km]
        upgrade_lengths_full = [
            round(px * cfg.upgrade_percent[i]) if i < len(cfg.upgrade_percent) else 0
            for i, px in enumerate(predicted_px)
        ]
        upgrade_lengths = []
        for i, t in enumerate(unique_types):
            if i < len(upgrade_lengths_full):
                upgrade_lengths.append(max(0, round(upgrade_lengths_full[i] * crop_fractions.get(t, 1.0))))
            else:
                upgrade_lengths.append(0)

        max_r = max(1, round(cfg.max_radius * lin_scale))
        min_u = max(1, round(cfg.min_upgrade_per_intersect * area_scale))

        print(f"  max_radius={max_r} px, min_upgrade={min_u} px")
        print(f"  upgrade_lengths (px, crop-scaled) = {upgrade_lengths}")

        Nr, Nc = rt_data.shape
        final_map = np.zeros((Nr, Nc), dtype=np.float32)

        for ti, rtype in enumerate(unique_types):
            if ti >= len(upgrade_lengths):
                break
            ul = upgrade_lengths[ti]

            A1 = (rt_data == rtype).astype(np.float64)
            if ti + 1 < len(unique_types):
                nxt = unique_types[ti + 1]
                A2 = (rt_data == nxt).astype(np.float64)
            else:
                m = compute_upgrade_metrics(np.empty((0,2),dtype=int), np.zeros_like(A1), 0, res, rtype)
                all_metrics.append(m)
                print(f"  Type {rtype}: highest, skipped.")
                continue

            ipts = find_intersection_pixels(A1, A2)
            n_int = len(ipts)

            if ul <= 0:
                m = compute_upgrade_metrics(np.empty((0,2),dtype=int), A2.copy(), n_int, res, rtype)
                all_metrics.append(m)
                print(f"  Type {rtype}: ul=0, {n_int} intersections.")
                continue

            print(f"  Type {rtype}→{nxt}: ul={ul}px ({ul*res/1000:.1f}km), "
                  f"{n_int} ints …", end=" ", flush=True)

            upgraded, A2_up = find_upgrade_locations(A1, A2, rd_map, max_r, ul, min_u, rng)
            n_act = len(upgraded)
            print(f"→ {n_act}px ({n_act*res/1000:.1f}km)")

            if n_act > 0:
                final_map[upgraded[:, 0], upgraded[:, 1]] = rtype

            m = compute_upgrade_metrics(upgraded, A2_up, n_int, res, rtype)
            if m["centroid_row_px"] is not None:
                mx, my = pixel_to_map(m["centroid_row_px"], m["centroid_col_px"], rt_prof["transform"])
                m["centroid_x_map"] = round(mx, 2)
                m["centroid_y_map"] = round(my, 2)
            else:
                m["centroid_x_map"] = m["centroid_y_map"] = None
            all_metrics.append(m)

            del A1, A2, upgraded, A2_up
            gc.collect()

        out_tif = os.path.join(cfg.output_dir,
            f"{cfg.country}_upgrades_{res}m_{cfg.forecast_year}_{cfg.forecast_scenario}_sc{cfg.scenario}.tif")
        save_raster(out_tif, final_map, rt_prof)
        print(f"  → {out_tif}\n")

        del resampled, final_map
        gc.collect()

    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(cfg.output_dir,
        f"{cfg.country}_upgrade_sensitivity_{cfg.forecast_year}_{cfg.forecast_scenario}_sc{cfg.scenario}.csv")
    df.to_csv(csv_path, index=False)

    print("=" * 72)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 72)
    if not df.empty:
        for col in ["n_intersect_pts", "n_upgraded_px", "upgraded_km",
                     "n_segments", "mean_seg_len_km"]:
            if col in df.columns:
                print(f"\n  {col}:")
                piv = df.pivot_table(index="resolution_m", columns="road_type",
                                     values=col, aggfunc="first")
                print(piv.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n  Centroid Shift from 300 m Baseline (m):")
    bl = df[df["resolution_m"] == 300]
    for _, row in df.iterrows():
        if row["resolution_m"] == 300 or row.get("centroid_x_map") is None:
            continue
        b = bl[bl["road_type"] == row["road_type"]]
        if b.empty or b.iloc[0]["centroid_x_map"] is None:
            continue
        dx = row["centroid_x_map"] - b.iloc[0]["centroid_x_map"]
        dy = row["centroid_y_map"] - b.iloc[0]["centroid_y_map"]
        print(f"    Type {int(row['road_type'])}, {int(row['resolution_m'])}m: "
              f"Δx={dx:.0f}, Δy={dy:.0f}, shift={np.sqrt(dx**2+dy**2):.0f} m")

    print("\n  Aggregate Totals:")
    agg = df.groupby("resolution_m").agg(
        total_intersect=("n_intersect_pts","sum"),
        total_upgraded_px=("n_upgraded_px","sum"),
        total_upgraded_km=("upgraded_km","sum"),
        total_segments=("n_segments","sum"),
        mean_seg_km=("mean_seg_len_km","mean"),
    ).round(4)
    print(agg.to_string())

    print(f"\nCSV  : {csv_path}")
    print(f"TIFFs: {cfg.output_dir}")
    return df


if __name__ == "__main__":
    cfg = Config()
    df = run_upgrade_analysis(cfg)
