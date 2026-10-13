import os
DATA = os.environ.get("ROAD_DATA", "data")
import os, warnings, gc
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject as rasterio_reproject
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple

warnings.filterwarnings("ignore")

CITIES = {"Dhaka": (90.4125, 23.8103), "Chittagong": (91.7832, 22.3569)}
CROP_RADIUS_M = 200_000


@dataclass
class Config:
    base_dir: str = DATA
    country: str = "BGD"
    scenario: int = 1
    target_resolutions: List[int] = field(default_factory=lambda: [30, 100, 300, 500, 1000])
    native_res: int = 300
    epsilon_native_px: float = 3.0
    minpts_native: int = 5
    min_pixel_count: int = 2
    output_dir: str = ""

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = os.path.join(self.base_dir, "urban_cluster_sensitivity_output")

    @property
    def input_dir(self):
        return os.path.join(self.base_dir, "Processed", self.country, f"scenario_{self.scenario}")

    @property
    def epsilon_m(self):
        return self.epsilon_native_px * self.native_res


def compute_crop_bbox(crs):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = [], []
    for lon, lat in CITIES.values():
        cx, cy = t.transform(lon, lat)
        xs += [cx - CROP_RADIUS_M, cx + CROP_RADIUS_M]
        ys += [cy - CROP_RADIUS_M, cy + CROP_RADIUS_M]
    return min(xs), min(ys), max(xs), max(ys)


def crop_raster(data, profile, bbox):
    transform = profile["transform"]
    rb = rasterio.transform.array_bounds(data.shape[0], data.shape[1], transform)
    xmin, ymin = max(bbox[0], rb[0]), max(bbox[1], rb[1])
    xmax, ymax = min(bbox[2], rb[2]), min(bbox[3], rb[3])
    if xmin >= xmax or ymin >= ymax:
        return np.zeros((1, 1), dtype=data.dtype), profile
    inv = ~transform
    c0, r1 = [int(v) for v in inv * (xmin, ymin)]
    c1, r0 = [int(v) for v in inv * (xmax, ymax)]
    r0, r1 = max(0, r0), min(data.shape[0], r1)
    c0, c1 = max(0, c0), min(data.shape[1], c1)
    cropped = data[r0:r1, c0:c1].copy()
    new_t = rasterio.transform.from_origin(
        transform.c + c0 * transform.a, transform.f + r0 * transform.e,
        abs(transform.a), abs(transform.e))
    p = profile.copy()
    p.update(height=cropped.shape[0], width=cropped.shape[1], transform=new_t)
    return cropped, p


def resample_raster(data, src_profile, target_res, method=Resampling.nearest):
    src_res_x = src_profile["transform"].a
    src_res_y = abs(src_profile["transform"].e)
    new_h = max(1, int(round(data.shape[0] * src_res_y / target_res)))
    new_w = max(1, int(round(data.shape[1] * src_res_x / target_res)))
    bounds = rasterio.transform.array_bounds(data.shape[0], data.shape[1], src_profile["transform"])
    new_t = from_bounds(*bounds, new_w, new_h)
    p = src_profile.copy()
    p.update(width=new_w, height=new_h, transform=new_t)
    src_arr = data.astype(np.float64).reshape(1, *data.shape)
    out = np.empty((1, new_h, new_w), dtype=np.float64)
    rasterio_reproject(src_arr, out, src_transform=src_profile["transform"],
                       src_crs=src_profile["crs"], dst_transform=new_t,
                       dst_crs=src_profile["crs"], resampling=method)
    return out[0], p


def save_raster(path, data, profile):
    p = profile.copy(); p.update(dtype="float32", count=1, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(data.astype(np.float32), 1)


def pixel_to_map(row, col, transform):
    return (transform.c + col * transform.a + row * transform.b,
            transform.f + col * transform.d + row * transform.e)


def get_urban_coords_at_resolution(
    urban_rows_300: np.ndarray,
    urban_cols_300: np.ndarray,
    urban_vals_300: np.ndarray,
    src_shape: Tuple[int, int],
    native_res: int,
    target_res: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    scale = native_res / target_res
    target_h = max(1, int(round(src_shape[0] * scale)))
    target_w = max(1, int(round(src_shape[1] * scale)))

    if target_res >= native_res:
        target_rows = np.round(urban_rows_300 * scale).astype(int)
        target_cols = np.round(urban_cols_300 * scale).astype(int)
        target_rows = np.clip(target_rows, 0, target_h - 1)
        target_cols = np.clip(target_cols, 0, target_w - 1)
        linear = target_rows * target_w + target_cols
        _, unique_idx = np.unique(linear, return_index=True)
        return target_rows[unique_idx], target_cols[unique_idx], urban_vals_300[unique_idx], (target_h, target_w)
    else:
        block_h = max(1, int(round(native_res / target_res)))
        block_w = max(1, int(round(native_res / target_res)))

        all_rows, all_cols, all_vals = [], [], []
        for dr in range(block_h):
            for dc in range(block_w):
                tr = np.clip((urban_rows_300 * block_h + dr).astype(int), 0, target_h - 1)
                tc = np.clip((urban_cols_300 * block_w + dc).astype(int), 0, target_w - 1)
                all_rows.append(tr)
                all_cols.append(tc)
                all_vals.append(urban_vals_300)

        all_rows = np.concatenate(all_rows)
        all_cols = np.concatenate(all_cols)
        all_vals = np.concatenate(all_vals)

        linear = all_rows * target_w + all_cols
        _, unique_idx = np.unique(linear, return_index=True)
        return all_rows[unique_idx], all_cols[unique_idx], all_vals[unique_idx], (target_h, target_w)


def detect_clusters_sparse(
    rows: np.ndarray, cols: np.ndarray,
    pop_data_300: np.ndarray, height_data_300: np.ndarray,
    src_rows_300: np.ndarray, src_cols_300: np.ndarray,
    native_res: int, target_res: int,
    grid_shape: Tuple[int, int],
    epsilon_px: float, minpts: int, min_pixel_count: int,
):
    n_urban = len(rows)
    if n_urban == 0:
        return np.empty((0, 5)), np.array([]), {}, 0

    coords = np.column_stack((rows, cols)).astype(np.float64)

    db = DBSCAN(eps=epsilon_px, min_samples=minpts, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(coords)

    cluster_ids = np.unique(labels)
    cluster_ids = cluster_ids[cluster_ids >= 0]

    valid_ids = [c for c in cluster_ids if np.sum(labels == c) >= min_pixel_count]
    valid_ids = np.array(valid_ids)

    if len(valid_ids) == 0:
        return np.empty((0, 5)), labels, {}, 0

    scale = native_res / target_res
    src_r = np.clip(np.round(rows / scale).astype(int), 0, pop_data_300.shape[0] - 1)
    src_c = np.clip(np.round(cols / scale).astype(int), 0, pop_data_300.shape[1] - 1)

    centers = np.zeros((len(valid_ids), 5))
    cluster_pixels = {}
    n_clustered = 0

    for i, cid in enumerate(valid_ids):
        mask = labels == cid
        r_c, c_c = rows[mask], cols[mask]
        sr, sc = src_r[mask], src_c[mask]

        centers[i, 0] = r_c.mean()
        centers[i, 1] = c_c.mean()
        centers[i, 3] = mask.sum()
        centers[i, 2] = pop_data_300[sr, sc].mean()
        centers[i, 4] = height_data_300[sr, sc].mean()

        cluster_pixels[i] = (r_c, c_c)
        n_clustered += mask.sum()

    if centers[:, 2].max() > 0:
        centers[:, 2] /= centers[:, 2].max()
    if centers[:, 4].max() > 0:
        centers[:, 4] /= centers[:, 4].max()

    return centers, labels, cluster_pixels, int(n_clustered)


def detect_clusters_morphological(
    rows: np.ndarray, cols: np.ndarray,
    pop_data_300: np.ndarray, height_data_300: np.ndarray,
    native_res: int, target_res: int,
    grid_shape: Tuple[int, int],
    epsilon_px: float, minpts: int, min_pixel_count: int,
):
    from scipy.ndimage import label, binary_dilation
    from skimage.morphology import disk

    n_urban = len(rows)
    if n_urban == 0:
        return np.empty((0, 5)), np.array([]), {}, 0

    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    pad = int(np.ceil(epsilon_px)) + 1
    local_rows = rows - r_min + pad
    local_cols = cols - c_min + pad
    local_h = (r_max - r_min) + 2 * pad + 1
    local_w = (c_max - c_min) + 2 * pad + 1

    print(f"    Morphological grid: {local_h}x{local_w} = {local_h*local_w/1e6:.1f}M px")

    mask = np.zeros((local_h, local_w), dtype=bool)
    mask[local_rows, local_cols] = True

    dilation_radius = max(1, int(np.ceil(epsilon_px / 2)))
    se = disk(dilation_radius)
    dilated = binary_dilation(mask, structure=se)
    del mask; gc.collect()

    labeled, n_comp = label(dilated)
    del dilated; gc.collect()

    pixel_labels = labeled[local_rows, local_cols]
    del labeled; gc.collect()

    unique_labels, counts = np.unique(pixel_labels, return_counts=True)
    valid_mask = counts >= max(minpts, min_pixel_count)
    valid_labels = set(unique_labels[valid_mask].tolist()) - {0}

    scale = native_res / target_res
    src_r = np.clip(np.round(rows / scale).astype(int), 0, pop_data_300.shape[0] - 1)
    src_c = np.clip(np.round(cols / scale).astype(int), 0, pop_data_300.shape[1] - 1)

    centers_list = []
    cluster_pixels = {}
    n_clustered = 0
    idx = 0

    for lab in sorted(valid_labels):
        m = pixel_labels == lab
        r_c, c_c = rows[m], cols[m]
        sr, sc = src_r[m], src_c[m]

        center = np.zeros(5)
        center[0] = r_c.mean()
        center[1] = c_c.mean()
        center[3] = m.sum()
        center[2] = pop_data_300[sr, sc].mean()
        center[4] = height_data_300[sr, sc].mean()

        centers_list.append(center)
        cluster_pixels[idx] = (r_c, c_c)
        n_clustered += int(m.sum())
        idx += 1

    if len(centers_list) == 0:
        return np.empty((0, 5)), pixel_labels, {}, 0

    centers = np.array(centers_list)
    if centers[:, 2].max() > 0:
        centers[:, 2] /= centers[:, 2].max()
    if centers[:, 4].max() > 0:
        centers[:, 4] /= centers[:, 4].max()

    return centers, pixel_labels, cluster_pixels, n_clustered


def jaccard_from_sets(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 1.0


def centroid_match_rate(ref_centers, test_centers, tolerance_m,
                        ref_transform, test_transform):
    if len(ref_centers) == 0 or len(test_centers) == 0:
        return 0.0, float("nan")
    ref_map = np.array([pixel_to_map(r, c, ref_transform) for r, c in ref_centers[:, :2]])
    test_map = np.array([pixel_to_map(r, c, test_transform) for r, c in test_centers[:, :2]])
    tree = cKDTree(test_map)
    dists, _ = tree.query(ref_map)
    matched = dists <= tolerance_m
    return float(matched.mean()), float(dists[matched].mean()) if matched.any() else float("nan")


def coords_to_300m_set(rows, cols, target_res, native_res, shape_300):
    scale = native_res / target_res
    sr = np.clip(np.round(rows / scale).astype(int), 0, shape_300[0] - 1)
    sc = np.clip(np.round(cols / scale).astype(int), 0, shape_300[1] - 1)
    return set(zip(sr.tolist(), sc.tolist()))


def run_urban_cluster_sensitivity(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 72)
    print("Urban Cluster Detection Sensitivity — Bangladesh (BGD)")
    print("=" * 72)
    print(f"Input        : {cfg.input_dir}")
    print(f"Output       : {cfg.output_dir}")
    print(f"Resolutions  : {cfg.target_resolutions} m")
    print(f"DBSCAN ε     : {cfg.epsilon_m:.0f} m physical, minpts={cfg.minpts_native} (at 300m)")
    print(f"Crop         : {list(CITIES.keys())}, {CROP_RADIUS_M/1000:.0f} km radius")
    print()

    prefix = os.path.join(cfg.input_dir, f"{cfg.country}_")

    print("Loading & cropping 300m layers …")
    with rasterio.open(prefix + "urban.tif") as src:
        u_raw = src.read(1).astype(np.float64); u_prof = src.profile.copy()
    with rasterio.open(prefix + "pop_density_total.tif") as src:
        p_raw = src.read(1).astype(np.float64); p_prof = src.profile.copy()
    with rasterio.open(prefix + "height.tif") as src:
        h_raw = src.read(1).astype(np.float64); h_prof = src.profile.copy()

    bbox = compute_crop_bbox(u_prof["crs"])
    urban_300, urban_300_prof = crop_raster(u_raw, u_prof, bbox)
    pop_300, _ = crop_raster(p_raw, p_prof, bbox)
    height_300, _ = crop_raster(h_raw, h_prof, bbox)
    del u_raw, p_raw, h_raw; gc.collect()

    min_r = min(urban_300.shape[0], pop_300.shape[0], height_300.shape[0])
    min_c = min(urban_300.shape[1], pop_300.shape[1], height_300.shape[1])
    urban_300 = urban_300[:min_r, :min_c]
    pop_300 = pop_300[:min_r, :min_c]
    height_300 = height_300[:min_r, :min_c]
    urban_300_prof.update(height=min_r, width=min_c)

    pop_300[pop_300 < 0] = 0
    pm = pop_300.max(); pop_300 = pop_300 / pm if pm > 0 else pop_300
    hm = height_300.max(); height_300 = height_300 / hm if hm > 0 else height_300

    ur300, uc300 = np.where(urban_300 > 0)
    uv300 = urban_300[ur300, uc300]
    shape_300 = urban_300.shape
    print(f"  Urban crop: {shape_300}, {len(ur300)} urban pixels")
    print()

    results = {}
    metrics = []

    for res in cfg.target_resolutions:
        print("-" * 72)
        print(f"RESOLUTION: {res} m")
        print("-" * 72)

        rows, cols, vals, grid_shape = get_urban_coords_at_resolution(
            ur300, uc300, uv300, shape_300, cfg.native_res, res)
        n_urban = len(rows)

        eps_px = max(1.5, cfg.epsilon_m / res)
        area_ratio = (cfg.native_res / res) ** 2
        minpts_scaled = max(2, round(cfg.minpts_native * area_ratio))

        print(f"  Grid: {grid_shape}, urban px: {n_urban}")
        print(f"  DBSCAN: ε={eps_px:.2f} px ({cfg.epsilon_m:.0f} m), minpts={minpts_scaled}")

        t300 = urban_300_prof["transform"]
        bounds = rasterio.transform.array_bounds(shape_300[0], shape_300[1], t300)
        t_res = from_bounds(*bounds, grid_shape[1], grid_shape[0])
        res_prof = urban_300_prof.copy()
        res_prof.update(height=grid_shape[0], width=grid_shape[1], transform=t_res)

        if res <= 30:
            print(f"  Using morphological connected-component approach (memory-safe)")
            centers, labels, cpx, n_clustered = detect_clusters_morphological(
                rows, cols, pop_300, height_300,
                cfg.native_res, res, grid_shape,
                eps_px, minpts_scaled, cfg.min_pixel_count,
            )
        else:
            centers, labels, cpx, n_clustered = detect_clusters_sparse(
                rows, cols, pop_300, height_300, ur300, uc300,
                cfg.native_res, res, grid_shape,
                eps_px, minpts_scaled, cfg.min_pixel_count,
            )

        n_clusters = len(centers)
        pct = 100 * n_clustered / n_urban if n_urban > 0 else 0

        print(f"  Clusters: {n_clusters}, clustered px: {n_clustered} ({pct:.1f}%)")
        if n_clusters > 0:
            si = np.argsort(-centers[:, 3])
            for rank, idx in enumerate(si[:5]):
                mx, my = pixel_to_map(centers[idx, 0], centers[idx, 1], t_res)
                print(f"    #{rank+1}: {int(centers[idx,3])} px, map=({mx:.0f},{my:.0f})")

        results[res] = {
            "centers": centers, "labels": labels, "cpx": cpx,
            "profile": res_prof, "n_urban": n_urban,
            "rows": rows, "cols": cols, "n_clustered": n_clustered,
        }

        metrics.append({
            "resolution_m": res, "grid_rows": grid_shape[0], "grid_cols": grid_shape[1],
            "epsilon_px": round(eps_px, 2), "minpts": minpts_scaled,
            "n_urban_px": n_urban, "n_clusters": n_clusters,
            "n_clustered_px": n_clustered, "pct_clustered": round(pct, 2),
            "mean_cluster_size_px": round(centers[:, 3].mean(), 1) if n_clusters > 0 else 0,
            "median_cluster_size_px": round(float(np.median(centers[:, 3])), 1) if n_clusters > 0 else 0,
            "max_cluster_size_px": int(centers[:, 3].max()) if n_clusters > 0 else 0,
            "std_cluster_size_px": round(centers[:, 3].std(), 1) if n_clusters > 0 else 0,
        })

        if res >= 100:
            lmap = np.zeros(grid_shape, dtype=np.float32)
            for i, (cr, cc) in cpx.items():
                lmap[cr, cc] = i + 1
            out_tif = os.path.join(cfg.output_dir, f"{cfg.country}_clusters_{res}m.tif")
            save_raster(out_tif, lmap, res_prof)
            print(f"  → {out_tif}")
            del lmap
        print()
        gc.collect()

    print("=" * 72)
    print("CROSS-RESOLUTION AGREEMENT (vs 300 m)")
    print("=" * 72)

    ref = results[300]
    ref_clustered_set = set()
    for cr, cc in ref["cpx"].values():
        ref_clustered_set.update(zip(cr.tolist(), cc.tolist()))
    ref_set_300 = ref_clustered_set

    agreement = []
    for res in cfg.target_resolutions:
        r = results[res]
        if res == 300:
            agreement.append({
                "resolution_m": 300, "jaccard_vs_300m": 1.0,
                "centroid_match_1km": 100.0, "centroid_match_3km": 100.0,
                "centroid_match_5km": 100.0,
                "mean_matched_dist_m": 0.0, "cluster_count_ratio": 1.0,
            })
            continue

        test_set_300 = set()
        for cr, cc in r["cpx"].values():
            s = coords_to_300m_set(cr, cc, res, cfg.native_res, shape_300)
            test_set_300.update(s)
        jac = jaccard_from_sets(ref_set_300, test_set_300)

        m1, d1 = centroid_match_rate(ref["centers"], r["centers"], 1000,
                                      ref["profile"]["transform"], r["profile"]["transform"])
        m3, d3 = centroid_match_rate(ref["centers"], r["centers"], 3000,
                                      ref["profile"]["transform"], r["profile"]["transform"])
        m5, d5 = centroid_match_rate(ref["centers"], r["centers"], 5000,
                                      ref["profile"]["transform"], r["profile"]["transform"])

        ratio = r["centers"].shape[0] / ref["centers"].shape[0] if ref["centers"].shape[0] > 0 else 0

        print(f"\n  {res} m:")
        print(f"    Jaccard (on 300m grid): {jac:.4f}")
        print(f"    Centroid match: 1km={m1*100:.1f}%, 3km={m3*100:.1f}%, 5km={m5*100:.1f}%")
        print(f"    Mean matched dist (1km tol): {d1:.0f} m")
        print(f"    Cluster count ratio: {ratio:.3f}")

        agreement.append({
            "resolution_m": res, "jaccard_vs_300m": round(jac, 4),
            "centroid_match_1km": round(m1 * 100, 1),
            "centroid_match_3km": round(m3 * 100, 1),
            "centroid_match_5km": round(m5 * 100, 1),
            "mean_matched_dist_m": round(d1, 1) if not np.isnan(d1) else None,
            "cluster_count_ratio": round(ratio, 3),
        })

    print("\n\n  Pairwise Jaccard Matrix (mapped to 300m grid):")
    ress = cfg.target_resolutions
    sets_300 = {}
    for res in ress:
        s = set()
        for cr, cc in results[res]["cpx"].values():
            s.update(coords_to_300m_set(cr, cc, res, cfg.native_res, shape_300))
        sets_300[res] = s

    jac_mat = np.eye(len(ress))
    for i, r1 in enumerate(ress):
        for j, r2 in enumerate(ress):
            if i < j:
                jac_mat[i, j] = jaccard_from_sets(sets_300[r1], sets_300[r2])
                jac_mat[j, i] = jac_mat[i, j]
    jac_df = pd.DataFrame(jac_mat, index=ress, columns=ress).round(4)
    print(jac_df.to_string())

    df_m = pd.DataFrame(metrics)
    df_a = pd.DataFrame(agreement)
    csv1 = os.path.join(cfg.output_dir, f"{cfg.country}_cluster_metrics.csv")
    csv2 = os.path.join(cfg.output_dir, f"{cfg.country}_cluster_agreement.csv")
    csv3 = os.path.join(cfg.output_dir, f"{cfg.country}_jaccard_matrix.csv")
    df_m.to_csv(csv1, index=False); df_a.to_csv(csv2, index=False); jac_df.to_csv(csv3)

    print("\n\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("\n  Cluster Detection:")
    print(df_m.to_string(index=False))
    print("\n  Agreement vs 300 m:")
    print(df_a.to_string(index=False))
    print(f"\nCSVs: {cfg.output_dir}")

    return df_m, df_a, jac_df


if __name__ == "__main__":
    cfg = Config()
    run_urban_cluster_sensitivity(cfg)
