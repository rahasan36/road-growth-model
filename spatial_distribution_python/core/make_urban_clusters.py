"""make_urban_clusters.py  -- port of make_urban_clusters.m

Identifies urban center clusters via DBSCAN and records, for each cluster, the
centroid (row, col), normalised mean population density, pixel count, and
normalised mean height.  Saves ``<country>_uc_data.mat`` (consumed by
make_road) and ``<country>_urban_cluster_centers.tif``.
"""

import os
import numpy as np
from sklearn.cluster import DBSCAN

from matlab_compat import geotiffread, geotiffwrite, find, sub2ind, save_mat, matlab_round


def _last_is_one(s):
    try:
        return float(s[-1]) == 1
    except (ValueError, IndexError):
        return False


def make_urban_clusters(input_dir, output_dir, country_name, base_year):
    if base_year == 0 or _last_is_one(input_dir):
        file_name = os.path.join(input_dir, f"{country_name}_urban.tif")
        height_file = os.path.join(input_dir, f"{country_name}_height.tif")
    else:
        temp_dir = input_dir[:-1] + "1"
        file_name = os.path.join(temp_dir, f"{country_name}_urban.tif")
        height_file = os.path.join(temp_dir, f"{country_name}_height.tif")

    pop_dens_file = os.path.join(input_dir, f"{country_name}_pop_density_total.tif")

    min_pixel_count = 5
    save_filename = os.path.join(output_dir, f"{country_name}_uc_data.mat")

    epsilon = 3
    minpts = 5

    A, R = geotiffread(file_name)
    A = np.asarray(A, dtype=float)
    urban_data = A / np.max(A)
    N_r, N_c = urban_data.shape

    u, v, _ = find(urban_data)  # 1-based, column-major order
    pts = np.column_stack((u, v)).astype(float)
    idx = DBSCAN(eps=epsilon, min_samples=minpts, metric="euclidean").fit_predict(pts)

    cluster_id = np.unique(idx)
    cluster_id = cluster_id[1:]  # drop first label (noise = -1), as in MATLAB
    num_pts = np.array([np.sum(idx == cid) for cid in cluster_id], dtype=np.int64)

    print(f"total number of clusters:{len(num_pts)}")
    keep = num_pts >= min_pixel_count
    cluster_id = cluster_id[keep]
    num_pts = num_pts[keep]
    print(f"total number of clusters with at least {min_pixel_count} pixels : {len(num_pts)}")

    A, R = geotiffread(pop_dens_file)
    pop_data = np.asarray(A, dtype=float)
    pop_data[pop_data < 0] = 0
    pop_data = pop_data / np.max(pop_data)

    A, R = geotiffread(height_file)
    height_data = np.asarray(A, dtype=float) / np.max(np.asarray(A, dtype=float))

    if pop_data.shape == urban_data.shape:
        urban_center = np.zeros((len(cluster_id), 5))
    else:
        raise ValueError(
            f"dimension of population data: {pop_data.shape}\n"
            f"dimension of urban data: {urban_data.shape}")

    for i in range(len(cluster_id)):
        mask = idx == cluster_id[i]
        um = u[mask]
        vm = v[mask]
        urban_center[i, 0] = np.mean(um)
        urban_center[i, 1] = np.mean(vm)
        urban_center[i, 2] = np.mean(pop_data[um - 1, vm - 1])
        urban_center[i, 4] = np.mean(height_data[um - 1, vm - 1])
        urban_center[i, 3] = np.sum(mask)

    urban_center[:, 2] = urban_center[:, 2] / np.max(urban_center[:, 2])
    urban_center[:, 4] = urban_center[:, 4] / np.max(urban_center[:, 4])

    save_mat(save_filename, urban_center=urban_center)

    urban_center_r = matlab_round(urban_center)
    mask_uc = np.zeros((N_r, N_c))
    mask_uc[urban_center_r[:, 0].astype(int) - 1, urban_center_r[:, 1].astype(int) - 1] = 1

    geotiffwrite(os.path.join(output_dir, f"{country_name}_urban_cluster_centers.tif"),
                 mask_uc, R)
