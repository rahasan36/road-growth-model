"""get_next_best_cand.py  -- port of get_next_best_cand.m

After each road pixel is predicted, the remaining suitable neighbouring pixels
are ranked.  Returns (cx, cy, val): the next road cell (1-based) and its score.

``uc_map`` is either an (M, >=5) array of urban centers, or, for the
'dist_only' + POI mode, a tuple ``(uc_array, poi_array, uc_poi_weight_ratio)``
mirroring the MATLAB cell array.
"""

import numpy as np
from get_adjusted_roi import get_adjusted_ROI


def _weighted_nearest(uc_map, cand, weights_num, search_radius, invert=False,
                      denom_extra=None):
    """Shared logic for the 'dist_*_combo' selection modes.

    weights = weights_num / dist   (or 1/(weights_num*dist) when invert=True)
    """
    dist_from_current = np.maximum(np.sum((uc_map[:, 0:2] - cand[0:2]) ** 2, axis=1), 1e-3)
    if invert:
        weights = 1.0 / (weights_num * dist_from_current)
    else:
        weights = weights_num / dist_from_current
    if search_radius != 0:
        weights = weights.copy()
        weights[dist_from_current > search_radius ** 2] = 0
    sorted_idx = np.argsort(-weights, kind="stable")
    if weights[sorted_idx[0]] - weights[sorted_idx[1]] <= 0.05 * weights[sorted_idx[0]]:
        pair = dist_from_current[sorted_idx[0:2]]
        sel_idx = int(np.argmin(pair))
        nearest_uc = sorted_idx[sel_idx]
    else:
        nearest_uc = sorted_idx[0]
    nn_uc_dist = dist_from_current[nearest_uc]
    return int(nearest_uc), float(nn_uc_dist)


def get_next_best_cand(cand, uc_map, score, sparse_cand_map, search_radius, sel_mode):
    cand = np.asarray(cand, dtype=float)
    N_r, N_c = score.shape
    ROI_r, ROI_c = get_adjusted_ROI(int(cand[0]), int(cand[1]), 1, N_r, 1, N_c)

    is_cell = isinstance(uc_map, (tuple, list))
    sel_uc = None
    nearest_uc = None

    if sel_mode == "dist_only":
        if is_cell:
            uc_map_temp = np.asarray(uc_map[0], dtype=float)
            poi_map_temp = np.asarray(uc_map[1], dtype=float)
            ratio = uc_map[2]
            d_uc = np.sum((uc_map_temp[:, 0:2] - cand[0:2]) ** 2, axis=1)
            nearest_uc = int(np.argmin(d_uc))
            nn_uc_dist = float(d_uc[nearest_uc])
            d_poi = np.sum((poi_map_temp[:, 0:2] - cand[0:2]) ** 2, axis=1)
            nearest_poi = int(np.argmin(d_poi))
            nn_poi_dist = float(d_poi[nearest_poi])
            if nn_uc_dist / ratio > nn_poi_dist:
                if search_radius == 0:
                    nn_uc_dist = nn_poi_dist
                    sel_uc = poi_map_temp[nearest_poi, 0:2]
                elif nn_poi_dist <= search_radius ** 2:
                    nn_uc_dist = nn_poi_dist
                    sel_uc = poi_map_temp[nearest_poi, 0:2]
            else:
                sel_uc = uc_map_temp[nearest_uc, 0:2]
        else:
            uc_arr = np.asarray(uc_map, dtype=float)
            d_uc = np.sum((uc_arr[:, 0:2] - cand[0:2]) ** 2, axis=1)
            nearest_uc = int(np.argmin(d_uc))
            nn_uc_dist = float(d_uc[nearest_uc])
    elif sel_mode == "dist_pop_combo":
        uc_arr = np.asarray(uc_map, dtype=float)
        nearest_uc, nn_uc_dist = _weighted_nearest(uc_arr, cand, uc_arr[:, 2], search_radius)
    elif sel_mode == "dist_height_combo":
        uc_arr = np.asarray(uc_map, dtype=float)
        nearest_uc, nn_uc_dist = _weighted_nearest(uc_arr, cand, uc_arr[:, 4], search_radius)
    elif sel_mode == "dist_pop_height_combo":
        uc_arr = np.asarray(uc_map, dtype=float)
        nearest_uc, nn_uc_dist = _weighted_nearest(
            uc_arr, cand, 0.5 * (uc_arr[:, 2] + uc_arr[:, 4]), search_radius)
    elif sel_mode == "dist_pop_combo_inv":
        uc_arr = np.asarray(uc_map, dtype=float)
        nearest_uc, nn_uc_dist = _weighted_nearest(uc_arr, cand, uc_arr[:, 2], search_radius, invert=True)
    elif sel_mode == "dist_pop_urban":
        uc_arr = np.asarray(uc_map, dtype=float)
        nearest_uc, nn_uc_dist = _weighted_nearest(
            uc_arr, cand, uc_arr[:, 2] * uc_arr[:, 3], search_radius)
    elif sel_mode == "dist_urban_inv":
        uc_arr = np.asarray(uc_map, dtype=float)
        nearest_uc, nn_uc_dist = _weighted_nearest(uc_arr, cand, uc_arr[:, 3], search_radius, invert=True)
    else:
        raise ValueError(f"{sel_mode} is not implemented")

    if not is_cell:
        uc_arr = np.asarray(uc_map, dtype=float)
        sel_uc = uc_arr[nearest_uc, 0:2]

    if search_radius == 0 or nn_uc_dist <= search_radius ** 2:
        coords = np.column_stack((ROI_r, ROI_c)).astype(float)
        d2 = np.sum((coords - sel_uc) ** 2, axis=1)
        sorted_idx = np.argsort(d2, kind="stable")
        dir_score = np.zeros(len(ROI_r))
        dir_score[sorted_idx[:3]] = 1
        sc = score[ROI_r - 1, ROI_c - 1]
        scm = sparse_cand_map[ROI_r - 1, ROI_c - 1]
        prod = dir_score * sc * (1 - scm)
        idx_nn = int(np.argmax(prod))
        val = float(prod[idx_nn])
        cx = int(ROI_r[idx_nn])
        cy = int(ROI_c[idx_nn])
    else:
        val = 0.0
        cx = int(cand[0])
        cy = int(cand[1])

    return cx, cy, val
