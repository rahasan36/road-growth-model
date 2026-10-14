"""get_grown_roads.py  -- port of get_grown_roads.m

Iterates the "linear growth + post-processing" cycle: grow roads, drop paths
that run too close to existing roads, drop near-parallel duplicates among the
newly grown roads, then refresh the candidate scores for the next iteration.
"""

import numpy as np
from matlab_compat import find_linear, ind2sub, pdist2_smallest
from make_linear_growth import make_linear_growth


def _nanmean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return float(np.mean(x))


def get_grown_roads(remove_iteration, min_road_sep_1, min_road_sep_2, start_road,
                    all_high_roads, uc_map, score_map, road_demand, road_spacing,
                    search_radius_for_urban, min_number_road_pixel, sel_mode,
                    N_iter, debug):
    A1 = np.asarray(start_road, dtype=float)
    score_2 = np.asarray(score_map, dtype=float)
    N_r, N_c = A1.shape
    all_high_roads = np.asarray(all_high_roads)

    road_grown = 0.0
    old_cand_map = None
    path_list = []
    old_path_length = []

    # defaults so the function still returns sensibly if the loop body changes
    processed_road_2 = A1.copy()
    new_rd = np.zeros((N_r, N_c))
    added_road = np.zeros((N_r, N_c))

    for remove_iter in range(1, remove_iteration + 1):
        (added_road, updated_road, num_new_road, path_list, old_path_length,
         old_cand_map, _) = make_linear_growth(
            start_road, uc_map, score_map, road_demand - road_grown, road_spacing,
            search_radius_for_urban, min_number_road_pixel, sel_mode, N_iter,
            old_cand_map, path_list, old_path_length, debug)

        # ---- drop paths too close to existing roads ----------------------
        P1 = list(path_list)  # snapshot (element reassignment won't affect it)
        idx_removed_paths = []
        for idx in range(1, len(P1) + 1):
            if P1[idx - 1]:
                road_idx_mask = np.vstack(P1[idx - 1]).astype(float)
                mx = road_idx_mask.max(axis=0)
                mn = road_idx_mask.min(axis=0)
                bd_up = np.minimum(mx + min_road_sep_1, [N_r, N_c]).astype(int)
                bd_low = np.maximum(mn - min_road_sep_1, [1, 1]).astype(int)
                A_temp = all_high_roads[bd_low[0] - 1:bd_up[0], bd_low[1] - 1:bd_up[1]]
                lin = find_linear(A_temp)
                if lin.size:
                    rx, ry = ind2sub(A_temp.shape, lin)
                    pts = np.column_stack((rx, ry)).astype(float) + (bd_low - 1)
                    avg = _nanmean(pdist2_smallest(pts, road_idx_mask, "euclidean", 1))
                else:
                    avg = np.nan
                if avg < min_road_sep_1:
                    path_list[idx - 1] = []
                    old_path_length[idx - 1] = 0
                    idx_removed_paths.append(idx)

        # ---- drop near-parallel duplicates among grown roads -------------
        P2 = list(path_list)
        n = len(P2)
        for idx in range(1, n):
            for idx2 in range(idx + 1, n + 1):
                if P2[idx - 1] and P2[idx2 - 1]:
                    if len(P2[idx - 1]) > len(P2[idx2 - 1]):
                        b = idx
                        a = idx2
                    else:
                        a = idx
                        b = idx2
                    mask_a = np.vstack(P2[a - 1]).astype(float)
                    mask_b = np.vstack(P2[b - 1]).astype(float)
                    avg = _nanmean(pdist2_smallest(mask_b, mask_a, "euclidean", 1))
                    if avg < min_road_sep_2:
                        P2[a - 1] = []
                        path_list[a - 1] = []
                        old_path_length[a - 1] = 0
                        idx_removed_paths.append(a)

        # ---- rasterise surviving roads ----------------------------------
        new_rd = np.zeros((N_r, N_c))
        for p in P2:
            if p:
                for coord in p:
                    new_rd[int(coord[0]) - 1, int(coord[1]) - 1] = 1

        processed_road_2 = A1 + new_rd
        updated_score = score_2 * (1 - processed_road_2)

        for idx in idx_removed_paths:
            old_cand_map[old_cand_map[:, 3] == idx, 2] = 0

        for idx in range(1, len(path_list) + 1):
            if path_list[idx - 1]:
                temp_mask = old_cand_map[:, 3] == idx
                temp_info = old_cand_map[temp_mask, :]
                r = int(temp_info[0, 0])
                c = int(temp_info[0, 1])
                old_cand_map[temp_mask, 2] = updated_score[r - 1, c - 1]

        road_grown = float(np.sum(processed_road_2) - np.sum(A1))
        start_road = processed_road_2
        score_map = updated_score

        print(f"road grown: {road_grown:g} iteration: {remove_iter}")

        if road_grown >= road_demand:
            print("stopping growth process")
            break

        if np.max(old_cand_map[:, 2]) <= 0:
            print("Warning: max score becomes zero; terminating the growth process")
            break

        if np.sum(old_cand_map[:, 0] == 0) or np.sum(old_cand_map[:, 1] == 0):
            print("Warning: candidate index is zero")

    return processed_road_2, new_rd, road_grown, added_road
