"""make_radial_growth.py  -- port of make_radial_growth.m

Function for radial growth.  ``factor_2``: for a candidate with max radius r and
center c_0, all lower-ranked candidates with center c are eliminated if
dist(c, c_0) <= r * factor_2.

Consumes the list-of-dicts returned by :func:`get_radial_info`.
"""

import os
import numpy as np

from matlab_compat import (colon, matlab_round, sub2ind, ind2sub, save_mat,
                           randperm0)
from get_radial_info import get_radial_info
from get_next_best_cand_radial import get_next_best_cand_radial


def make_radial_growth(A_road, A_road_1_5, uc_map, score, search_rad,
                       new_total_length, factor_2, theta_r, factor_sector_mask,
                       radial_r, randomized, output_dir, country_name,
                       skip_radial, debug):
    A_road = np.asarray(A_road, dtype=float)
    N_r, N_c = A_road.shape
    score = np.asarray(score, dtype=float)
    A_road_update = np.asarray(A_road_1_5, dtype=float).copy()
    A_road_1_5 = np.asarray(A_road_1_5, dtype=float)

    path_list = []
    current_px_count = np.array([], dtype=float)

    if skip_radial == 0:
        if isinstance(uc_map, (tuple, list)):
            uc_map = np.asarray(uc_map[0], dtype=float)
        else:
            uc_map = np.asarray(uc_map, dtype=float)

        sorted_idx = np.argsort(-uc_map[:, 2], kind="stable")

        R_ind = []
        R_s = []
        C_ind = []
        C_s = []
        candidate_node_list = []
        current_px_count_list = []

        if debug:
            sorted_idx = sorted_idx[:2]

        for uc_idx in sorted_idx:
            cx = uc_map[uc_idx, 0]
            cy = uc_map[uc_idx, 1]

            ROI_r = matlab_round(colon(max(1, cx - search_rad), min(N_r, cx + search_rad))).astype(int)
            ROI_c = matlab_round(colon(max(1, cy - search_rad), min(N_c, cy + search_rad))).astype(int)

            cropped_A = A_road[np.ix_(ROI_r - 1, ROI_c - 1)]
            cropped_score = score[np.ix_(ROI_r - 1, ROI_c - 1)]

            current_radial, candidate_node = get_radial_info(
                cropped_A, cropped_score, radial_r, theta_r, factor_sector_mask, 0)

            if len(current_radial) > 0:
                off = np.array([ROI_r[0] - 1, ROI_c[0] - 1])

                for cn in candidate_node:
                    candidate_node_list.append(cn["end_pts"] + off)

                radial_adjusted_subs = np.vstack([cr["subs"] for cr in current_radial]) + off
                radial_stats = [cr["score"] for cr in current_radial]
                current_px_count_list.extend([cr["px_count"] for cr in current_radial])

                R_ind.extend(sub2ind((N_r, N_c), radial_adjusted_subs[:, 0],
                                     radial_adjusted_subs[:, 1]).tolist())
                R_s.extend(radial_stats)

                if len(candidate_node) > 0:
                    cand_adjusted_subs = np.vstack([cn["subs"] for cn in candidate_node]) + off
                    cand_stats = [cn["score"] for cn in candidate_node]
                    C_ind.extend(sub2ind((N_r, N_c), cand_adjusted_subs[:, 0],
                                         cand_adjusted_subs[:, 1]).tolist())
                    C_s.extend(cand_stats)

        total_len = float(np.sum(A_road))
        R_ind = np.array(R_ind, dtype=np.int64)
        current_px_count_list = np.array(current_px_count_list, dtype=float)
        if R_ind.size:
            _, id_r = np.unique(R_ind, return_index=True)
            current_px_count = current_px_count_list[id_r]
        else:
            current_px_count = np.array([], dtype=float)

        new_radial = int(matlab_round(len(current_px_count) * (new_total_length / total_len)))
        new_radial_length = int(matlab_round(np.sum(current_px_count) * (new_total_length / total_len)))
        print(f"Current number of radial centers:{len(current_px_count)}")
        print(f"Current length of radial centers:{np.sum(current_px_count):g}")
        print(f"Estimated number of new radial centers:{new_radial}")
        print(f"Estimated length of new radial centers:{new_radial_length}")
        print(f"Length of existing roads:{total_len:g}")
        print(f"Length of new additional roads to grow:{new_total_length:g}")

        C_ind = np.array(C_ind, dtype=np.int64)
        C_s = np.array(C_s, dtype=float)
        if C_ind.size:
            _, id_c = np.unique(C_ind, return_index=True)   # 0-based first occurrence
            C_s_u = C_s[id_c]
            sorted_idx2 = np.argsort(-C_s_u, kind="stable")
            picked_cand = id_c[sorted_idx2]                  # 0-based indices into C_ind
        else:
            picked_cand = np.array([], dtype=np.int64)

        if randomized and picked_cand.size:
            picked_cand = picked_cand[randperm0(len(picked_cand))]

        if picked_cand.size:
            A1, A2 = ind2sub((N_r, N_c), C_ind[picked_cand])
        else:
            A1 = np.array([], dtype=np.int64)
            A2 = np.array([], dtype=np.int64)
        A12 = np.column_stack((A1, A2)).astype(float) if picked_cand.size else np.zeros((0, 2))

        valid_cand = np.zeros(len(picked_cand))
        radial_left = new_radial_length
        print(f"Creating {new_radial} radial centers")

        for grow_idx in range(len(picked_cand)):
            if radial_left <= 0:
                break
            if valid_cand[grow_idx]:
                continue
            a1 = int(A1[grow_idx])
            a2 = int(A2[grow_idx])
            road_type = A_road_1_5[a1 - 1, a2 - 1]
            if road_type == 0:
                raise ValueError("type 0!!")
            end_idx = candidate_node_list[picked_cand[grow_idx]]
            r_max_squared = np.max(np.sum((end_idx.astype(float) - [a1, a2]) ** 2, axis=1))
            rest = slice(grow_idx + 1, len(picked_cand))
            d = np.sum((A12[rest] - [a1, a2]) ** 2, axis=1)
            valid_cand[grow_idx + 1:] = valid_cand[grow_idx + 1:] + (d <= (factor_2 ** 2) * r_max_squared)

            for i in range(end_idx.shape[0]):
                x = get_next_best_cand_radial([a1, a2], end_idx[i, :], score)
                path_list.append(x)
                A_road_update[x[:, 0] - 1, x[:, 1] - 1] = road_type + 10

            radial_left = new_radial_length - int(np.sum(A_road_update != 0)) + int(np.sum(A_road_1_5 != 0))

        if radial_left > 0:
            print(f"{radial_left} radial lengths cannot be added")

    if len(path_list) > 0:
        path_arr = np.vstack(path_list).astype(np.int64)
        added_road = np.zeros((N_r, N_c))
        added_road[path_arr[:, 0] - 1, path_arr[:, 1] - 1] = 1
        updated_road = A_road.copy()
        updated_road[path_arr[:, 0] - 1, path_arr[:, 1] - 1] = 1
        num_new_road = float(np.sum(updated_road) - np.sum(A_road))

        save_filename = os.path.join(output_dir, f"{country_name}_radial_current_and_new.mat")
        save_mat(save_filename, current_px_count=current_px_count, num_new_road=num_new_road)
    else:
        print("no new road is added")
        added_road = np.zeros((N_r, N_c))
        updated_road = A_road.copy()
        num_new_road = 0.0

    return added_road, updated_road, num_new_road, A_road_update
