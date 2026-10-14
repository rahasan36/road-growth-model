"""make_linear_growth.py  -- port of make_linear_growth.m

Simulates one pass of linear road growth.

Data structures (mirroring the MATLAB originals):
  * ``cand_map``  : (N, 4) float array. cols 0,1 = row,col (1-based);
                    col 2 = score value; col 3 = path id (0 = unassigned).
  * ``path_list`` : list whose k-th element is a list of np.array([r, c])
                    coordinate pairs for path id (k+1).  An empty list means
                    the path was removed (MATLAB ``path_list{idx} = []``).
  * ``path_length``: list of ints parallel to ``path_list``.

``uc_map`` is an array, or a tuple (uc, poi, ratio) for the POI mode.
"""

import numpy as np
from matlab_compat import conv2_same, find
from get_next_best_cand import get_next_best_cand


def make_linear_growth(road_data, uc_map, score, new_total_length, road_spacing,
                       search_radius_for_urban, min_number_road_pixel, sel_mode,
                       N_iter, old_cand_map, old_path_list, old_path_length, debug):
    road_data = np.asarray(road_data, dtype=float)
    # MATLAB passes ``score`` by value; copy so the caller's array is untouched.
    score = np.array(score, dtype=float, copy=True)
    N_r, N_c = road_data.shape
    road_data_sum = float(np.sum(road_data))

    # ------------------------------------------------------------------
    # construct candidates
    # ------------------------------------------------------------------
    if old_cand_map is None or (isinstance(old_cand_map, np.ndarray) and old_cand_map.size == 0):
        print("Generating candidates")
        cand_A = 1 * (conv2_same(road_data, np.ones((3, 3))) >= 1)
        cand_A = cand_A - road_data
        a, b, v = find(score * cand_A)
        cand_map = np.zeros((len(a), 4))
        cand_map[:, 0] = a
        cand_map[:, 1] = b
        cand_map[:, 2] = v

        sparse_cand_map = np.zeros((N_r, N_c))
        sparse_cand_map[cand_map[:, 0].astype(int) - 1, cand_map[:, 1].astype(int) - 1] = 1

        cand_map_filtered = cand_map.copy()
        loop_n = cand_map.shape[0]
        if debug:
            loop_n = min(100, loop_n)
        print(f"Number of initial candidates: {cand_map_filtered.shape[0]}")

        for idx in range(loop_n):
            _, _, val = get_next_best_cand(cand_map[idx, :], uc_map, score,
                                           sparse_cand_map, search_radius_for_urban, sel_mode)
            if val == 0:
                cand_map_filtered[idx, 2] = 0

        path_list = []
        path_length = []
        path_num = 0
    else:
        cand_map_filtered = np.asarray(old_cand_map, dtype=float).copy()
        path_list = old_path_list
        path_length = old_path_length
        path_num = len(path_list)

    # ------------------------------------------------------------------
    # grow road
    # ------------------------------------------------------------------
    cand_map = cand_map_filtered
    sparse_cand_map = np.zeros((N_r, N_c))
    sparse_cand_map[cand_map_filtered[:, 0].astype(int) - 1,
                    cand_map_filtered[:, 1].astype(int) - 1] = 1 * (cand_map_filtered[:, 2] != 0)

    if debug == 0:
        N_iter = int(new_total_length)

    for _ in range(int(N_iter)):
        id_max = int(np.argmax(cand_map[:, 2]))
        v_max = cand_map[id_max, 2]
        if v_max == 0:
            print("no more candidates available")
            break

        row = cand_map[id_max, :].copy()
        rc = cand_map[id_max, 0:2].copy()
        if cand_map[id_max, 3] == 0:
            path_num += 1
            path_list.append([rc.copy()])
            path_length.append(1)
            cand_map[id_max, 3] = path_num
        else:
            pid = int(cand_map[id_max, 3])
            path_list[pid - 1].append(rc.copy())
            path_length[pid - 1] += 1

        score[int(rc[0]) - 1, int(rc[1]) - 1] = 0
        dist = np.sqrt(np.sum((rc - cand_map[:, 0:2]) ** 2, axis=1))
        dist_mask = (dist <= road_spacing) & (cand_map[:, 3] == 0)
        cand_map[dist_mask, 2] = 0

        cx, cy, val = get_next_best_cand(row, uc_map, score, sparse_cand_map,
                                         search_radius_for_urban, sel_mode)
        cand_map[id_max, 0:3] = [cx, cy, val]

        if sum(path_length) - road_data_sum >= new_total_length:
            print("path limit reached")
            break

    total_px = sum(len(p) for p in path_list)
    print(f"Total increased road length: {total_px}")

    # ------------------------------------------------------------------
    # min pixel handling: try to extend too-short paths; remove if impossible
    # ------------------------------------------------------------------
    path_length_arr = np.array(path_length)
    path_left = int(np.sum(path_length_arr[path_length_arr < min_number_road_pixel])) if path_length_arr.size else 0
    road_idx = (np.where(path_length_arr < min_number_road_pixel)[0] + 1) if path_length_arr.size else np.array([], dtype=int)

    print(f"length of road with less than {min_number_road_pixel} pixels: {path_left}")

    for idx in road_idx:
        idx = int(idx)
        if path_left > 0:
            id_next_arr = np.where(cand_map[:, 3] == idx)[0]
            id_next = int(id_next_arr[0])
            if cand_map[id_next, 2] == 0:
                path_list[idx - 1] = []
                path_length[idx - 1] = 0
            else:
                idx2 = 0
                n_inner = path_left
                for _ in range(int(n_inner)):
                    idx2 += 1
                    rc = cand_map[id_next, 0:2].copy()
                    path_list[idx - 1].append(rc.copy())
                    score[int(rc[0]) - 1, int(rc[1]) - 1] = 0
                    dist = np.sqrt(np.sum((rc - cand_map[:, 0:2]) ** 2, axis=1))
                    dist_mask = (dist <= road_spacing) & (cand_map[:, 3] == 0)
                    cand_map[dist_mask, 2] = 0
                    cx, cy, val = get_next_best_cand(cand_map[id_next, :], uc_map, score,
                                                     sparse_cand_map, search_radius_for_urban, sel_mode)
                    cand_map[id_next, 0:3] = [cx, cy, val]
                    if val == 0:
                        break
                if path_length[idx - 1] + idx2 >= min_number_road_pixel:
                    path_length[idx - 1] = path_length[idx - 1] + idx2
                    path_left = path_left - path_length[idx - 1]
                else:
                    path_list[idx - 1] = []
                    path_length[idx - 1] = 0
                    cand_map[cand_map[:, 3] == idx, 2] = 0
        else:
            path_list[idx - 1] = []
            path_length[idx - 1] = 0
            cand_map[cand_map[:, 3] == idx, 2] = 0

    print(f"{path_left}road left to be added")
    total_px = sum(len(p) for p in path_list)
    print(f"Total increased road length: {total_px}")

    # ------------------------------------------------------------------
    # rasterise grown paths
    # ------------------------------------------------------------------
    new_rd = np.zeros((N_r, N_c))
    for p in path_list:
        if p:
            for coord in p:
                new_rd[int(coord[0]) - 1, int(coord[1]) - 1] = 1

    added_road = new_rd
    updated_road = new_rd + road_data
    updated_road = (updated_road != 0) * 1.0
    num_new_road = float(np.sum(updated_road) - np.sum(road_data))

    return (added_road, updated_road, num_new_road, path_list, path_length,
            cand_map, score)
