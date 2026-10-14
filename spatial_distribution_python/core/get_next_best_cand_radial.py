"""get_next_best_cand_radial.py  -- port of get_next_best_cand_radial.m

Ranking radial candidates: walk from ``start_pt`` toward ``uc_map`` (an end
point), greedily following the 3 cells most aligned toward the target that have
the highest score, until the target is reached or no scored cell remains.

Returns ``path_list`` as an (N, 2) array of 1-based [row, col] coordinates.
"""

import numpy as np
from get_adjusted_roi import get_adjusted_ROI


def get_next_best_cand_radial(start_pt, uc_map, score):
    # MATLAB passes ``score`` by value; copy so we never mutate the caller's.
    score = np.array(score, dtype=float, copy=True)
    N_r, N_c = score.shape

    start_pt = np.asarray(start_pt, dtype=np.int64).reshape(2)
    uc_map = np.asarray(uc_map, dtype=float).reshape(2)

    path_list = [start_pt.copy()]
    cx = int(start_pt[0])
    cy = int(start_pt[1])

    while True:
        ROI_r, ROI_c = get_adjusted_ROI(cx, cy, 1, N_r, 1, N_c)
        coords = np.column_stack((ROI_r, ROI_c)).astype(float)
        d2 = np.sum((coords - uc_map) ** 2, axis=1)
        sorted_idx = np.argsort(d2, kind="stable")
        dir_score = np.zeros(len(ROI_r))
        dir_score[sorted_idx[:3]] = 1
        sc = score[ROI_r - 1, ROI_c - 1]
        prod = dir_score * sc
        idx_nn = int(np.argmax(prod))
        val = prod[idx_nn]
        cx = int(ROI_r[idx_nn])
        cy = int(ROI_c[idx_nn])
        path_list.append(np.array([cx, cy], dtype=np.int64))
        score[cx - 1, cy - 1] = 0
        if np.sum((np.array([cx, cy], dtype=float) - uc_map) ** 2) <= 2 or val == 0:
            if val != 0:
                path_list.append(uc_map.astype(np.int64))
            break

    return np.array(path_list, dtype=np.int64)
