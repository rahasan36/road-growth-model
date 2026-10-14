"""get_upgraded_points.py  -- port of get_upgraded_points.m

Identifies which road pixels of A2 (the next-higher road class) are upgraded,
based on intersections with A1 and a per-intersection upgrade radius scaled by
local road density.  ``randperm`` is used to randomise the processing order
(seed-gated via matlab_compat.set_seed).
"""

import numpy as np
from matlab_compat import bwconncomp, ind2sub, sub2ind, pdist2_smallest, randperm0


def _get_subs(linidx, N_r, N_c):
    r, c = ind2sub((N_r, N_c), np.asarray(linidx))
    return np.column_stack((r, c)).astype(np.int64)


def get_upgraded_points(A1, A2, road_density_map, max_radius, upgrade_length,
                        min_upgrade_per_intersect):
    A1 = np.asarray(A1)
    A2 = np.asarray(A2)
    N_r, N_c = A1.shape

    x1 = [_get_subs(p, N_r, N_c) for p in bwconncomp(A1, 8)]
    x2 = [_get_subs(p, N_r, N_c) for p in bwconncomp(A2, 8)]

    intersect_blocks = []
    segment_id = []
    sqrt2 = np.sqrt(2)
    for i1 in range(len(x1)):
        for i2 in range(len(x2)):
            b = pdist2_smallest(x1[i1], x2[i2], "euclidean", 1)
            temp = b <= sqrt2
            if np.any(temp):
                intersect_blocks.append(x2[i2][temp, :])
                segment_id.extend([(i2 + 1)] * int(np.sum(temp)))

    if len(intersect_blocks) == 0:
        return np.empty((0, 2), dtype=np.int64), A2.copy()

    intersect_pts = np.vstack(intersect_blocks)
    segment_id = np.asarray(segment_id, dtype=np.int64)  # 1-based A2 component id

    lin = sub2ind((N_r, N_c), intersect_pts[:, 0], intersect_pts[:, 1])
    lin_unique, ia = np.unique(lin, return_index=True)  # sorted asc, first occurrence
    segment_id = segment_id[ia]
    u, v = ind2sub((N_r, N_c), lin_unique)
    intersect_pts = np.column_stack((u, v)).astype(np.int64)

    density_vals = road_density_map[intersect_pts[:, 0] - 1, intersect_pts[:, 1] - 1].astype(float)
    upgrade_radius_per_point = max_radius * density_vals / np.max(density_vals)

    A2_upgrade = A2.astype(float).copy()
    upgrade_left = upgrade_length
    to_be_upgraded = []

    rand_order = randperm0(len(segment_id))
    for idx in rand_order:
        if upgrade_left <= 0:
            break
        comp = x2[segment_id[idx] - 1]
        d2 = np.sum((intersect_pts[idx, :].astype(float) - comp.astype(float)) ** 2, axis=1)
        mask = d2 <= upgrade_radius_per_point[idx] ** 2
        pts = comp[mask, :]
        rows = pts[:, 0] - 1
        cols = pts[:, 1] - 1
        # number of newly-set -1 cells
        new_count = int(np.sum(A2_upgrade[rows, cols] != -1))
        if new_count >= min_upgrade_per_intersect:
            to_be_upgraded.append(pts)
            A2_upgrade[rows, cols] = -1
            upgrade_left = upgrade_length - int(np.sum(A2_upgrade == -1))

    if len(to_be_upgraded) == 0:
        return np.empty((0, 2), dtype=np.int64), A2_upgrade
    return np.vstack(to_be_upgraded).astype(np.int64), A2_upgrade
