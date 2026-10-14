"""Memory-saving versions of the linear-growth functions, with the same results as core/make_linear_growth.py and core/get_grown_roads.py."""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
# ---------------------------------------------------------------- linear growth

class _ArgmaxTree:
    """Segment tree giving np.argmax semantics (first index of the maximum) under point updates."""
    def __init__(self, v):
        n = v.size; size = 1
        while size < max(n, 1): size *= 2
        self.size = size; self.v = np.full(size, -np.inf); self.v[:n] = v
        self.t = np.zeros(2 * size, np.int64); self.t[size:] = np.arange(size)
        lvl = size
        while lvl > 1:
            lo = lvl // 2
            L = self.t[2 * lo:2 * lvl:2]; R = self.t[2 * lo + 1:2 * lvl:2]
            self.t[lo:lvl] = np.where(self.v[L] >= self.v[R], L, R)
            lvl = lo
    def set(self, i, x):
        self.v[i] = x; j = (i + self.size) // 2
        t, v = self.t, self.v
        while j >= 1:
            a, b = t[2 * j], t[2 * j + 1]
            t[j] = a if v[a] >= v[b] else b
            j //= 2
    def argmax(self):
        return int(self.t[1])

def make_linear_growth_lean(road_data, uc_map, score, new_total_length, road_spacing,
                            search_radius_for_urban, min_number_road_pixel, sel_mode,
                            N_iter, old_cand_map, old_path_list, old_path_length, debug):
    from matlab_compat import conv2_same, find
    import make_linear_growth as mlg
    get_next_best_cand = mlg.get_next_best_cand
    road_data = np.asarray(road_data)
    score = np.asarray(score)
    assert score.dtype == np.float64
    _undo = {}   # score is modified in place and restored on exit (same as working on a copy)
    N_r, N_c = road_data.shape
    road_data_sum = float(np.sum(road_data, dtype=np.float64))
    if old_cand_map is None or (isinstance(old_cand_map, np.ndarray) and old_cand_map.size == 0):
        print("Generating candidates")
        # == find(score .* (1*(conv2(road,ones(3),'same')>=1) - road)) for non-negative road data,
        #    evaluated only where the factor is non-zero; column-major order kept
        assert road_data.min() >= 0 and np.all(np.mod(road_data[road_data != 0], 1) == 0)
        dil = ndimage.binary_dilation(road_data != 0, structure=np.ones((3, 3), bool))
        fac = dil.astype(np.float32) - road_data.astype(np.float32)
        del dil
        cc0, rr0 = np.nonzero(((fac != 0) & (score != 0)).T)   # column-major order
        val = score[rr0, cc0] * fac[rr0, cc0].astype(np.float64)
        keep = val != 0
        a = (rr0[keep] + 1).astype(float); b = (cc0[keep] + 1).astype(float); v = val[keep]
        del fac, cc0, rr0, val, keep
        cand_map = np.zeros((len(a), 4))
        cand_map[:, 0] = a; cand_map[:, 1] = b; cand_map[:, 2] = v
        sparse_cand_map = np.zeros((N_r, N_c), np.int16)
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
        path_list = []; path_length = []; path_num = 0
    else:
        cand_map_filtered = np.asarray(old_cand_map, dtype=float).copy()
        path_list = old_path_list; path_length = old_path_length
        path_num = len(path_list)
    cand_map = cand_map_filtered
    sparse_cand_map = np.zeros((N_r, N_c), np.int16)   # small integer counts; (1 - map) stays exact
    np.add.at(sparse_cand_map, (cand_map_filtered[:, 0].astype(int) - 1,
              cand_map_filtered[:, 1].astype(int) - 1), (cand_map_filtered[:, 2] != 0).astype(np.int16))
    if debug == 0:
        N_iter = int(new_total_length)
    # rows with path id 0 never move; index their (static) positions
    free_rows = np.where(cand_map[:, 3] == 0)[0]
    tree = cKDTree(cand_map[free_rows, 0:2]) if free_rows.size else None
    sum_pl = float(sum(path_length))
    AT = _ArgmaxTree(cand_map[:, 2].copy())
    for _ in range(int(N_iter)):
        id_max = AT.argmax()
        v_max = cand_map[id_max, 2]
        if v_max == 0:
            print("no more candidates available")
            break
        row = cand_map[id_max, :].copy()
        rc = cand_map[id_max, 0:2].copy()
        if cand_map[id_max, 3] == 0:
            path_num += 1
            path_list.append([rc.copy()]); path_length.append(1)
            cand_map[id_max, 3] = path_num
        else:
            pid = int(cand_map[id_max, 3])
            path_list[pid - 1].append(rc.copy()); path_length[pid - 1] += 1
        sum_pl += 1
        _k = (int(rc[0]) - 1, int(rc[1]) - 1); _undo.setdefault(_k, score[_k]); score[_k] = 0
        if tree is not None:
            near = free_rows[tree.query_ball_point(rc, road_spacing + 1e-6)]
            if near.size:
                d = np.sqrt(np.sum((rc - cand_map[near, 0:2]) ** 2, axis=1))
                sel = near[(d <= road_spacing) & (cand_map[near, 3] == 0)]
                cand_map[sel, 2] = 0
                for j_ in sel: AT.set(int(j_), 0.0)
        cx, cy, val = get_next_best_cand(row, uc_map, score, sparse_cand_map,
                                         search_radius_for_urban, sel_mode)
        cand_map[id_max, 0:3] = [cx, cy, val]
        AT.set(id_max, float(cand_map[id_max, 2]))
        if sum_pl - road_data_sum >= new_total_length:
            print("path limit reached")
            break
    total_px = sum(len(p) for p in path_list)
    print(f"Total increased road length: {total_px}")
    path_length_arr = np.array(path_length)
    path_left = int(np.sum(path_length_arr[path_length_arr < min_number_road_pixel])) if path_length_arr.size else 0
    road_idx = (np.where(path_length_arr < min_number_road_pixel)[0] + 1) if path_length_arr.size else np.array([], dtype=int)
    print(f"length of road with less than {min_number_road_pixel} pixels: {path_left}")
    # each path id owns exactly one row of cand_map (assigned once, never reset)
    pid_col = cand_map[:, 3].astype(np.int64)
    nz = np.where(pid_col > 0)[0]
    row_of = {}
    for r_ in nz:
        row_of.setdefault(int(pid_col[r_]), int(r_))
    for idx in road_idx:
        idx = int(idx)
        if path_left > 0:
            id_next = row_of[idx]
            if cand_map[id_next, 2] == 0:
                path_list[idx - 1] = []; path_length[idx - 1] = 0
            else:
                idx2 = 0
                n_inner = path_left
                for _ in range(int(n_inner)):
                    idx2 += 1
                    rc = cand_map[id_next, 0:2].copy()
                    path_list[idx - 1].append(rc.copy())
                    _k = (int(rc[0]) - 1, int(rc[1]) - 1); _undo.setdefault(_k, score[_k]); score[_k] = 0
                    if tree is not None:
                        near = free_rows[tree.query_ball_point(rc, road_spacing + 1e-6)]
                        if near.size:
                            d = np.sqrt(np.sum((rc - cand_map[near, 0:2]) ** 2, axis=1))
                            sel = near[(d <= road_spacing) & (cand_map[near, 3] == 0)]
                            cand_map[sel, 2] = 0
                    cx, cy, val = get_next_best_cand(cand_map[id_next, :], uc_map, score,
                                                     sparse_cand_map, search_radius_for_urban, sel_mode)
                    cand_map[id_next, 0:3] = [cx, cy, val]
                    if val == 0:
                        break
                if path_length[idx - 1] + idx2 >= min_number_road_pixel:
                    path_length[idx - 1] = path_length[idx - 1] + idx2
                    path_left = path_left - path_length[idx - 1]
                else:
                    path_list[idx - 1] = []; path_length[idx - 1] = 0
                    cand_map[id_next, 2] = 0
        else:
            path_list[idx - 1] = []; path_length[idx - 1] = 0
            if idx in row_of:
                cand_map[row_of[idx], 2] = 0
    print(f"{path_left}road left to be added")
    total_px = sum(len(p) for p in path_list)
    print(f"Total increased road length: {total_px}")
    new_rd = np.zeros((N_r, N_c), np.uint8)
    for p in path_list:
        if p:
            pa = np.vstack(p).astype(int)
            new_rd[pa[:, 0] - 1, pa[:, 1] - 1] = 1
    added_road = new_rd
    updated_road = None   # not used by get_grown_roads
    num_new_road = float(np.count_nonzero((new_rd != 0) | (road_data != 0)) - np.sum(road_data, dtype=np.float64))
    for (r_, c_), v_ in _undo.items():
        score[r_, c_] = v_
    return (added_road, updated_road, num_new_road, path_list, path_length, cand_map, None)


def get_grown_roads_lean(remove_iteration, min_road_sep_1, min_road_sep_2, start_road,
                         all_high_roads, uc_map, score_map, road_demand, road_spacing,
                         search_radius_for_urban, min_number_road_pixel, sel_mode,
                         N_iter, debug):
    from matlab_compat import find_linear, ind2sub, pdist2_smallest
    import get_grown_roads as ggr
    _nanmean = ggr._nanmean
    A1 = np.asarray(start_road)
    assert A1.dtype == np.uint8
    score_2 = np.asarray(score_map, dtype=float)
    N_r, N_c = A1.shape
    all_high_roads = np.asarray(all_high_roads)
    road_grown = 0.0; old_cand_map = None; path_list = []; old_path_length = []
    processed_road_2 = A1.copy(); new_rd = np.zeros((N_r, N_c), np.uint8); added_road = np.zeros((N_r, N_c), np.uint8)
    for remove_iter in range(1, remove_iteration + 1):
        (added_road, updated_road, num_new_road, path_list, old_path_length,
         old_cand_map, _) = make_linear_growth_lean(
            start_road, uc_map, score_map, road_demand - road_grown, road_spacing,
            search_radius_for_urban, min_number_road_pixel, sel_mode, N_iter,
            old_cand_map, path_list, old_path_length, debug)
        P1 = list(path_list)
        idx_removed_paths = []
        for idx in range(1, len(P1) + 1):
            if P1[idx - 1]:
                road_idx_mask = np.vstack(P1[idx - 1]).astype(float)
                mx = road_idx_mask.max(axis=0); mn = road_idx_mask.min(axis=0)
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
                    path_list[idx - 1] = []; old_path_length[idx - 1] = 0
                    idx_removed_paths.append(idx)
        # parallel removal; pairs whose bounding boxes are >= min_road_sep_2 apart
        # can never satisfy avg(min dist) < min_road_sep_2, so skipping them is exact
        P2 = list(path_list)
        n = len(P2)
        if n > 1:
            bb = np.full((n, 4), np.nan)
            for k in range(n):
                if P2[k]:
                    m_ = np.vstack(P2[k]); bb[k] = [m_[:, 0].min(), m_[:, 0].max(), m_[:, 1].min(), m_[:, 1].max()]
            valid = ~np.isnan(bb[:, 0])
            for idx in range(1, n):
                if not P2[idx - 1]:
                    continue
                b0 = bb[idx - 1]
                rest = np.arange(idx, n)
                gr = np.maximum(0, np.maximum(bb[rest, 0] - b0[1], b0[0] - bb[rest, 1]))
                gc = np.maximum(0, np.maximum(bb[rest, 2] - b0[3], b0[2] - bb[rest, 3]))
                close = valid[rest] & (np.sqrt(gr ** 2 + gc ** 2) < min_road_sep_2)
                for j in rest[close]:
                    idx2 = int(j) + 1
                    if P2[idx - 1] and P2[idx2 - 1]:
                        if len(P2[idx - 1]) > len(P2[idx2 - 1]):
                            b = idx; a = idx2
                        else:
                            a = idx; b = idx2
                        mask_a = np.vstack(P2[a - 1]).astype(float)
                        mask_b = np.vstack(P2[b - 1]).astype(float)
                        avg = _nanmean(pdist2_smallest(mask_b, mask_a, "euclidean", 1))
                        if avg < min_road_sep_2:
                            P2[a - 1] = []; path_list[a - 1] = []; old_path_length[a - 1] = 0
                            idx_removed_paths.append(a)
                    if not P2[idx - 1]:
                        break
        start_road = None; score_map = None   # release previous iteration's arrays (A1/score_2 kept)
        new_rd = np.zeros((N_r, N_c), np.uint8)
        for p in P2:
            if p:
                pa = np.vstack(p).astype(int); new_rd[pa[:, 0] - 1, pa[:, 1] - 1] = 1
        processed_road_2 = A1.astype(np.float32); processed_road_2 += new_rd   # values 0/1/2 exact
        updated_score = score_2 * (1 - processed_road_2)
        col3 = old_cand_map[:, 3]
        if idx_removed_paths:
            old_cand_map[np.isin(col3, np.array(idx_removed_paths, dtype=float)), 2] = 0
        live = np.array([k + 1 for k in range(len(path_list)) if path_list[k]], dtype=float)
        if live.size:
            rows_live = np.where(np.isin(col3, live))[0]
            ids = col3[rows_live]
            uid, first = np.unique(ids, return_index=True)
            fr = rows_live[first]
            val_of = dict(zip(uid.tolist(), updated_score[old_cand_map[fr, 0].astype(int) - 1,
                                                          old_cand_map[fr, 1].astype(int) - 1].tolist()))
            old_cand_map[rows_live, 2] = np.array([val_of[i] for i in ids.tolist()])
        road_grown = float(np.sum(processed_road_2, dtype=np.float64) - np.sum(A1, dtype=np.float64))
        start_road = processed_road_2
        score_map = updated_score
        print(f"road grown: {road_grown:g} iteration: {remove_iter}")
        if road_grown >= road_demand:
            print("stopping growth process"); break
        if np.max(old_cand_map[:, 2]) <= 0:
            print("Warning: max score becomes zero; terminating the growth process"); break
        if np.sum(old_cand_map[:, 0] == 0) or np.sum(old_cand_map[:, 1] == 0):
            print("Warning: candidate index is zero")
    return processed_road_2, new_rd, road_grown, added_road

