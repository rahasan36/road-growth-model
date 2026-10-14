"""Faster versions of the slowest functions of the core modules, with the same results."""
import itertools
import numpy as np
import networkx as nx
from scipy import ndimage
import matlab_compat as mc
import get_radial_info as gri_mod
from matlab_compat import (ind2sub, sub2ind, bwlabeln, bwconncomp, find_linear,
                           wrapTo360, atan2d, sind, cosd, matlab_round)
from make_skel import make_skel

def label2idx_fast(L):
    L = np.asarray(L)
    n = int(L.max()) if L.size else 0
    R = L.shape[0]
    rows, cols = np.nonzero(L)
    labs = L[rows, cols]
    lin = cols.astype(np.int64) * R + rows + 1
    order = np.lexsort((lin, labs))
    labs_s = labs[order]; lin_s = lin[order]
    bounds = np.searchsorted(labs_s, np.arange(1, n + 2))
    return [lin_s[bounds[k]:bounds[k + 1]].astype(np.int64) for k in range(n)]

def bwconncomp_fast(BW, conn=8):
    L, _ = bwlabeln(BW, conn)
    return label2idx_fast(L)

def _connected_nodes(shape, node_list, pixel_lists):
    """connected_nodes[e] = ascending list of node ids whose pixels lie within
    squared distance 2 (8-neighbourhood incl. overlap) of any pixel of edge e."""
    N_r, N_c = shape
    E = np.zeros((N_r + 2, N_c + 2), dtype=np.int64)  # padded edge-label image
    for e_idx, p in enumerate(pixel_lists):
        r, c = ind2sub(shape, p)
        E[r, c] = e_idx + 1  # +1 pad offset cancels 1-based
    pairs = set()
    for node_idx in range(1, len(node_list) + 1):
        r, c = ind2sub(shape, node_list[node_idx - 1])
        labs = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                v = E[r + dr, c + dc]
                labs.update(v[v > 0].tolist())
        for l in labs:
            pairs.add((l - 1, node_idx))
    connected = [[] for _ in range(len(pixel_lists))]
    for e, n in sorted(pairs, key=lambda t: (t[1], t[0])):
        connected[e].append(n)
    return connected

def get_radial_info_fast(cropped_A, cropped_score, radial_r, theta_r,
                         factor_sector_mask, show_graph=0):
    cropped_A = np.asarray(cropped_A)
    cropped_score = np.asarray(cropped_score, dtype=float)
    N_r, N_c = cropped_A.shape
    cropped_score_F = cropped_score.flatten(order="F")
    br_mask, ep_mask = make_skel(cropped_A)
    disjoint_img = cropped_A - br_mask - ep_mask
    BW = br_mask + ep_mask
    L, _ = bwlabeln(BW, 8)
    node_list = label2idx_fast(L)
    pixel_lists = bwconncomp_fast(disjoint_img, 8)
    edge_collection = [gri_mod._get_sub(BW.shape, p) for p in pixel_lists]
    connected_nodes = _connected_nodes(BW.shape, node_list, pixel_lists)
    edge_rows = []
    for nodes in connected_nodes:
        if len(nodes) >= 2:
            for a, b in itertools.combinations(nodes, 2):
                edge_rows.append((a, b))
    if len(edge_rows) == 0:
        return [], []
    edge_map = np.array(edge_rows, dtype=np.int64)
    max_node = int(edge_map.max())
    G = nx.MultiGraph()
    G.add_nodes_from(range(1, max_node + 1))
    G.add_edges_from([(int(a), int(b)) for a, b in edge_rows])
    comp = {}
    for ci, cc in enumerate(nx.connected_components(G)):
        for n in cc: comp[n] = ci
    node_subs_all = np.vstack([gri_mod._get_nn_px(gri_mod._get_sub(BW.shape, node_list[k]))
                               for k in range(len(node_list))])
    node_subs = node_subs_all[:max_node, :]
    deg_node = np.array([G.degree(i) for i in range(1, max_node + 1)], dtype=np.int64)
    node_edges = {i: [] for i in range(1, max_node + 1)}
    for e_idx, nodes in enumerate(connected_nodes):
        for n in nodes:
            if n <= max_node:
                node_edges[n].append(e_idx)
    def graph_reachable(s, t):
        return comp[s] == comp[t]
    grid_cand = np.where(deg_node >= 5)[0] + 1
    radial_list = []; px_count = []
    for cand in grid_cand:
        center_pt = node_subs[cand - 1]
        within = np.where(np.sum((node_subs - center_pt) ** 2, axis=1) <= radial_r ** 2)[0] + 1
        nbr_nodes = np.setdiff1d(within, np.array([cand]))
        r_list = []
        for nb in nbr_nodes:
            if graph_reachable(int(cand), int(nb)):
                r_list.append(np.sqrt(np.sum((node_subs[nb - 1] - center_pt) ** 2)))
        r_list = np.array(r_list, dtype=float)
        if r_list.size:
            m = r_list.mean()
            r_list = r_list[np.abs(r_list - m) <= 5]
        if r_list.size >= 5 and r_list.mean() >= 5:
            radial_list.append(cand); px_count.append(float(np.sum(r_list)))
    current_radial = []
    for i, node in enumerate(radial_list):
        current_radial.append({"subs": node_subs[node - 1].copy(),
                               "score": gri_mod._score_mean(cropped_score_F, node_list[node - 1]),
                               "px_count": px_count[i]})
    grid_cand = np.where((deg_node >= 3) & (deg_node <= 4))[0] + 1
    candidate_node = []
    for cand in grid_cand:
        node_to_grow = int(cand)
        center_pt = node_subs[node_to_grow - 1]
        within = np.where(np.sum((node_subs - center_pt) ** 2, axis=1) <= radial_r ** 2)[0] + 1
        nbr_nodes = np.setdiff1d(within, np.array([node_to_grow]))
        r_list = []; end_pts = []
        for nb in nbr_nodes:
            if graph_reachable(node_to_grow, int(nb)):
                r_list.append(np.sqrt(np.sum((node_subs[nb - 1] - center_pt) ** 2)))
                end_pts.append(node_subs[nb - 1])
        if len(end_pts) == 0:
            continue
        r_list = np.array(r_list, dtype=float); end_pts = np.vstack(end_pts)
        x = end_pts - center_pt
        angle_list = wrapTo360(atan2d(x[:, 0], x[:, 1]))
        sorted_idx = np.argsort(angle_list, kind="stable")
        a = angle_list[sorted_idx]
        temp = np.diff(np.append(a, a[0] + 360))
        keep = np.abs(temp) >= 15
        filtered_end_pts_idx = sorted_idx[keep]
        filtered_end_pts = end_pts[filtered_end_pts_idx, :]
        r_list = r_list[filtered_end_pts_idx]
        rad_flag = np.abs(r_list - r_list.mean()) <= 5 if r_list.size else np.array([], dtype=bool)
        r_list = r_list[rad_flag]
        if r_list.size and r_list.mean() >= 5:
            num_new_edge = 5 - r_list.size
        else:
            num_new_edge = 0
        if not (1 <= num_new_edge <= 2):
            continue
        x = filtered_end_pts - center_pt
        b = list(wrapTo360(atan2d(x[:, 0], x[:, 1])))
        c = []
        for _ in range(num_new_edge):
            a2 = np.sort(np.array(b))
            temp2 = np.diff(np.append(a2, a2[0] + 360))
            sel_a = a2[np.abs(temp2) >= 15]
            temp2f = temp2[np.abs(temp2) >= 15]
            max_angle_idx = int(np.argmax(temp2f))
            new_c = sel_a[max_angle_idx] + temp2f[max_angle_idx] / 2.0
            c.append(new_c); b.append(new_c)
        new_angle = np.array(c, dtype=float)
        r_new = r_list.mean()
        new_end_pt = matlab_round(center_pt + r_new * np.column_stack((sind(new_angle), cosd(new_angle))))
        valid_end_pt = True
        for i in range(num_new_edge):
            sector_mask = gri_mod._get_sector_mask(center_pt, new_end_pt[i, :], theta_r, N_r, N_c)
            r_squared = np.sum((center_pt - new_end_pt[i, :]) ** 2)
            sector_mask_ind = find_linear(sector_mask)
            common_ind = []
            for e_idx in node_edges[node_to_grow]:
                edge = edge_collection[e_idx]
                edge_ind = sub2ind((N_r, N_c), edge[:, 0], edge[:, 1])
                common_ind.append(np.intersect1d(sector_mask_ind, edge_ind))
            n_common = np.unique(np.concatenate(common_ind)).size if common_ind else 0
            if n_common >= np.sqrt(r_squared) * factor_sector_mask:
                valid_end_pt = False
                break
        if valid_end_pt:
            candidate_node.append({"subs": node_subs[node_to_grow - 1].copy(),
                                   "score": gri_mod._score_mean(cropped_score_F, node_list[node_to_grow - 1]),
                                   "end_pts": new_end_pt.astype(np.int64)})
    return current_radial, candidate_node

def apply():
    import make_radial_growth as mrg
    mrg.get_radial_info = get_radial_info_fast
    mc.label2idx = label2idx_fast
    mc.bwconncomp = bwconncomp_fast
    import sys
    for name, mod in list(sys.modules.items()):
        if mod is None or name == __name__: continue
        if getattr(mod, 'label2idx', None) is not None and getattr(mod,'label2idx') is not label2idx_fast and hasattr(mod,'__file__') and mod.__file__ and 'rgm/py' in mod.__file__:
            mod.label2idx = label2idx_fast
        if getattr(mod, 'bwconncomp', None) is not None and hasattr(mod,'__file__') and mod.__file__ and 'rgm/py' in mod.__file__:
            mod.bwconncomp = bwconncomp_fast

# ---------------------------------------------------------------- upgrades
def get_upgraded_points_fast(A1, A2, road_density_map, max_radius, upgrade_length,
                             min_upgrade_per_intersect):
    import get_upgraded_points as gup
    A1 = np.asarray(A1); A2 = np.asarray(A2)
    N_r, N_c = A1.shape
    x2_lists = bwconncomp_fast(A2, 8)
    x2 = [gup._get_subs(p, N_r, N_c) for p in x2_lists]
    # A2 pixels within Euclidean sqrt(2) of any A1 pixel == 8-neighbourhood dilation
    near = ndimage.binary_dilation(A1 != 0, structure=np.ones((3, 3), bool))
    cand = near & (A2 != 0)
    if not cand.any():
        return np.empty((0, 2), dtype=np.int64), A2.copy()
    comp_lab = np.zeros((N_r, N_c), dtype=np.int64)
    for k, p in enumerate(x2_lists):
        r, c = ind2sub((N_r, N_c), p); comp_lab[r - 1, c - 1] = k + 1
    lin_unique = find_linear(cand)               # sorted ascending, column-major 1-based
    u, v = ind2sub((N_r, N_c), lin_unique)
    intersect_pts = np.column_stack((u, v)).astype(np.int64)
    segment_id = comp_lab[u - 1, v - 1]
    density_vals = road_density_map[intersect_pts[:, 0] - 1, intersect_pts[:, 1] - 1].astype(float)
    upgrade_radius_per_point = max_radius * density_vals / np.max(density_vals)
    A2_upgrade = A2.astype(float).copy()
    upgrade_left = upgrade_length
    to_be_upgraded = []
    n_marked = 0
    rand_order = mc.randperm0(len(segment_id))
    for idx in rand_order:
        if upgrade_left <= 0:
            break
        comp = x2[segment_id[idx] - 1]
        d2 = np.sum((intersect_pts[idx, :].astype(float) - comp.astype(float)) ** 2, axis=1)
        mask = d2 <= upgrade_radius_per_point[idx] ** 2
        pts = comp[mask, :]
        rows = pts[:, 0] - 1; cols = pts[:, 1] - 1
        new_count = int(np.sum(A2_upgrade[rows, cols] != -1))
        if new_count >= min_upgrade_per_intersect:
            to_be_upgraded.append(pts)
            A2_upgrade[rows, cols] = -1
            n_marked += new_count
            upgrade_left = upgrade_length - n_marked
    if len(to_be_upgraded) == 0:
        return np.empty((0, 2), dtype=np.int64), A2_upgrade
    return np.vstack(to_be_upgraded).astype(np.int64), A2_upgrade

def apply_upgrades():
    import make_road, make_road_till_upgrade
    make_road.get_upgraded_points = get_upgraded_points_fast
    make_road_till_upgrade.get_upgraded_points = get_upgraded_points_fast

# ---------------------------------------------------------------- linear growth
from scipy.spatial import cKDTree

def make_linear_growth_fast(road_data, uc_map, score, new_total_length, road_spacing,
                            search_radius_for_urban, min_number_road_pixel, sel_mode,
                            N_iter, old_cand_map, old_path_list, old_path_length, debug):
    from matlab_compat import conv2_same, find
    import make_linear_growth as mlg
    get_next_best_cand = mlg.get_next_best_cand
    road_data = np.asarray(road_data, dtype=float)
    score = np.array(score, dtype=float, copy=True)
    N_r, N_c = road_data.shape
    road_data_sum = float(np.sum(road_data))
    if old_cand_map is None or (isinstance(old_cand_map, np.ndarray) and old_cand_map.size == 0):
        print("Generating candidates")
        cand_A = 1 * (conv2_same(road_data, np.ones((3, 3))) >= 1)
        cand_A = cand_A - road_data
        a, b, v = find(score * cand_A)
        cand_map = np.zeros((len(a), 4))
        cand_map[:, 0] = a; cand_map[:, 1] = b; cand_map[:, 2] = v
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
        path_list = []; path_length = []; path_num = 0
    else:
        cand_map_filtered = np.asarray(old_cand_map, dtype=float).copy()
        path_list = old_path_list; path_length = old_path_length
        path_num = len(path_list)
    cand_map = cand_map_filtered
    sparse_cand_map = np.zeros((N_r, N_c))
    np.add.at(sparse_cand_map, (cand_map_filtered[:, 0].astype(int) - 1,
              cand_map_filtered[:, 1].astype(int) - 1), 1.0 * (cand_map_filtered[:, 2] != 0))
    if debug == 0:
        N_iter = int(new_total_length)
    # rows with path id 0 never move; index their (static) positions
    free_rows = np.where(cand_map[:, 3] == 0)[0]
    tree = cKDTree(cand_map[free_rows, 0:2]) if free_rows.size else None
    sum_pl = float(sum(path_length))
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
            path_list.append([rc.copy()]); path_length.append(1)
            cand_map[id_max, 3] = path_num
        else:
            pid = int(cand_map[id_max, 3])
            path_list[pid - 1].append(rc.copy()); path_length[pid - 1] += 1
        sum_pl += 1
        score[int(rc[0]) - 1, int(rc[1]) - 1] = 0
        if tree is not None:
            near = free_rows[tree.query_ball_point(rc, road_spacing + 1e-6)]
            if near.size:
                d = np.sqrt(np.sum((rc - cand_map[near, 0:2]) ** 2, axis=1))
                sel = near[(d <= road_spacing) & (cand_map[near, 3] == 0)]
                cand_map[sel, 2] = 0
        cx, cy, val = get_next_best_cand(row, uc_map, score, sparse_cand_map,
                                         search_radius_for_urban, sel_mode)
        cand_map[id_max, 0:3] = [cx, cy, val]
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
                    score[int(rc[0]) - 1, int(rc[1]) - 1] = 0
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
    new_rd = np.zeros((N_r, N_c))
    for p in path_list:
        if p:
            pa = np.vstack(p).astype(int)
            new_rd[pa[:, 0] - 1, pa[:, 1] - 1] = 1
    added_road = new_rd
    updated_road = (new_rd + road_data != 0) * 1.0
    num_new_road = float(np.sum(updated_road) - np.sum(road_data))
    return (added_road, updated_road, num_new_road, path_list, path_length, cand_map, score)


def get_grown_roads_fast(remove_iteration, min_road_sep_1, min_road_sep_2, start_road,
                         all_high_roads, uc_map, score_map, road_demand, road_spacing,
                         search_radius_for_urban, min_number_road_pixel, sel_mode,
                         N_iter, debug):
    from matlab_compat import find_linear, ind2sub, pdist2_smallest
    import get_grown_roads as ggr
    _nanmean = ggr._nanmean
    A1 = np.asarray(start_road, dtype=float)
    score_2 = np.asarray(score_map, dtype=float)
    N_r, N_c = A1.shape
    all_high_roads = np.asarray(all_high_roads)
    road_grown = 0.0; old_cand_map = None; path_list = []; old_path_length = []
    processed_road_2 = A1.copy(); new_rd = np.zeros((N_r, N_c)); added_road = np.zeros((N_r, N_c))
    for remove_iter in range(1, remove_iteration + 1):
        (added_road, updated_road, num_new_road, path_list, old_path_length,
         old_cand_map, _) = make_linear_growth_fast(
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
        new_rd = np.zeros((N_r, N_c))
        for p in P2:
            if p:
                pa = np.vstack(p).astype(int); new_rd[pa[:, 0] - 1, pa[:, 1] - 1] = 1
        processed_road_2 = A1 + new_rd
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
        road_grown = float(np.sum(processed_road_2) - np.sum(A1))
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

def apply_growth():
    import make_road, make_road_till_upgrade
    make_road.get_grown_roads = get_grown_roads_fast
    make_road_till_upgrade.get_grown_roads = get_grown_roads_fast
