"""get_radial_info.py  -- port of get_radial_info.m

Identifies where to perform radial growth.  Builds a graph over the road
skeleton (branch/end nodes connected by edges), then classifies high-degree
nodes as existing radial centers and medium-degree nodes as candidates for new
radial spokes.

The MATLAB ``current_radial`` and ``candidate_node`` 3xK cell arrays are
returned here as lists of dicts:

    current_radial[i]  = {'subs': array([r, c]), 'score': float, 'px_count': float}
    candidate_node[i]  = {'subs': array([r, c]), 'score': float,
                          'end_pts': array (num_new_edge, 2)}

All coordinates are 1-based (matching MATLAB).
"""

import itertools
import numpy as np
import networkx as nx

from matlab_compat import (ind2sub, sub2ind, bwlabeln, label2idx, bwconncomp,
                           pdist2_smallest, find_linear, poly2mask,
                           wrapTo360, atan2d, sind, cosd, matlab_round)
from make_skel import make_skel


def _get_sub(shape, linidx):
    r, c = ind2sub(shape, np.asarray(linidx))
    return np.column_stack((r, c)).astype(np.int64)


def _get_nn_px(X):
    if X.shape[0] == 1:
        return X[0].astype(float)
    x_mean = X.mean(axis=0)
    d = np.sum((x_mean - X) ** 2, axis=1)
    return X[int(np.argmin(d))].astype(float)


def _score_mean(cropped_score_F, lin):
    return float(np.mean(cropped_score_F[np.asarray(lin) - 1]))


def _get_sector_mask(x_0, x_f, theta_r, N_r, N_c):
    x_0 = np.asarray(x_0, dtype=float)
    x_f = np.asarray(x_f, dtype=float)
    d_0 = x_f - x_0
    r_0 = np.sqrt(np.sum(d_0 ** 2))
    theta_0 = wrapTo360(atan2d(d_0[0], d_0[1]))
    new_angle = theta_0 + np.array([theta_r, -theta_r]) / 2.0
    x_vert = matlab_round(x_0 + r_0 * np.column_stack((sind(new_angle), cosd(new_angle))))
    poly_vertices = np.vstack((x_0, x_vert[0], x_f, x_vert[1], x_0))
    return poly2mask(poly_vertices[:, 1], poly_vertices[:, 0], N_r, N_c)


def get_radial_info(cropped_A, cropped_score, radial_r, theta_r,
                    factor_sector_mask, show_graph=0):
    cropped_A = np.asarray(cropped_A)
    cropped_score = np.asarray(cropped_score, dtype=float)
    N_r, N_c = cropped_A.shape
    cropped_score_F = cropped_score.flatten(order="F")  # column-major linear access

    br_mask, ep_mask = make_skel(cropped_A)
    disjoint_img = cropped_A - br_mask - ep_mask
    BW = br_mask + ep_mask

    L, _ = bwlabeln(BW, 8)
    node_list = label2idx(L)  # node_list[k-1] = 1-based col-major linidx of node k
    pixel_lists = bwconncomp(disjoint_img, 8)
    edge_collection = [_get_sub(BW.shape, p) for p in pixel_lists]

    connected_nodes = [[] for _ in range(len(edge_collection))]
    for node_idx in range(1, len(node_list) + 1):
        rc = _get_sub(BW.shape, node_list[node_idx - 1]).astype(float)
        for e_idx, edge in enumerate(edge_collection):
            d = pdist2_smallest(edge.astype(float), rc, "squaredeuclidean", 1)
            if d.size and np.min(d) <= 2:
                connected_nodes[e_idx].append(node_idx)

    # build edge_map (pairs of node indices) from each edge's connected nodes
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

    # node representative coordinates, truncated to nodes referenced by edges
    node_subs_all = np.vstack([_get_nn_px(_get_sub(BW.shape, node_list[k]))
                               for k in range(len(node_list))])
    node_subs = node_subs_all[:max_node, :]  # 1-based indexing: node n -> row n-1

    deg_node = np.array([G.degree(i) for i in range(1, max_node + 1)], dtype=np.int64)

    # node -> incident edges map (mirrors recomputed connected_edge_idx)
    node_edges = {i: [] for i in range(1, max_node + 1)}
    for e_idx, nodes in enumerate(connected_nodes):
        for n in nodes:
            if n <= max_node:
                node_edges[n].append(e_idx)

    def graph_reachable(s, t):
        try:
            nx.shortest_path_length(G, s, t)
            return True
        except nx.NetworkXNoPath:
            return False

    # ---- existing radial centers (degree >= 5) ------------------------------
    grid_cand = np.where(deg_node >= 5)[0] + 1  # 1-based node numbers, ascending
    radial_list = []
    px_count = []
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
            radial_list.append(cand)
            px_count.append(float(np.sum(r_list)))

    current_radial = []
    for i, node in enumerate(radial_list):
        current_radial.append({
            "subs": node_subs[node - 1].copy(),
            "score": _score_mean(cropped_score_F, node_list[node - 1]),
            "px_count": px_count[i],
        })

    # ---- new radial candidates (degree 3..4) --------------------------------
    grid_cand = np.where((deg_node >= 3) & (deg_node <= 4))[0] + 1
    candidate_node = []
    for cand in grid_cand:
        node_to_grow = int(cand)
        center_pt = node_subs[node_to_grow - 1]
        within = np.where(np.sum((node_subs - center_pt) ** 2, axis=1) <= radial_r ** 2)[0] + 1
        nbr_nodes = np.setdiff1d(within, np.array([node_to_grow]))

        r_list = []
        end_pts = []
        for nb in nbr_nodes:
            if graph_reachable(node_to_grow, int(nb)):
                r_list.append(np.sqrt(np.sum((node_subs[nb - 1] - center_pt) ** 2)))
                end_pts.append(node_subs[nb - 1])  # P(end) == target node
        if len(end_pts) == 0:
            continue
        r_list = np.array(r_list, dtype=float)
        end_pts = np.vstack(end_pts)

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
            c.append(new_c)
            b.append(new_c)

        new_angle = np.array(c, dtype=float)
        r_new = r_list.mean()
        new_end_pt = matlab_round(
            center_pt + r_new * np.column_stack((sind(new_angle), cosd(new_angle))))

        valid_end_pt = True
        for i in range(num_new_edge):
            sector_mask = _get_sector_mask(center_pt, new_end_pt[i, :], theta_r, N_r, N_c)
            r_squared = np.sum((center_pt - new_end_pt[i, :]) ** 2)
            sector_mask_ind = find_linear(sector_mask)
            common_ind = []
            for e_idx in node_edges[node_to_grow]:
                edge = edge_collection[e_idx]
                edge_ind = sub2ind((N_r, N_c), edge[:, 0], edge[:, 1])
                common_ind.append(np.intersect1d(sector_mask_ind, edge_ind))
            if common_ind:
                n_common = np.unique(np.concatenate(common_ind)).size
            else:
                n_common = 0
            if n_common >= np.sqrt(r_squared) * factor_sector_mask:
                valid_end_pt = False
                break

        if valid_end_pt:
            candidate_node.append({
                "subs": node_subs[node_to_grow - 1].copy(),
                "score": _score_mean(cropped_score_F, node_list[node_to_grow - 1]),
                "end_pts": new_end_pt.astype(np.int64),
            })

    return current_radial, candidate_node
