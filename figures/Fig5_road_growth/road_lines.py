import numpy as np
import shapely
from pyproj import Transformer
from skimage.morphology import skeletonize

SIMPLIFY_CELLS = 0.6
CHAIKIN_ITER = 2
SNAP_CELLS = 1.5
ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def skeleton_pixels(mask, tile=4096, pad=48):
    H, W = mask.shape
    R, C = [], []
    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            a0, b0 = max(r0 - pad, 0), max(c0 - pad, 0)
            a1, b1 = min(r0 + tile + pad, H), min(c0 + tile + pad, W)
            sub = mask[a0:a1, b0:b1]
            if not sub.any():
                continue
            rr, cc = np.nonzero(skeletonize(sub))
            rr, cc = rr + a0, cc + b0
            k = (rr >= r0) & (rr < min(r0 + tile, H)) & (cc >= c0) & (cc < min(c0 + tile, W))
            R.append(rr[k])
            C.append(cc[k])
    if not R:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(R).astype(np.int64), np.concatenate(C).astype(np.int64)


def pixel_graph(rr, cc, W):
    key = rr * W + cc
    o = np.argsort(key)
    key, rr, cc = key[o], rr[o], cc[o]
    n = len(key)

    def find(r, c):
        q = r * W + c
        p = np.minimum(np.searchsorted(key, q), n - 1)
        return np.where((key[p] == q) & (c >= 0) & (c < W) & (r >= 0), p, -1)

    I, J = [], []
    for dr, dc in ORTH:
        j = find(rr + dr, cc + dc)
        m = j >= 0
        I.append(np.flatnonzero(m))
        J.append(j[m])
    for dr, dc in DIAG:
        j = find(rr + dr, cc + dc)
        m = (j >= 0) & (find(rr + dr, cc) < 0) & (find(rr, cc + dc) < 0)
        I.append(np.flatnonzero(m))
        J.append(j[m])
    I, J = np.concatenate(I), np.concatenate(J)
    o = np.argsort(I, kind="stable")
    I, J = I[o], J[o]
    start = np.searchsorted(I, np.arange(n + 1))
    return rr, cc, J, start


def trace_paths(rr, cc, W):
    rr, cc, J, start = pixel_graph(rr, cc, W)
    n = len(rr)
    deg = np.diff(start)

    def nb(i):
        return J[start[i]:start[i + 1]]

    seen, paths = set(), []
    for v in np.flatnonzero(deg != 2):
        for w in nb(v):
            e = (min(v, w), max(v, w))
            if e in seen:
                continue
            seen.add(e)
            path, prev, cur = [v, w], v, w
            while deg[cur] == 2:
                a, b = nb(cur)
                nxt = b if a == prev else a
                e = (min(cur, nxt), max(cur, nxt))
                if e in seen:
                    break
                seen.add(e)
                path.append(nxt)
                prev, cur = cur, nxt
            paths.append(np.asarray(path))
    done = np.zeros(n, bool)
    for p in paths:
        done[p] = True
    for v in np.flatnonzero((deg == 2) & ~done):
        if done[v]:
            continue
        path, prev, cur = [v], v, nb(v)[0]
        done[v] = True
        while cur != v and not done[cur]:
            path.append(cur)
            done[cur] = True
            a, b = nb(cur)
            nxt = b if a == prev else a
            prev, cur = cur, nxt
        path.append(v)
        paths.append(np.asarray(path))
    return paths, deg, rr, cc


def make_lines(xys):
    if not xys:
        return np.array([], object)
    n = np.array([len(x) for x in xys])
    return shapely.linestrings(np.concatenate(xys), indices=np.repeat(np.arange(len(xys)), n))


def chaikin(xy, iters):
    for _ in range(iters):
        if len(xy) < 3:
            return xy
        p, q = xy[:-1], xy[1:]
        mid = np.empty((2 * len(p), 2))
        mid[0::2] = 0.75 * p + 0.25 * q
        mid[1::2] = 0.25 * p + 0.75 * q
        xy = np.vstack([xy[:1], mid[1:-1], xy[-1:]]) if len(mid) > 2 else np.vstack([xy[:1], xy[-1:]])
    return xy


def smooth(xy, tol, iters=CHAIKIN_ITER):
    if len(xy) > 2:
        xy = shapely.get_coordinates(shapely.simplify(shapely.linestrings(xy), tol, preserve_topology=False))
    return chaikin(xy, iters)


def new_road_lines(a, T, src_crs, dst, existing=None, snap=True):
    cell = abs(T.a)
    rr, cc = skeleton_pixels(a > 0)
    paths, deg, rr, cc = trace_paths(rr, cc, a.shape[1])
    X, Y = Transformer.from_crs(src_crs, dst, always_xy=True).transform(T.c + (cc + 0.5) * T.a, T.f + (rr + 0.5) * T.e)
    X, Y = np.asarray(X), np.asarray(Y)
    cls = a[rr, cc]
    geoms, ks, ends = [], [], []
    for p in paths:
        geoms.append(smooth(np.column_stack([X[p], Y[p]]), SIMPLIFY_CELLS * cell))
        ks.append(np.bincount(cls[p], minlength=5)[1:].argmax() + 1)
        ends.append((deg[p[0]] == 1, deg[p[-1]] == 1))
    iso = np.flatnonzero(deg == 0)
    nsnap, nlink = 0, 0
    if snap and existing is not None and len(existing):
        tree = shapely.STRtree(existing)
        for i, (e0, e1) in enumerate(ends):
            xy = geoms[i]
            for at_start, flag in ((True, e0), (False, e1)):
                if not flag:
                    continue
                pt = shapely.points(xy[0] if at_start else xy[-1])
                j = tree.query_nearest(pt, max_distance=SNAP_CELLS * cell)
                if len(j) == 0:
                    continue
                q = shapely.get_coordinates(shapely.shortest_line(pt, existing[j[0]]))[-1]
                xy = np.vstack([q, xy]) if at_start else np.vstack([xy, q])
                nsnap += 1
            geoms[i] = xy
        for v in iso:
            pt = shapely.points(X[v], Y[v])
            j = tree.query(pt, predicate="dwithin", distance=SNAP_CELLS * cell)
            if len(j) == 0:
                continue
            j = j[np.argsort(shapely.distance(pt, existing[j]))]
            q1 = shapely.get_coordinates(shapely.shortest_line(pt, existing[j[0]]))[-1]
            if len(j) > 1:
                q2 = shapely.get_coordinates(shapely.shortest_line(pt, existing[j[1]]))[-1]
                xy = np.vstack([q1, [X[v], Y[v]], q2])
            else:
                xy = np.vstack([[X[v], Y[v]], q1])
            if np.hypot(*(xy[-1] - xy[0])) < 1.0:
                continue
            geoms.append(xy)
            ks.append(int(cls[v]))
            nlink += 1
    stats = dict(skeleton_px=len(rr), paths=len(paths), isolated_px=len(iso), isolated_linked=nlink, snapped_ends=nsnap)
    return make_lines(geoms), np.asarray(ks, int), stats


def upgraded_on_lines(lines, k_old, a, T, src_crs, dst):
    cell = abs(T.a)
    xy, idx = shapely.get_coordinates(shapely.segmentize(lines, cell / 3.0), return_index=True)
    same = idx[1:] == idx[:-1]
    mid = 0.5 * (xy[1:] + xy[:-1])
    mx, my = Transformer.from_crs(dst, src_crs, always_xy=True).transform(mid[:, 0], mid[:, 1])
    col = np.floor((np.asarray(mx) - T.c) / T.a).astype(np.int64)
    row = np.floor((np.asarray(my) - T.f) / T.e).astype(np.int64)
    ok = same & (row >= 0) & (row < a.shape[0]) & (col >= 0) & (col < a.shape[1])
    v = np.zeros(len(mid), np.uint8)
    v[ok] = a[row[ok], col[ok]]
    flag = v > 0
    cont = np.zeros(len(v), bool)
    cont[1:] = flag[1:] & flag[:-1] & same[:-1] & (v[1:] == v[:-1])
    starts, stops = np.flatnonzero(flag & ~cont), np.flatnonzero(~cont)
    geoms, kn, ko = [], [], []
    for s in starts:
        p = np.searchsorted(stops, s, side="right")
        e = stops[p] if p < len(stops) else len(v)
        geoms.append(xy[s:e + 1])
        kn.append(int(v[s]))
        ko.append(int(k_old[idx[s]]))
    hit = np.unique(row[flag] * a.shape[1] + col[flag])
    stats = dict(runs=len(geoms), upgraded_cells_on_lines=round(len(hit) / max(int((a > 0).sum()), 1), 3))
    return make_lines(geoms), np.asarray(kn, int), np.asarray(ko, int), stats
