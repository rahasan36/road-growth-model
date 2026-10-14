import numpy as np
import shapely
import geopandas as gpd

OFFS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def cell_graph(code):
    H, W = code.shape
    rr, cc = np.nonzero(code > 0)
    lin = rr.astype(np.int64) * W + cc

    def find(r, c):
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        key = r.astype(np.int64) * W + c
        pos = np.searchsorted(lin, key)
        pos[pos >= len(lin)] = 0
        hit = ok & (lin[pos] == key)
        return hit, pos

    us, vs = [], []
    for dr, dc in OFFS:
        ok, pos = find(rr + dr, cc + dc)
        if dr == 1 and dc != 0:
            ok &= ~find(rr, cc + dc)[0] & ~find(rr + dr, cc)[0]
        us.append(np.flatnonzero(ok))
        vs.append(pos[ok])
    return rr, cc, np.concatenate(us), np.concatenate(vs)


def trace(n, u, v, breaks_extra=None):
    m = len(u)
    eu = np.concatenate([u, v])
    ev = np.concatenate([v, u])
    eid = np.concatenate([np.arange(m), np.arange(m)])
    order = np.argsort(eu, kind='stable')
    nbr, eids = ev[order], eid[order]
    start = np.searchsorted(eu[order], np.arange(n + 1))
    deg = np.diff(start)
    brk = deg != 2
    if breaks_extra is not None:
        brk |= breaks_extra
    used = np.zeros(m, bool)
    paths = []

    def walk(a, k):
        path = [a]
        prev, cur, e = a, nbr[k], eids[k]
        used[e] = True
        path.append(cur)
        while not brk[cur]:
            s = start[cur]
            k2 = s + 1 if eids[s] == e else s
            e = eids[k2]
            if used[e]:
                break
            used[e] = True
            prev, cur = cur, nbr[k2]
            path.append(cur)
        return path

    for a in np.flatnonzero(brk & (deg > 0)):
        for k in range(start[a], start[a + 1]):
            if not used[eids[k]]:
                paths.append(walk(a, k))
    for e in np.flatnonzero(~used):
        if used[e]:
            continue
        a = u[e]
        k = start[a] + int(np.flatnonzero(eids[start[a]:start[a + 1]] == e)[0])
        p = walk(a, k)
        paths.append(p)
    return paths, deg


def vectorize(code, transform, crs, split_classes=False):
    rr, cc, u, v = cell_graph(code)
    n = len(rr)
    xs = transform.c + (cc + 0.5) * transform.a
    ys = transform.f + (rr + 0.5) * transform.e
    val = code[rr, cc].astype(np.int64)
    if split_classes:
        diff = val[u] != val[v]
        mid_ids = n + np.arange(int(diff.sum()))
        mx = (xs[u[diff]] + xs[v[diff]]) / 2
        my = (ys[u[diff]] + ys[v[diff]]) / 2
        u2 = np.concatenate([u[~diff], u[diff], mid_ids])
        v2 = np.concatenate([v[~diff], mid_ids, v[diff]])
        xs = np.concatenate([xs, mx])
        ys = np.concatenate([ys, my])
        val = np.concatenate([val, np.zeros(len(mid_ids), np.int64)])
        extra = np.zeros(len(xs), bool)
        extra[n:] = True
        paths, _ = trace(len(xs), u2, v2, extra)
    else:
        paths, _ = trace(n, u, v)
    lens = np.array([len(p) for p in paths])
    flat = np.concatenate(paths)
    coords = np.column_stack([xs[flat], ys[flat]])
    offs = np.concatenate([[0], np.cumsum(lens)])
    lines = shapely.linestrings(coords, indices=np.repeat(np.arange(len(paths)), lens))
    gc = np.zeros(len(paths), np.int64)
    if split_classes:
        a, b = val[flat[offs[:-1]]], val[flat[offs[:-1] + 1]]
        gc = np.where(a > 0, a, b)
    return gpd.GeoDataFrame({'grid_code': gc}, geometry=lines, crs=crs)
