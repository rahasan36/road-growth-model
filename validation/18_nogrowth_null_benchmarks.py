import sys
import json
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import disk


def map_gprtp(v, n):
    if n == 5:
        return {1: 1, 2: 1, 3: 2, 4: 3, 5: 4}.get(v, 0)
    return v


def rasterize_grip(path, T, crs, shape):
    g = gpd.read_file(path)
    if g.crs != crs:
        g = g.to_crs(crs)
    n = len(g['GP_RTP'].unique())
    shapes = [(x, 1) for x, v in zip(g.geometry, g['GP_RTP']) if x is not None and not x.is_empty and map_gprtp(int(v), n) > 0]
    return rasterize(shapes, out_shape=shape, transform=T, fill=0, dtype=np.uint8) > 0


def boundary(path, T, crs, shape):
    g = gpd.read_file(path).to_crs(crs)
    return rasterize([(x, 1) for x in g.geometry], out_shape=shape, transform=T, fill=0, dtype=np.uint8) > 0


def metrics(pred, obs, buf, px, tot):
    npred, nobs = int(pred.sum()), int(obs.sum())
    if buf == 0:
        tp = int((pred & obs).sum())
        fn = nobs - tp
    else:
        s = disk(max(1, round(buf / px)))
        tp = int((pred & binary_dilation(obs, structure=s)).sum())
        fn = nobs - int((obs & binary_dilation(pred, structure=s)).sum())
    fp = npred - tp
    tn = tot - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    oa = (tp + tn) / tot
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / tot ** 2
    k = (oa - pe) / (1 - pe) if pe < 1 else 0.0
    d = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = (float(tp) * tn - float(fp) * fn) / d if d else 0.0
    return dict(N_pred=npred, N_obs=nobs, TP=tp, FP=fp, FN=fn, Precision=round(p, 4), Recall=round(r, 4), F1=round(f1, 4),
                IoU=round(tp / (tp + fp + fn), 4) if tp + fp + fn else 0.0, Kappa=round(k, 4), MCC=round(mcc, 4))


def aggregate(a, f):
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:h, :w].reshape(h // f, f, w // f, f).any(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--country', required=True)
    ap.add_argument('--forecast', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--reference_kind', choices=['osm_only', 'osm_union_grip'], required=True)
    ap.add_argument('--grip', required=True)
    ap.add_argument('--boundary', required=True)
    ap.add_argument('--aggregate', type=int, default=1)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--shift_km', default='10,30')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no_nulls', action='store_true')
    a = ap.parse_args()
    with rasterio.open(a.forecast) as s:
        fc = s.read(1)
        T, crs, shape, px = s.transform, s.crs, s.shape, abs(s.transform.a)
    with rasterio.open(a.reference) as s:
        ref = s.read(1) > 0
    grip = rasterize_grip(a.grip, T, crs, shape)
    land = boundary(a.boundary, T, crs, shape)
    fc_all = fc > 0
    fc_radial = (fc >= 11) & (fc <= 14)
    f = a.aggregate
    if f > 1:
        fc_all, fc_radial, ref, grip, land = [aggregate(x, f) for x in (fc_all, fc_radial, ref, grip, land)]
        px *= f
    tot_grid = int(land.size)
    tot = int(land.sum())
    osm_only_part = ref & ~grip
    fc_new = fc_radial & ~grip
    obs_new = ref & ~grip
    obs_new_far = ref & ~binary_dilation(grip, structure=disk(1))
    summary = dict(country=a.country, reference_kind=a.reference_kind, cell_m=px, land_cells=tot, forecast_cells=int(fc_all.sum()),
                   grip_cells=int(grip.sum()), reference_cells=int(ref.sum()), radial_coded_cells=int(fc_radial.sum()),
                   radial_coded_on_grip_cells=int((fc_radial & grip).sum()), forecast_new_cells=int(fc_new.sum()),
                   observed_new_cells=int(obs_new.sum()), observed_new_beyond_1cell=int(obs_new_far.sum()))
    rows = []
    for b in (0, 300, 600, 900):
        if f > 1 and b and b < px:
            continue
        for name, p in (('Forecast 2030 SSP1', fc_all), ('No growth: GRIP 2015', grip)):
            rows.append(dict(Assessment='All roads', Model=name, Buffer_m=b, **metrics(p, ref, b, px, tot_grid)))
    if a.no_nulls:
        pd.DataFrame(rows).to_csv(f'{a.out}/{a.country}_benchmarks_{a.reference_kind}_allroads.csv', index=False)
        json.dump(summary, open(f'{a.out}/{a.country}_benchmarks_{a.reference_kind}_summary.json', 'w'), indent=1)
        return
    rng = np.random.default_rng(42)
    cand = np.flatnonzero(land & ~grip)
    dist = distance_transform_edt(~grip).astype(np.float32) * px
    cand_d = dist.ravel()[cand]
    bins = np.array([0, 300, 600, 900, 1500, 3000, 6000, 1e9])
    h, _ = np.histogram(dist[fc_new], bins)
    n = int(fc_new.sum())
    lo, hi = [int(round(float(x) * 1000 / px)) for x in a.shift_km.split(',')]

    def random_land():
        m = np.zeros(fc.size if f == 1 else land.size, bool)
        m[rng.choice(cand, n, replace=False)] = True
        return m.reshape(land.shape)

    def distance_matched():
        sel = []
        for i in range(len(bins) - 1):
            pool = cand[(cand_d >= bins[i]) & (cand_d < bins[i + 1])]
            if h[i] and len(pool):
                sel.append(rng.choice(pool, min(h[i], len(pool)), replace=False))
        m = np.zeros(land.size, bool)
        m[np.concatenate(sel)] = True
        return m.reshape(land.shape)

    def shifted():
        while True:
            dy, dx = rng.integers(-hi, hi + 1, size=2)
            if lo <= np.hypot(dy, dx) <= hi:
                break
        return np.roll(np.roll(fc_new, dy, axis=0), dx, axis=1) & land & ~grip

    nulls = {'Null: random cells': [random_land() for _ in range(a.reps)],
             'Null: random cells, distance-matched': [distance_matched() for _ in range(a.reps)],
             'Null: forecast pattern shifted 10-30 km': [shifted() for _ in range(a.reps)]}
    for b in (0, 300, 600, 900):
        if f > 1 and b and b < px:
            continue
        for rn, r in (('OSM 2026 not in GRIP 2015', obs_new), ('OSM 2026 beyond 1 cell of GRIP 2015', obs_new_far)):
            rows.append(dict(Assessment=f'New roads vs {rn}', Model='Forecast new roads', Buffer_m=b, **metrics(fc_new, r, b, px, tot_grid)))
            rows.append(dict(Assessment=f'New roads vs {rn}', Model='Forecast radial-coded cells incl. those on GRIP roads', Buffer_m=b, **metrics(fc_radial, r, b, px, tot_grid)))
            for nn, reps in nulls.items():
                ms = pd.DataFrame([metrics(x, r, b, px, tot_grid) for x in reps])
                rows.append(dict(Assessment=f'New roads vs {rn}', Model=nn + f' (mean of {a.reps})', Buffer_m=b,
                                 **{k: round(float(ms[k].mean()), 4) for k in ms.columns}))
                rows.append(dict(Assessment=f'New roads vs {rn}', Model=nn + ' (sd)', Buffer_m=b,
                                 **{k: round(float(ms[k].std(ddof=1)), 4) for k in ('Precision', 'Recall', 'F1')}))
    pd.DataFrame(rows).to_csv(f'{a.out}/{a.country}_benchmarks_{a.reference_kind}.csv', index=False)
    json.dump(summary, open(f'{a.out}/{a.country}_benchmarks_{a.reference_kind}_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
