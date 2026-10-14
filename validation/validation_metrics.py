import math
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
from scipy.ndimage import binary_dilation, maximum_filter
from scipy.spatial import cKDTree
from skimage.morphology import disk, skeletonize

OSM_TYPE_MAP = {
    'motorway': 1, 'motorway_link': 1, 'trunk': 1, 'trunk_link': 1, 'primary': 1, 'primary_link': 1,
    'secondary': 2, 'secondary_link': 2, 'tertiary': 3, 'tertiary_link': 3,
    'residential': 4, 'unclassified': 4, 'living_street': 4, 'service': 4,
}
BUFFERS = [0, 300, 600, 900]


def read_grid(path):
    with rasterio.open(path) as s:
        return dict(T=s.transform, crs=s.crs, shape=s.shape, px=abs(s.transform.a), profile=s.profile.copy())


def read(path):
    with rasterio.open(path) as s:
        return s.read(1)


def rasterize_geofabrik(paths, g):
    if isinstance(paths, str):
        paths = [paths]
    gdf = pd.concat([gpd.read_file(p) for p in paths], ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=gpd.read_file(paths[0], rows=1).crs)
    gdf = gdf[gdf['fclass'].isin(OSM_TYPE_MAP)].copy()
    gdf['rt'] = gdf['fclass'].map(OSM_TYPE_MAP)
    gdf = gdf.to_crs(g['crs'])
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    b = rasterize(((x, 1) for x in gdf.geometry), out_shape=g['shape'], transform=g['T'], fill=0, dtype='uint8') > 0
    gdf = gdf.sort_values('rt', ascending=False)
    t = rasterize(((x, int(v)) for x, v in zip(gdf.geometry, gdf['rt'])), out_shape=g['shape'], transform=g['T'], fill=0, dtype='uint8')
    km = float(gdf.geometry.length.sum() / 1000)
    return b, t, km


def near(a, buf, px):
    if buf == 0:
        return a
    return binary_dilation(a, structure=disk(max(1, round(buf / px))))


def two_sided(pred, obs, buf, px, tot):
    npred, nobs = int(pred.sum()), int(obs.sum())
    tp = int((pred & near(obs, buf, px)).sum())
    fn = nobs - int((obs & near(pred, buf, px)).sum())
    fp = npred - tp
    tn = tot - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return dict(N_pred=npred, N_obs=nobs, TP=tp, FP=fp, FN=fn, Precision=round(p, 4), Recall=round(r, 4),
                F1=round(f1, 4), IoU=round(iou, 4))


def percell_map(pred, ref, buf, px, mask):
    rn, pn = near(ref, buf, px), near(pred, buf, px)
    m = np.zeros(pred.shape, np.uint8)
    m[(pred & rn) | (ref & pn)] = 1
    m[pred & ~rn] = 2
    m[ref & ~pn] = 3
    m[~mask] = 0
    return m


def all_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    pred, obs = tp + fp, tp + fn
    p = tp / pred if pred else 0
    r = tp / obs if obs else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0
    po = (tp + tn) / total
    pe = ((tp + fp) * (tp + fn) + (fp + tn) * (fn + tn)) / total ** 2
    k = (po - pe) / (1 - pe) if 1 - pe else 0
    d = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (float(tp) * tn - float(fp) * fn) / d if d else 0
    qd = abs(pred - obs) / total
    ad = 2 * min(fp, fn) / total
    return dict(Precision=round(p, 3), Recall=round(r, 3), F1=round(f1, 3), IoU=round(iou, 3), Kappa=round(k, 3),
                MCC=round(mcc, 3), FoM=round(iou, 3), QD=round(qd, 4), AD=round(ad, 4), TD=round(qd + ad, 4),
                OA=round(po, 3))


def map_metrics(m, mask):
    tp, fp, fn = (int((m == k).sum()) for k in (1, 2, 3))
    tn = int(mask.sum()) - tp - fp - fn
    out = all_metrics(tp, fp, fn, tn)
    out.update(TP=tp, FP=fp, FN=fn, TN=tn)
    return out


def maxpool(a, f):
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:h, :w].reshape(h // f, f, w // f, f).any(axis=(1, 3))


def density_peaks(b, window=7, nms=5, q=0.70):
    k = np.ones((window, window), np.float32) / window ** 2
    d = ndimage.convolve(b.astype(np.float32), k, mode='constant', cval=0)
    d[~b] = 0
    v = d > 0
    if not v.any():
        return np.empty((0, 2))
    thr = np.quantile(d[v], q)
    peaks = (d == maximum_filter(d, size=2 * nms + 1)) & (d >= thr) & v
    rows, cols = np.where(peaks)
    order = np.argsort(-d[rows, cols])
    pts = np.column_stack([rows, cols]).astype(np.float64)
    tree = cKDTree(pts)
    sup = np.zeros(len(pts), bool)
    kept = []
    for i in order:
        if sup[i]:
            continue
        kept.append(i)
        for j in tree.query_ball_point(pts[i], nms):
            if j != i:
                sup[j] = True
    return pts[kept]


def locational_accuracy(pred, obs, px, buffers, sample_n=50000):
    sp = np.argwhere(skeletonize(pred)).astype(np.float64)
    so = np.argwhere(skeletonize(obs)).astype(np.float64)
    np.random.seed(42)
    ip = np.random.choice(len(sp), sample_n, replace=False) if len(sp) > sample_n else np.arange(len(sp))
    io = np.random.choice(len(so), sample_n, replace=False) if len(so) > sample_n else np.arange(len(so))
    dp2o, _ = cKDTree(so).query(sp[ip], k=1)
    do2p, _ = cKDTree(sp).query(so[io], k=1)
    res = dict(skel_mean_sym=round((dp2o.mean() + do2p.mean()) / 2 * px), skel_med_sym=round((np.median(dp2o) + np.median(do2p)) / 2 * px),
               skel_mean_p2o=round(dp2o.mean() * px), skel_mean_o2p=round(do2p.mean() * px))
    pp, po = density_peaks(pred), density_peaks(obs)
    res.update(n_peaks_pred=len(pp), n_peaks_obs=len(po), peak_ratio=round(len(pp) / max(len(po), 1), 3))
    a, _ = cKDTree(po).query(pp, k=1)
    b, _ = cKDTree(pp).query(po, k=1)
    rows = []
    for bm in buffers:
        bp = bm / px
        corr = float((dp2o <= bp).mean())
        comp = float((do2p <= bp).mean())
        mr = float((a <= bp).mean())
        dr = float((b <= bp).mean())
        rows.append(dict(buffer_m=bm, correctness=round(corr, 3), completeness=round(comp, 3),
                         match_rate=round(mr, 3), detect_rate=round(dr, 3), conv_f1=round(2 * mr * dr / (mr + dr) if mr + dr else 0, 3)))
    res['table'] = rows
    return res
