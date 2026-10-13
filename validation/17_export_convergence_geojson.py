import os
DATA = os.environ.get("ROAD_DATA", "data")
import numpy as np, rasterio, gc, json, os, sys, time, warnings
from scipy import ndimage
from scipy.ndimage import maximum_filter
warnings.filterwarnings('ignore')

OUT = os.path.join(DATA, "output")
EXPORT = os.path.join(DATA, "RoadModel_SeeForData/geospatial_exports")
os.makedirs(EXPORT, exist_ok=True)

code = sys.argv[1]
WINDOW = 7; NMS_R = 5
kern = np.ones((WINDOW,WINDOW), dtype=np.float32)/(WINDOW*WINDOW)

def find_peaks_fast(density, nms_r=5, q=0.70):
    valid = density > 0
    if not valid.any(): return np.empty((0,2))
    thresh = np.quantile(density[valid], q)
    lm = maximum_filter(density, size=2*nms_r+1)
    peaks = (density == lm) & (density >= thresh) & (density > 0)
    del lm; gc.collect()
    rows, cols = np.where(peaks)
    if len(rows)==0: return np.empty((0,2))
    dens = density[rows, cols]
    order = np.argsort(-dens)
    grid = {}; kept = []
    for idx in order:
        r, c = int(rows[idx]), int(cols[idx])
        gr, gc_ = r // nms_r, c // nms_r
        dominated = False
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                key = (gr+dr, gc_+dc)
                if key in grid:
                    for kr, kc in grid[key]:
                        if np.sqrt((r-kr)**2 + (c-kc)**2) <= nms_r:
                            dominated = True; break
                if dominated: break
            if dominated: break
        if not dominated:
            kept.append(idx)
            key = (gr, gc_)
            if key not in grid: grid[key] = []
            grid[key].append((r, c))
    return np.array([[rows[i], cols[i]] for i in kept], dtype=np.float64)

def pixel_to_xy(transform, r, c):
    x = transform.c + c * transform.a + r * transform.b
    y = transform.f + c * transform.d + r * transform.e
    return float(x), float(y)

def save_geojson(points_rc, transform, crs, path, label):
    features = []
    for i, (r, c) in enumerate(points_rc):
        x, y = pixel_to_xy(transform, r, c)
        features.append({"type":"Feature",
            "geometry":{"type":"Point","coordinates":[x, y]},
            "properties":{"id":i, "type":label}})
    gj = {"type":"FeatureCollection",
          "crs":{"type":"name","properties":{"name":str(crs)}},
          "features":features}
    with open(path, 'w') as f:
        json.dump(gj, f)
    print(f"  Saved: {path} ({len(features):,} pts)")

if code == 'BRA':
    print(f"Brazil: loading from saved npz...")
    conv = np.load(f"{OUT}/BRA_convergence_raw.npz")
    pp = conv['pp']; po = conv['po']
    DS = 3
    with rasterio.open(f"{OUT}/BRA_spatial_TPFPFN_exact_20260607.tif") as src:
        transform_orig = src.transform
        crs = src.crs
    pp_orig = pp * DS
    po_orig = po * DS
    save_geojson(pp_orig, transform_orig, crs,
                 f"{EXPORT}/BRA_convergence_predicted.geojson", "predicted")
    save_geojson(po_orig, transform_orig, crs,
                 f"{EXPORT}/BRA_convergence_observed.geojson", "observed")
else:
    name = {'BGD':'Bangladesh','NGA':'Nigeria'}[code]
    print(f"{name}: computing convergence points...")
    with rasterio.open(f"{OUT}/{code}_spatial_TPFPFN_exact_20260607.tif") as src:
        d = src.read(1)
        transform = src.transform
        crs = src.crs
    pred = ((d==1)|(d==2)).astype(np.uint8)
    obs  = ((d==1)|(d==3)).astype(np.uint8)
    del d; gc.collect()

    t0 = time.time()
    dp = ndimage.convolve(pred.astype(np.float32), kern, mode='constant', cval=0)
    dp[pred==0]=0; del pred; gc.collect()
    pp = find_peaks_fast(dp, NMS_R); del dp; gc.collect()
    print(f"  Pred: {len(pp):,}")

    do_ = ndimage.convolve(obs.astype(np.float32), kern, mode='constant', cval=0)
    do_[obs==0]=0; del obs; gc.collect()
    po = find_peaks_fast(do_, NMS_R); del do_; gc.collect()
    print(f"  Obs: {len(po):,} ({time.time()-t0:.1f}s)")

    save_geojson(pp, transform, crs,
                 f"{EXPORT}/{code}_convergence_predicted.geojson", "predicted")
    save_geojson(po, transform, crs,
                 f"{EXPORT}/{code}_convergence_observed.geojson", "observed")

print("Done.")
