import os
DATA = os.environ.get("ROAD_DATA", "data")
import numpy as np, rasterio, gc, time, warnings
from scipy import ndimage
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')

OUT = os.path.join(DATA, "output")
DS = 3; EFF = 300 * DS

def maxpool(arr, f):
    h,w = arr.shape; nh=(h//f)*f; nw=(w//f)*f
    return arr[:nh,:nw].reshape(nh//f,f,nw//f,f).max(axis=(1,3)).astype(np.uint8)

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
    grid = {}
    kept = []
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
                            dominated = True
                            break
                if dominated: break
            if dominated: break
        if not dominated:
            kept.append(idx)
            key = (gr, gc_)
            if key not in grid:
                grid[key] = []
            grid[key].append((r, c))
    return np.array([[rows[i], cols[i]] for i in kept], dtype=np.float64)

print("Phase 2: Convergence points")
with rasterio.open(f"{OUT}/BRA_spatial_TPFPFN_exact_20260607.tif") as src:
    d = src.read(1)
    shape_orig = d.shape

pred = maxpool(((d==1)|(d==2)).astype(np.uint8), DS)
obs  = maxpool(((d==1)|(d==3)).astype(np.uint8), DS)
del d; gc.collect()
shape = pred.shape
print(f"  DS shape: {shape}")

WINDOW = 7; NMS_R = 5
kern = np.ones((WINDOW,WINDOW), dtype=np.float32)/(WINDOW*WINDOW)

t0 = time.time()
dp = ndimage.convolve(pred.astype(np.float32), kern, mode='constant', cval=0)
dp[pred==0]=0; del pred; gc.collect()
pp = find_peaks_fast(dp, NMS_R); del dp; gc.collect()
print(f"  Pred peaks: {len(pp):,} ({time.time()-t0:.1f}s)")

t0 = time.time()
do_ = ndimage.convolve(obs.astype(np.float32), kern, mode='constant', cval=0)
do_[obs==0]=0; del obs; gc.collect()
po = find_peaks_fast(do_, NMS_R); del do_; gc.collect()
print(f"  Obs peaks: {len(po):,} ({time.time()-t0:.1f}s)")

tree_po = cKDTree(po); tree_pp = cKDTree(pp)
d_p2o, _ = tree_po.query(pp, k=1)
d_o2p, _ = tree_pp.query(po, k=1)

int_bufs = [0, 900, 1800, 2700, 3600, 5400, 9000]
print(f"\n  {'Buffer':>8} {'MR':>8} {'DR':>8} {'F1':>8}")
int_mr=[]; int_dr=[]; int_f1=[]
for b in int_bufs:
    bp = b/EFF
    mp = int((d_p2o<=bp).sum()) if bp>0 else int((d_p2o==0).sum())
    mo = int((d_o2p<=bp).sum()) if bp>0 else int((d_o2p==0).sum())
    mr = mp/len(pp); dr = mo/len(po)
    f1 = 2*mr*dr/(mr+dr) if (mr+dr)>0 else 0
    int_mr.append(mr); int_dr.append(dr); int_f1.append(f1)
    print(f"  {b:>6}m   {mr:>8.3f}   {dr:>8.3f}   {f1:>8.3f}")

np.savez_compressed(f"{OUT}/BRA_convergence_raw.npz",
    pp=pp, po=po, d_p2o=d_p2o, d_o2p=d_o2p,
    int_bufs=np.array(int_bufs), int_mr=np.array(int_mr),
    int_dr=np.array(int_dr), int_f1=np.array(int_f1),
    shape=np.array(shape))
print("  Saved convergence data.")
