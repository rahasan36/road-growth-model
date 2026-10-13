import os
DATA = os.environ.get("ROAD_DATA", "data")
import numpy as np, rasterio, csv, os, time, warnings
from skimage.morphology import skeletonize
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter
warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(DATA, "output")
PS = 300

print("=" * 72)
print("  NIGERIA — Locational Accuracy")
print("=" * 72)

with rasterio.open(f"{OUT_DIR}/NGA_spatial_TPFPFN_exact_20260607.tif") as src:
    d = src.read(1)
pred = ((d==1)|(d==2)).astype(np.uint8)
obs  = ((d==1)|(d==3)).astype(np.uint8)
del d
print(f"  Shape: {pred.shape}, pred={pred.sum():,}, obs={obs.sum():,}")

print("\n  Skeletonizing...")
t0 = time.time()
sk_p = skeletonize(pred > 0).astype(np.uint8)
sk_o = skeletonize(obs > 0).astype(np.uint8)
print(f"  Skeleton pred={sk_p.sum():,}  obs={sk_o.sum():,}  ({time.time()-t0:.1f}s)")

pc = np.argwhere(sk_p > 0).astype(np.float64)
oc = np.argwhere(sk_o > 0).astype(np.float64)
tree_o = cKDTree(oc); tree_p = cKDTree(pc)

np.random.seed(42)
N = 50000
ip = np.random.choice(len(pc), min(N, len(pc)), replace=False)
io_ = np.random.choice(len(oc), min(N, len(oc)), replace=False)
dp2o, _ = tree_o.query(pc[ip], k=1)
do2p, _ = tree_p.query(oc[io_], k=1)
dp2o_m = dp2o * PS; do2p_m = do2p * PS

print(f"\n  Skeleton Distances:")
print(f"    Pred→Obs:  mean={dp2o_m.mean():.0f}m  median={np.median(dp2o_m):.0f}m  90th={np.percentile(dp2o_m,90):.0f}m")
print(f"    Obs→Pred:  mean={do2p_m.mean():.0f}m  median={np.median(do2p_m):.0f}m  90th={np.percentile(do2p_m,90):.0f}m")
print(f"    Symmetric: mean={(dp2o_m.mean()+do2p_m.mean())/2:.0f}m  median={(np.median(dp2o_m)+np.median(do2p_m))/2:.0f}m")

skel_rows = []
print(f"\n  {'Buffer':>8} {'Correctness':>12} {'Completeness':>13} {'Quality':>8}")
for buf_m in [0, 300, 600, 900, 1200, 1500]:
    bp = buf_m / PS
    corr = float((dp2o <= bp).mean())
    comp = float((do2p <= bp).mean())
    qual = corr*comp/(corr+comp-corr*comp) if (corr+comp)>0 else 0
    skel_rows.append(dict(buffer_m=buf_m, correctness=round(corr,4), completeness=round(comp,4), quality=round(qual,4)))
    print(f"  {buf_m:>6}m   {corr:>12.3f}   {comp:>13.3f}   {qual:>8.3f}")

with open(f"{OUT_DIR}/NGA_skeleton_distance_metrics_20260612.csv",'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=skel_rows[0].keys()); w.writeheader(); w.writerows(skel_rows)

del sk_p, sk_o, pc, oc, tree_o, tree_p, dp2o, do2p

print("\n  Computing convergence points...")
t0 = time.time()
WINDOW = 7; NMS_R = 5
kern = np.ones((WINDOW,WINDOW), dtype=np.float32)/(WINDOW*WINDOW)
dp = ndimage.convolve(pred.astype(np.float32), kern, mode='constant', cval=0)
do_ = ndimage.convolve(obs.astype(np.float32), kern, mode='constant', cval=0)
dp[pred==0]=0; do_[obs==0]=0

def find_peaks(density, nms_r=5, q=0.70):
    valid = density > 0
    if not valid.any(): return np.empty((0,2))
    thresh = np.quantile(density[valid], q)
    cand = density >= thresh
    lm = maximum_filter(density, size=2*nms_r+1)
    peaks = (density == lm) & cand & (density > 0)
    rows, cols = np.where(peaks)
    if len(rows)==0: return np.empty((0,2))
    dens = density[rows, cols]
    order = np.argsort(-dens)
    kept=[]; supp=set()
    for idx in order:
        if idx in supp: continue
        kept.append(idx)
        r,c = rows[idx], cols[idx]
        for other in order:
            if other in supp or other==idx: continue
            if abs(rows[other]-r)<=nms_r and abs(cols[other]-c)<=nms_r:
                if np.sqrt((rows[other]-r)**2+(cols[other]-c)**2)<=nms_r:
                    supp.add(other)
    return np.array([[rows[i],cols[i]] for i in kept], dtype=np.float64)

pp = find_peaks(dp, NMS_R); po = find_peaks(do_, NMS_R)
print(f"  Peaks: pred={len(pp):,}  obs={len(po):,}  ratio={len(pp)/max(len(po),1):.3f}  ({time.time()-t0:.1f}s)")

tree_po = cKDTree(po); tree_pp = cKDTree(pp)
d_p2o, _ = tree_po.query(pp, k=1)
d_o2p, _ = tree_pp.query(po, k=1)
print(f"  NN: pred→obs mean={d_p2o.mean()*PS:.0f}m median={np.median(d_p2o)*PS:.0f}m")
print(f"      obs→pred mean={d_o2p.mean()*PS:.0f}m median={np.median(d_o2p)*PS:.0f}m")

int_rows = []
print(f"\n  {'Buffer':>8} {'MR':>8} {'DR':>8} {'F1':>8}")
for buf_m in [0, 300, 600, 900, 1200, 1500, 2100, 3000]:
    bp = buf_m/PS
    mp = int((d_p2o<=bp).sum()) if bp>0 else int((d_p2o==0).sum())
    mo = int((d_o2p<=bp).sum()) if bp>0 else int((d_o2p==0).sum())
    mr = mp/len(pp); dr = mo/len(po)
    f1 = 2*mr*dr/(mr+dr) if (mr+dr)>0 else 0
    int_rows.append(dict(buffer_m=buf_m, match_rate=round(mr,4), detect_rate=round(dr,4),
                         f1=round(f1,4), matched_pred=mp, matched_obs=mo,
                         n_pred=len(pp), n_obs=len(po)))
    print(f"  {buf_m:>6}m   {mr:>8.3f}   {dr:>8.3f}   {f1:>8.3f}")

with open(f"{OUT_DIR}/NGA_intersection_matching_20260612.csv",'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=int_rows[0].keys()); w.writeheader(); w.writerows(int_rows)

print("\n  Nigeria DONE.")
