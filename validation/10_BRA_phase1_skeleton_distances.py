import os
DATA = os.environ.get("ROAD_DATA", "data")
import numpy as np, rasterio, gc, time, warnings
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')

OUT = os.path.join(DATA, "output")
DS = 3; EFF = 300 * DS

print("Phase 1: Skeleton distances")
with rasterio.open(f"{OUT}/BRA_spatial_TPFPFN_exact_20260607.tif") as src:
    d = src.read(1)

def maxpool(arr, f):
    h,w = arr.shape; nh=(h//f)*f; nw=(w//f)*f
    return arr[:nh,:nw].reshape(nh//f,f,nw//f,f).max(axis=(1,3)).astype(np.uint8)

pred = maxpool(((d==1)|(d==2)).astype(np.uint8), DS)
obs  = maxpool(((d==1)|(d==3)).astype(np.uint8), DS)
del d; gc.collect()
print(f"  DS shape: {pred.shape}")

t0 = time.time()
sk_p = skeletonize(pred > 0).astype(np.uint8); del pred; gc.collect()
sk_o = skeletonize(obs > 0).astype(np.uint8); del obs; gc.collect()
print(f"  Skeleton: pred={sk_p.sum():,} obs={sk_o.sum():,} ({time.time()-t0:.1f}s)")

pc = np.argwhere(sk_p > 0).astype(np.float64); del sk_p; gc.collect()
oc = np.argwhere(sk_o > 0).astype(np.float64); del sk_o; gc.collect()

tree_o = cKDTree(oc); tree_p = cKDTree(pc)
np.random.seed(42); N = 50000
ip = np.random.choice(len(pc), min(N, len(pc)), replace=False)
io_ = np.random.choice(len(oc), min(N, len(oc)), replace=False)

dp2o, _ = tree_o.query(pc[ip], k=1)
do2p, _ = tree_p.query(oc[io_], k=1)
dp2o_m = dp2o * EFF; do2p_m = do2p * EFF
del tree_o, tree_p, pc, oc, dp2o, do2p; gc.collect()

print(f"  Pred→Obs: mean={dp2o_m.mean():.0f}m  median={np.median(dp2o_m):.0f}m")
print(f"  Obs→Pred: mean={do2p_m.mean():.0f}m  median={np.median(do2p_m):.0f}m")

np.savez_compressed(f"{OUT}/BRA_skeleton_distances_raw.npz", dp2o_m=dp2o_m, do2p_m=do2p_m)
print("  Saved skeleton distances.")
