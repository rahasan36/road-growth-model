import os
DATA = os.environ.get("ROAD_DATA", "data")
import numpy as np, rasterio, gc, time, warnings, csv
from skimage.morphology import skeletonize
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(DATA, "output")
PS = 300

def find_peaks(density, nms_r=5, q=0.70):
    valid = density > 0
    if not valid.any(): return np.empty((0,2))
    thresh = np.quantile(density[valid], q)
    lm = maximum_filter(density, size=2*nms_r+1)
    peaks = (density == lm) & (density >= thresh) & (density > 0)
    del lm; gc.collect()
    rows, cols = np.where(peaks)
    del peaks; gc.collect()
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

print("=" * 60)
print("  NIGERIA — 4-Panel Locational Accuracy Figure")
print("=" * 60)

with rasterio.open(f"{OUT_DIR}/NGA_spatial_TPFPFN_exact_20260607.tif") as src:
    d = src.read(1)
    transform = src.transform
    crs = src.crs
pred = ((d==1)|(d==2)).astype(np.uint8)
obs  = ((d==1)|(d==3)).astype(np.uint8)
shape = pred.shape
del d; gc.collect()
print(f"  Shape: {shape}")

print("  Skeletonizing...")
t0 = time.time()
sk_p = skeletonize(pred > 0).astype(np.uint8)
sk_o = skeletonize(obs > 0).astype(np.uint8)
print(f"  Skeleton pred={sk_p.sum():,}  obs={sk_o.sum():,}  ({time.time()-t0:.1f}s)")

pc = np.argwhere(sk_p > 0).astype(np.float64)
oc = np.argwhere(sk_o > 0).astype(np.float64)
del sk_p, sk_o; gc.collect()

tree_o = cKDTree(oc); tree_p = cKDTree(pc)
np.random.seed(42); N = 50000
ip = np.random.choice(len(pc), min(N, len(pc)), replace=False)
io_ = np.random.choice(len(oc), min(N, len(oc)), replace=False)

print("  Computing skeleton distances...")
dp2o, _ = tree_o.query(pc[ip], k=1)
do2p, _ = tree_p.query(oc[io_], k=1)
dp2o_m = dp2o * PS; do2p_m = do2p * PS
del tree_o, tree_p, pc, oc; gc.collect()

print(f"    Pred→Obs: mean={dp2o_m.mean():.0f}m  median={np.median(dp2o_m):.0f}m")
print(f"    Obs→Pred: mean={do2p_m.mean():.0f}m  median={np.median(do2p_m):.0f}m")
print(f"    Symmetric: mean={(dp2o_m.mean()+do2p_m.mean())/2:.0f}m")

skel_bufs = [0, 300, 600, 900, 1200, 1500]
skel_corr = [float((dp2o <= b/PS).mean()) for b in skel_bufs]
skel_comp = [float((do2p <= b/PS).mean()) for b in skel_bufs]
skel_qual = [c*cm/(c+cm-c*cm) if (c+cm)>0 else 0 for c,cm in zip(skel_corr, skel_comp)]

del dp2o; gc.collect()

print("  Computing convergence points...")
t0 = time.time()
WINDOW = 7; NMS_R = 5
kern = np.ones((WINDOW,WINDOW), dtype=np.float32)/(WINDOW*WINDOW)
dp_d = ndimage.convolve(pred.astype(np.float32), kern, mode='constant', cval=0)
dp_d[pred==0]=0; del pred; gc.collect()
do_d = ndimage.convolve(obs.astype(np.float32), kern, mode='constant', cval=0)
do_d[obs==0]=0; del obs; gc.collect()

pp = find_peaks(dp_d, NMS_R); del dp_d; gc.collect()
po = find_peaks(do_d, NMS_R); del do_d; gc.collect()
print(f"  Peaks: pred={len(pp):,}  obs={len(po):,}  ratio={len(pp)/len(po):.3f}  ({time.time()-t0:.1f}s)")

tree_po = cKDTree(po); tree_pp = cKDTree(pp)
d_p2o_int, _ = tree_po.query(pp, k=1)
d_o2p_int, _ = tree_pp.query(po, k=1)

int_bufs = [0, 300, 600, 900, 1500, 2100, 3000]
int_mr = []; int_dr = []; int_f1 = []
for b in int_bufs:
    bp = b/PS
    mp = int((d_p2o_int<=bp).sum()) if bp>0 else int((d_p2o_int==0).sum())
    mo = int((d_o2p_int<=bp).sum()) if bp>0 else int((d_o2p_int==0).sum())
    mr = mp/len(pp); dr = mo/len(po)
    f1 = 2*mr*dr/(mr+dr) if (mr+dr)>0 else 0
    int_mr.append(mr); int_dr.append(dr); int_f1.append(f1)

print("  Plotting 4-panel figure...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

C_CORR = '#2166AC'; C_COMP = '#B2182B'; C_QUAL = '#984EA3'
C_MR = '#2166AC'; C_DR = '#B2182B'; C_F1 = '#4DAF4A'

ax = axes[0,0]
bins_p = np.sort(dp2o_m)
bins_o = np.sort(do2p_m)
cdf_p = np.arange(1, len(bins_p)+1) / len(bins_p)
cdf_o = np.arange(1, len(bins_o)+1) / len(bins_o)
ax.plot(bins_p, cdf_p, color=C_CORR, linewidth=1.5, label='Correctness (Pred→Obs)')
ax.plot(bins_o, cdf_o, color=C_COMP, linewidth=1.5, label='Completeness (Obs→Pred)')
ax.set_xlim(0, 5000)
ax.set_ylim(0, 1.05)
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Cumulative fraction')
ax.set_title('(a) Skeleton distance CDF', fontweight='bold', fontsize=12)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.2)
ax.text(0.02, 0.02, f'pred: {len(dp2o_m):,}\nobs: {len(do2p_m):,}',
        transform=ax.transAxes, fontsize=7, va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax = axes[0,1]
ax.plot(skel_bufs, skel_corr, 'o-', color=C_CORR, linewidth=2, markersize=6, label='Correctness')
ax.plot(skel_bufs, skel_comp, 's-', color=C_COMP, linewidth=2, markersize=6, label='Completeness')
ax.plot(skel_bufs, skel_qual, '^-', color=C_QUAL, linewidth=2, markersize=6, label='Quality')
ax.axhline(0.90, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_xlim(-50, 1600)
ax.set_ylim(0, 1.05)
ax.set_xlabel('Buffer tolerance (m)')
ax.set_ylabel('Fraction')
ax.set_title('(b) Skeleton overlap vs. buffer', fontweight='bold', fontsize=12)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.2)
for i, b in enumerate(skel_bufs):
    if b in [0, 300, 900, 1500]:
        ax.annotate(f'{skel_corr[i]:.3f}', (b, skel_corr[i]), textcoords="offset points",
                    xytext=(5, 8), fontsize=6, color=C_CORR)

ax = axes[1,0]
ax.plot(int_bufs, int_mr, 'o-', color=C_MR, linewidth=2, markersize=6, label='Match rate (precision)')
ax.plot(int_bufs, int_dr, 's-', color=C_DR, linewidth=2, markersize=6, label='Detection rate (recall)')
ax.plot(int_bufs, int_f1, '^-', color=C_F1, linewidth=2, markersize=6, label='F1 score')
ax.set_ylim(0, 1.05)
ax.set_xlabel('Buffer tolerance (m)')
ax.set_ylabel('Rate')
ax.set_title('(c) Intersection matching vs. buffer', fontweight='bold', fontsize=12)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.2)
for i, b in enumerate(int_bufs):
    if b in [1500, 3000]:
        ax.annotate(f'F1={int_f1[i]:.3f}', (b, int_f1[i]), textcoords="offset points",
                    xytext=(5, -15), fontsize=7, color=C_F1)

ax = axes[1,1]
pp_km_x = pp[:,1] * PS / 1000
pp_km_y = pp[:,0] * PS / 1000
po_km_x = po[:,1] * PS / 1000
po_km_y = po[:,0] * PS / 1000

ax.scatter(po_km_x, po_km_y, s=2, c=C_COMP, alpha=0.4, linewidths=0, label=f'Observed ({len(po):,})')
ax.scatter(pp_km_x, pp_km_y, s=2, c=C_CORR, alpha=0.4, linewidths=0, label=f'Predicted ({len(pp):,})')
ax.set_xlim(0, shape[1]*PS/1000)
ax.set_ylim(shape[0]*PS/1000, 0)
ax.set_xlabel('Easting (km)')
ax.set_ylabel('Northing (km)')
ax.set_title('(d) Intersection density peaks', fontweight='bold', fontsize=12)
ax.legend(fontsize=8, loc='upper right', markerscale=4)
ax.set_aspect('equal')

plt.suptitle('Nigeria — Locational Accuracy Assessment', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(pad=1.5)
out = f"{OUT_DIR}/fig_NGA_locational_accuracy_20260612.png"
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
out_hi = f"{OUT_DIR}/fig_NGA_locational_accuracy_20260612_hires.png"
fig.savefig(out_hi, dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"  Saved: {out}")
print(f"  Saved: {out_hi}")
print("  DONE.")
