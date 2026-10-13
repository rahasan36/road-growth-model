import os
DATA = os.environ.get("ROAD_DATA", "data")
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from scipy import ndimage
import csv

OUT_DIR = os.path.join(DATA, "output")
PIXEL_SIZE = 300

print("Loading data...")
with rasterio.open(f"{OUT_DIR}/BGD_spatial_TPFPFN_exact_20260607.tif") as src:
    tpfpfn = src.read(1)
pred_bin = ((tpfpfn == 1) | (tpfpfn == 2)).astype(np.uint8)
obs_bin  = ((tpfpfn == 1) | (tpfpfn == 3)).astype(np.uint8)

print("Skeletonizing...")
skel_pred = skeletonize(pred_bin > 0).astype(np.uint8)
skel_obs  = skeletonize(obs_bin > 0).astype(np.uint8)

print("Computing skeleton distances...")
pred_coords = np.argwhere(skel_pred > 0).astype(np.float64)
obs_coords  = np.argwhere(skel_obs > 0).astype(np.float64)
tree_obs = cKDTree(obs_coords)
tree_pred = cKDTree(pred_coords)

np.random.seed(42)
N = 50000
idx_p = np.random.choice(len(pred_coords), N, replace=False)
idx_o = np.random.choice(len(obs_coords), N, replace=False)
d_p2o, _ = tree_obs.query(pred_coords[idx_p], k=1)
d_o2p, _ = tree_pred.query(obs_coords[idx_o], k=1)
d_p2o_m = d_p2o * PIXEL_SIZE
d_o2p_m = d_o2p * PIXEL_SIZE

peaks_pred = np.load(f"{OUT_DIR}/BGD_density_peaks_pred_20260612.npy")
peaks_obs  = np.load(f"{OUT_DIR}/BGD_density_peaks_obs_20260612.npy")

skel_data = []
with open(f"{OUT_DIR}/BGD_skeleton_distance_metrics_20260612.csv") as f:
    for row in csv.DictReader(f):
        skel_data.append({k: float(v) for k, v in row.items()})

int_data = []
with open(f"{OUT_DIR}/BGD_intersection_matching_20260612.csv") as f:
    for row in csv.DictReader(f):
        int_data.append({k: float(v) for k, v in row.items()})

print("Creating figure...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

C_PRED = '#2166AC'
C_OBS  = '#B2182B'
C_F1   = '#4DAF4A'
C_QUAL = '#984EA3'

ax = axes[0, 0]
bins = np.arange(0, 3001, 50)
for d, color, label in [(d_p2o_m, C_PRED, 'Correctness (Pred→Obs)'),
                         (d_o2p_m, C_OBS, 'Completeness (Obs→Pred)')]:
    sorted_d = np.sort(d)
    cdf = np.arange(1, len(sorted_d)+1) / len(sorted_d)
    ax.plot(sorted_d, cdf, color=color, linewidth=1.5, label=label)

ax.axhline(0.90, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax.axvline(300, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
ax.set_xlim(0, 3000)
ax.set_ylim(0, 1.02)
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Cumulative fraction')
ax.set_title('(a) Skeleton distance CDF', fontweight='bold', fontsize=11)
ax.legend(fontsize=8, loc='lower right')
ax.text(350, 0.05, '1 pixel\n(300 m)', fontsize=7, color='gray', va='bottom')
ax.grid(True, alpha=0.2)

ax = axes[0, 1]
bufs = [r['buffer_m'] for r in skel_data]
corrs = [r['correctness'] for r in skel_data]
comps = [r['completeness'] for r in skel_data]
quals = [r['quality'] for r in skel_data]

ax.plot(bufs, corrs, 'o-', color=C_PRED, linewidth=2, markersize=5, label='Correctness')
ax.plot(bufs, comps, 's-', color=C_OBS, linewidth=2, markersize=5, label='Completeness')
ax.plot(bufs, quals, '^-', color=C_QUAL, linewidth=2, markersize=5, label='Quality')
ax.axhline(0.90, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_xlim(-50, 1600)
ax.set_ylim(0.25, 1.02)
ax.set_xlabel('Buffer tolerance (m)')
ax.set_ylabel('Fraction')
ax.set_title('(b) Skeleton overlap vs. buffer', fontweight='bold', fontsize=11)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.2)

ax.annotate(f'{corrs[1]:.3f}', (bufs[1], corrs[1]), textcoords="offset points",
            xytext=(8, 5), fontsize=7, color=C_PRED)
ax.annotate(f'{comps[1]:.3f}', (bufs[1], comps[1]), textcoords="offset points",
            xytext=(8, -12), fontsize=7, color=C_OBS)

ax = axes[1, 0]
bufs_b = [r['buffer_m'] for r in int_data]
mrs = [r['match_rate'] for r in int_data]
drs = [r['detect_rate'] for r in int_data]
f1s = [r['f1'] for r in int_data]

ax.plot(bufs_b, mrs, 'o-', color=C_PRED, linewidth=2, markersize=5, label='Match rate (precision)')
ax.plot(bufs_b, drs, 's-', color=C_OBS, linewidth=2, markersize=5, label='Detection rate (recall)')
ax.plot(bufs_b, f1s, '^-', color=C_F1, linewidth=2, markersize=5, label='F1 score')
ax.set_xlim(-100, 3200)
ax.set_ylim(0, 1.02)
ax.set_xlabel('Buffer tolerance (m)')
ax.set_ylabel('Rate')
ax.set_title('(c) Intersection matching vs. buffer', fontweight='bold', fontsize=11)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.2)

idx_1500 = [i for i, r in enumerate(int_data) if r['buffer_m'] == 1500][0]
ax.annotate(f'F1={f1s[idx_1500]:.3f}', (bufs_b[idx_1500], f1s[idx_1500]),
            textcoords="offset points", xytext=(8, -10), fontsize=7, color=C_F1)

ax = axes[1, 1]

road_union = ((pred_bin > 0) | (obs_bin > 0)).astype(np.float32)
ax.imshow(road_union, cmap='Greys', alpha=0.15, interpolation='nearest',
          extent=[0, pred_bin.shape[1]*PIXEL_SIZE/1000, pred_bin.shape[0]*PIXEL_SIZE/1000, 0])

ax.scatter(peaks_obs[:, 1]*PIXEL_SIZE/1000, peaks_obs[:, 0]*PIXEL_SIZE/1000,
           s=3, c=C_OBS, alpha=0.4, label=f'Observed ({len(peaks_obs):,})', zorder=2, linewidths=0)
ax.scatter(peaks_pred[:, 1]*PIXEL_SIZE/1000, peaks_pred[:, 0]*PIXEL_SIZE/1000,
           s=3, c=C_PRED, alpha=0.4, label=f'Predicted ({len(peaks_pred):,})', zorder=3, linewidths=0)

ax.set_xlabel('Easting (km)')
ax.set_ylabel('Northing (km)')
ax.set_title('(d) Intersection density peaks', fontweight='bold', fontsize=11)
ax.legend(fontsize=8, loc='lower left', markerscale=3)
ax.set_aspect('equal')

plt.tight_layout(pad=1.5)

fig_path = f"{OUT_DIR}/fig_BGD_locational_accuracy_20260612.png"
fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {fig_path}")

fig.savefig(fig_path.replace('.png', '_hires.png'), dpi=600, bbox_inches='tight', facecolor='white')
print(f"Saved high-res version")

plt.close()
print("Done.")
