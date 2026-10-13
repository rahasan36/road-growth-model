import os
DATA = os.environ.get("ROAD_DATA", "data")
import numpy as np
import rasterio
from skimage.morphology import skeletonize
from scipy import ndimage
from scipy.spatial import cKDTree
import csv, os, time, warnings
warnings.filterwarnings('ignore')

TPFPFN_PATH = os.path.join(DATA, "output/BGD_spatial_TPFPFN_exact_20260607.tif")
OUT_DIR = os.path.join(DATA, "output")
PIXEL_SIZE = 300

print("=" * 72)
print("LOCATIONAL ACCURACY ASSESSMENT — Bangladesh")
print("=" * 72)

print("\n[1] Loading TPFPFN → binary rasters...")
with rasterio.open(TPFPFN_PATH) as src:
    tpfpfn = src.read(1)
    transform = src.transform
    crs = src.crs
    profile = src.profile.copy()

pred_bin = ((tpfpfn == 1) | (tpfpfn == 2)).astype(np.uint8)
obs_bin  = ((tpfpfn == 1) | (tpfpfn == 3)).astype(np.uint8)
print(f"  Shape: {pred_bin.shape}, ~{PIXEL_SIZE}m/px")
print(f"  Predicted: {pred_bin.sum():,} road pixels")
print(f"  Observed:  {obs_bin.sum():,} road pixels")

print("\n" + "=" * 72)
print("PART A: SKELETON DISTANCE METRICS")
print("=" * 72)

print("\n[A1] Skeletonizing...")
t0 = time.time()
skel_pred = skeletonize(pred_bin > 0).astype(np.uint8)
skel_obs  = skeletonize(obs_bin > 0).astype(np.uint8)
print(f"  Predicted skeleton: {skel_pred.sum():,} pixels")
print(f"  Observed skeleton:  {skel_obs.sum():,} pixels")
print(f"  Time: {time.time()-t0:.1f}s")

print("\n[A2] Computing skeleton-to-skeleton distances...")
t0 = time.time()
pred_coords = np.argwhere(skel_pred > 0).astype(np.float64)
obs_coords  = np.argwhere(skel_obs > 0).astype(np.float64)
print(f"  Predicted skeleton coords: {len(pred_coords):,}")
print(f"  Observed skeleton coords:  {len(obs_coords):,}")

tree_obs  = cKDTree(obs_coords)
tree_pred = cKDTree(pred_coords)

SAMPLE_SIZE = 50000
np.random.seed(42)

if len(pred_coords) > SAMPLE_SIZE:
    idx_p = np.random.choice(len(pred_coords), SAMPLE_SIZE, replace=False)
    pred_sample = pred_coords[idx_p]
else:
    pred_sample = pred_coords

if len(obs_coords) > SAMPLE_SIZE:
    idx_o = np.random.choice(len(obs_coords), SAMPLE_SIZE, replace=False)
    obs_sample = obs_coords[idx_o]
else:
    obs_sample = obs_coords

d_pred2obs, _ = tree_obs.query(pred_sample, k=1)
d_obs2pred, _ = tree_pred.query(obs_sample, k=1)

d_pred2obs_m = d_pred2obs * PIXEL_SIZE
d_obs2pred_m = d_obs2pred * PIXEL_SIZE

hausdorff_pred2obs = d_pred2obs_m.max()
hausdorff_obs2pred = d_obs2pred_m.max()
hausdorff_sym = max(hausdorff_pred2obs, hausdorff_obs2pred)

mean_pred2obs = d_pred2obs_m.mean()
med_pred2obs  = np.median(d_pred2obs_m)
mean_obs2pred = d_obs2pred_m.mean()
med_obs2pred  = np.median(d_obs2pred_m)

mean_sym = (mean_pred2obs + mean_obs2pred) / 2
med_sym = (med_pred2obs + med_obs2pred) / 2

print(f"  Time: {time.time()-t0:.1f}s")
print(f"\n  Skeleton distance metrics (sampled {SAMPLE_SIZE:,} points each):")
print(f"    Direction       | Mean (m)  | Median (m) | 90th %ile | Hausdorff (m)")
print(f"    ────────────────┼───────────┼────────────┼───────────┼──────────────")
print(f"    Pred→Obs (corr) | {mean_pred2obs:>8.0f}  | {med_pred2obs:>9.0f}  | {np.percentile(d_pred2obs_m, 90):>8.0f}  | {hausdorff_pred2obs:>12.0f}")
print(f"    Obs→Pred (comp) | {mean_obs2pred:>8.0f}  | {med_obs2pred:>9.0f}  | {np.percentile(d_obs2pred_m, 90):>8.0f}  | {hausdorff_obs2pred:>12.0f}")
print(f"    Symmetric       | {mean_sym:>8.0f}  | {med_sym:>9.0f}  |           | {hausdorff_sym:>12.0f}")

print("\n  Skeleton overlap at buffer distances:")
print(f"    {'Buffer':>8} | {'Correctness':>12} | {'Completeness':>13} | {'Quality':>8}")
print(f"    {'':>8} | {'(pred in obs)':>12} | {'(obs in pred)':>13} | {'':>8}")
print(f"    {'':>8}─┼──────────────┼───────────────┼──────────")

skel_results_a = []
for buf_m in [0, 300, 600, 900, 1200, 1500]:
    buf_px = buf_m / PIXEL_SIZE
    correctness = (d_pred2obs <= buf_px).mean()
    completeness = (d_obs2pred <= buf_px).mean()
    quality = correctness * completeness / (correctness + completeness - correctness * completeness) if (correctness + completeness) > 0 else 0

    skel_results_a.append(dict(buffer_m=buf_m, correctness=round(correctness, 4),
                                completeness=round(completeness, 4), quality=round(quality, 4)))
    print(f"    {buf_m:>6}m | {correctness:>12.3f} | {completeness:>13.3f} | {quality:>8.3f}")


print("\n" + "=" * 72)
print("PART B: INTERSECTION MATCHING (density-peak approach)")
print("=" * 72)


print("\n[B1] Computing local road density (7×7 window = ~2.1 km)...")
t0 = time.time()
WINDOW = 7
kernel_density = np.ones((WINDOW, WINDOW), dtype=np.float32) / (WINDOW * WINDOW)

density_pred = ndimage.convolve(pred_bin.astype(np.float32), kernel_density, mode='constant', cval=0)
density_obs  = ndimage.convolve(obs_bin.astype(np.float32), kernel_density, mode='constant', cval=0)

density_pred[pred_bin == 0] = 0
density_obs[obs_bin == 0] = 0

print(f"  Predicted density: max={density_pred.max():.3f}, mean={density_pred[pred_bin>0].mean():.3f}")
print(f"  Observed density:  max={density_obs.max():.3f}, mean={density_obs[obs_bin>0].mean():.3f}")
print(f"  Time: {time.time()-t0:.1f}s")

print("\n[B2] Finding density peaks via non-maximum suppression...")
t0 = time.time()
NMS_RADIUS = 5

def find_density_peaks(density, nms_radius=5, threshold_quantile=0.75):
    valid = density > 0
    if not valid.any():
        return np.empty((0, 2))

    threshold = np.quantile(density[valid], threshold_quantile)
    candidates = density >= threshold

    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(density, size=2*nms_radius+1)

    peaks = (density == local_max) & candidates & (density > 0)

    rows, cols = np.where(peaks)
    if len(rows) == 0:
        return np.empty((0, 2))

    densities = density[rows, cols]
    order = np.argsort(-densities)

    kept = []
    suppressed = set()
    for idx in order:
        if idx in suppressed:
            continue
        kept.append(idx)
        r, c = rows[idx], cols[idx]
        for other in order:
            if other in suppressed or other == idx:
                continue
            if abs(rows[other] - r) <= nms_radius and abs(cols[other] - c) <= nms_radius:
                dist = np.sqrt((rows[other]-r)**2 + (cols[other]-c)**2)
                if dist <= nms_radius:
                    suppressed.add(other)

    return np.array([[rows[i], cols[i]] for i in kept], dtype=np.float64)

peaks_pred = find_density_peaks(density_pred, NMS_RADIUS, 0.70)
peaks_obs  = find_density_peaks(density_obs, NMS_RADIUS, 0.70)

print(f"  Predicted density peaks: {len(peaks_pred):,}")
print(f"  Observed density peaks:  {len(peaks_obs):,}")
print(f"  Ratio (pred/obs):        {len(peaks_pred)/max(len(peaks_obs),1):.3f}")
print(f"  Time: {time.time()-t0:.1f}s")

print("\n[B3] Computing intersection matching metrics...")
t0 = time.time()

BUFFERS_B = [0, 300, 600, 900, 1200, 1500, 2100, 3000]

def match_pts(pts_a, pts_b, buf_px):
    if len(pts_a) == 0 or len(pts_b) == 0:
        return 0, np.array([])
    tree = cKDTree(pts_b)
    dists, _ = tree.query(pts_a, k=1)
    matched = int((dists <= buf_px).sum()) if buf_px > 0 else int((dists == 0).sum())
    return matched, dists

_, all_d_p2o = match_pts(peaks_pred, peaks_obs, 1e9)
_, all_d_o2p = match_pts(peaks_obs, peaks_pred, 1e9)

if len(all_d_p2o) > 0:
    print(f"\n  NN distances (pred→obs): mean={all_d_p2o.mean()*PIXEL_SIZE:.0f}m, "
          f"median={np.median(all_d_p2o)*PIXEL_SIZE:.0f}m, "
          f"90th={np.percentile(all_d_p2o, 90)*PIXEL_SIZE:.0f}m")
    print(f"  NN distances (obs→pred): mean={all_d_o2p.mean()*PIXEL_SIZE:.0f}m, "
          f"median={np.median(all_d_o2p)*PIXEL_SIZE:.0f}m, "
          f"90th={np.percentile(all_d_o2p, 90)*PIXEL_SIZE:.0f}m")

print(f"\n  {'Buffer':>8} | {'MR (corr)':>10} | {'DR (comp)':>10} | {'F1':>8} | {'Matched/Pred':>13} | {'Matched/Obs':>13}")
print(f"  {'':>8}─┼────────────┼────────────┼──────────┼───────────────┼──────────────")

results_b = []
for buf_m in BUFFERS_B:
    buf_px = buf_m / PIXEL_SIZE
    m_p, d_p = match_pts(peaks_pred, peaks_obs, buf_px)
    m_o, d_o = match_pts(peaks_obs, peaks_pred, buf_px)

    mr = m_p / len(peaks_pred) if len(peaks_pred) else 0
    dr = m_o / len(peaks_obs) if len(peaks_obs) else 0
    f1 = 2*mr*dr/(mr+dr) if (mr+dr)>0 else 0

    if buf_px > 0 and len(d_p) > 0:
        md = d_p[d_p <= buf_px]
        avg_disp = md.mean() * PIXEL_SIZE if len(md) > 0 else 0
    else:
        avg_disp = 0

    results_b.append(dict(buffer_m=buf_m, n_pred=len(peaks_pred), n_obs=len(peaks_obs),
                          matched_pred=m_p, matched_obs=m_o,
                          match_rate=round(mr,4), detect_rate=round(dr,4),
                          f1=round(f1,4), avg_disp_m=round(avg_disp,1)))

    print(f"  {buf_m:>6}m | {mr:>10.3f} | {dr:>10.3f} | {f1:>8.3f} | {m_p:>5}/{len(peaks_pred):<5} | {m_o:>5}/{len(peaks_obs):<5}")

print(f"  Time: {time.time()-t0:.1f}s")


print("\n" + "=" * 72)
print("SAVING RESULTS")
print("=" * 72)

csv_a = os.path.join(OUT_DIR, "BGD_skeleton_distance_metrics_20260612.csv")
with open(csv_a, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=skel_results_a[0].keys())
    w.writeheader(); w.writerows(skel_results_a)
print(f"  {csv_a}")

csv_b = os.path.join(OUT_DIR, "BGD_intersection_matching_20260612.csv")
with open(csv_b, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results_b[0].keys())
    w.writeheader(); w.writerows(results_b)
print(f"  {csv_b}")

np.save(os.path.join(OUT_DIR, "BGD_density_peaks_pred_20260612.npy"), peaks_pred)
np.save(os.path.join(OUT_DIR, "BGD_density_peaks_obs_20260612.npy"), peaks_obs)

profile.update(dtype='uint8', count=1, nodata=0)
for name, peaks_arr in [('pred', peaks_pred), ('obs', peaks_obs)]:
    peak_rast = np.zeros(pred_bin.shape, dtype=np.uint8)
    for r, c in peaks_arr.astype(int):
        peak_rast[r, c] = 1
    p = os.path.join(OUT_DIR, f"BGD_density_peaks_{name}_20260612.tif")
    with rasterio.open(p, 'w', **profile) as dst:
        dst.write(peak_rast, 1)
print(f"  Peak rasters saved")


print("\n" + "=" * 72)
print("COMBINED LOCATIONAL ACCURACY SUMMARY")
print("=" * 72)

print("\n  A. Skeleton Distance Metrics:")
print(f"     Mean symmetric distance:   {mean_sym:.0f} m")
print(f"     Median symmetric distance: {med_sym:.0f} m")
print(f"     Hausdorff (symmetric):     {hausdorff_sym:,.0f} m")
print(f"     Skeleton correctness (300m buffer): {skel_results_a[1]['correctness']:.3f}")
print(f"     Skeleton completeness (300m buffer):{skel_results_a[1]['completeness']:.3f}")

print(f"\n  B. Road Convergence Point (Intersection) Matching:")
print(f"     Predicted convergence pts: {len(peaks_pred):,}")
print(f"     Observed convergence pts:  {len(peaks_obs):,}")
print(f"     Count ratio:               {len(peaks_pred)/max(len(peaks_obs),1):.3f}")
if len(all_d_p2o) > 0:
    print(f"     Mean NN distance:          {all_d_p2o.mean()*PIXEL_SIZE:,.0f} m")
    print(f"     Median NN distance:        {np.median(all_d_p2o)*PIXEL_SIZE:,.0f} m")
for r in results_b:
    if r['buffer_m'] in [0, 300, 600, 900, 1500, 3000]:
        print(f"     Buffer {r['buffer_m']:>5}m: MR={r['match_rate']:.3f}  DR={r['detect_rate']:.3f}  F1={r['f1']:.3f}")

print("\n" + "=" * 72)

summary_path = os.path.join(OUT_DIR, "BGD_locational_accuracy_summary_20260612.txt")
with open(summary_path, 'w') as f:
    f.write("LOCATIONAL ACCURACY ASSESSMENT — Bangladesh\n")
    f.write("=" * 60 + "\n\n")
    f.write("A. Skeleton Distance Metrics\n")
    f.write(f"   Mean correctness distance (pred→obs): {mean_pred2obs:.0f} m\n")
    f.write(f"   Mean completeness distance (obs→pred): {mean_obs2pred:.0f} m\n")
    f.write(f"   Mean symmetric distance: {mean_sym:.0f} m\n")
    f.write(f"   Median symmetric distance: {med_sym:.0f} m\n")
    f.write(f"   90th percentile (pred→obs): {np.percentile(d_pred2obs_m, 90):.0f} m\n")
    f.write(f"   90th percentile (obs→pred): {np.percentile(d_obs2pred_m, 90):.0f} m\n")
    f.write(f"   Hausdorff (symmetric): {hausdorff_sym:.0f} m\n\n")
    f.write("   Skeleton overlap:\n")
    for r in skel_results_a:
        f.write(f"   {r['buffer_m']:>5}m: correctness={r['correctness']:.4f}  completeness={r['completeness']:.4f}  quality={r['quality']:.4f}\n")
    f.write(f"\nB. Intersection (Density Peak) Matching\n")
    f.write(f"   Predicted peaks: {len(peaks_pred)}\n")
    f.write(f"   Observed peaks: {len(peaks_obs)}\n")
    f.write(f"   Mean NN (pred→obs): {all_d_p2o.mean()*PIXEL_SIZE:.0f} m\n")
    f.write(f"   Median NN (pred→obs): {np.median(all_d_p2o)*PIXEL_SIZE:.0f} m\n\n")
    for r in results_b:
        f.write(f"   {r['buffer_m']:>5}m: MR={r['match_rate']:.4f}  DR={r['detect_rate']:.4f}  F1={r['f1']:.4f}\n")

print(f"\n  Summary: {summary_path}")
print("DONE.")
