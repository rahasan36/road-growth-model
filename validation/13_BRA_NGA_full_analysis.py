import os
DATA = os.environ.get("ROAD_DATA", "data")
import numpy as np
import rasterio
from skimage.morphology import skeletonize, disk, binary_dilation
from scipy import ndimage
from scipy.spatial import cKDTree
import csv, os, time, warnings, math
warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(DATA, "output")

def compute_all_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    pred = tp + fp
    obs = tp + fn

    prec = tp / pred if pred > 0 else 0
    rec  = tp / obs  if obs > 0  else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    po = (tp + tn) / total
    pe = ((tp+fp)*(tp+fn) + (fp+tn)*(fn+tn)) / (total * total)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

    num = tp * tn - fp * fn
    denom = math.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0 else 1
    mcc = num / denom if denom > 0 else 0

    fom = iou

    qd = abs(pred - obs) / total
    ad = 2 * min(fp, fn) / total
    td = qd + ad

    oa = (tp + tn) / total if total > 0 else 0

    return dict(
        Precision=round(prec, 3), Recall=round(rec, 3), F1=round(f1, 3),
        IoU=round(iou, 3), Kappa=round(kappa, 3), MCC=round(mcc, 3),
        FoM=round(fom, 3), QD=round(qd, 4), AD=round(ad, 4),
        TD=round(td, 4), OA=round(oa, 3)
    )


def locational_accuracy(pred_bin, obs_bin, pixel_size, country, sample_n=50000):
    results = {}

    print(f"    Skeletonizing...")
    t0 = time.time()
    skel_pred = skeletonize(pred_bin > 0).astype(np.uint8)
    skel_obs  = skeletonize(obs_bin > 0).astype(np.uint8)
    print(f"    Skeleton pred={skel_pred.sum():,}  obs={skel_obs.sum():,}  ({time.time()-t0:.1f}s)")

    pred_coords = np.argwhere(skel_pred > 0).astype(np.float64)
    obs_coords  = np.argwhere(skel_obs > 0).astype(np.float64)

    print(f"    Building KD-trees and computing distances...")
    t0 = time.time()
    np.random.seed(42)
    n_p = min(sample_n, len(pred_coords))
    n_o = min(sample_n, len(obs_coords))
    idx_p = np.random.choice(len(pred_coords), n_p, replace=False) if len(pred_coords) > n_p else np.arange(len(pred_coords))
    idx_o = np.random.choice(len(obs_coords), n_o, replace=False) if len(obs_coords) > n_o else np.arange(len(obs_coords))

    tree_obs = cKDTree(obs_coords)
    tree_pred = cKDTree(pred_coords)

    d_p2o, _ = tree_obs.query(pred_coords[idx_p], k=1)
    d_o2p, _ = tree_pred.query(obs_coords[idx_o], k=1)

    d_p2o_m = d_p2o * pixel_size
    d_o2p_m = d_o2p * pixel_size
    print(f"    Distances computed ({time.time()-t0:.1f}s)")

    mean_p2o = d_p2o_m.mean()
    med_p2o  = np.median(d_p2o_m)
    p90_p2o  = np.percentile(d_p2o_m, 90)
    mean_o2p = d_o2p_m.mean()
    med_o2p  = np.median(d_o2p_m)
    p90_o2p  = np.percentile(d_o2p_m, 90)
    mean_sym = (mean_p2o + mean_o2p) / 2
    med_sym  = (med_p2o + med_o2p) / 2

    results['skel_mean_sym'] = round(mean_sym, 0)
    results['skel_med_sym'] = round(med_sym, 0)
    results['skel_mean_p2o'] = round(mean_p2o, 0)
    results['skel_mean_o2p'] = round(mean_o2p, 0)
    results['skel_p90_p2o'] = round(p90_p2o, 0)
    results['skel_p90_o2p'] = round(p90_o2p, 0)

    skel_table = []
    for buf_m in [0, 300, 600, 900, 1200, 1500]:
        buf_px = buf_m / pixel_size
        corr = float((d_p2o <= buf_px).mean())
        comp = float((d_o2p <= buf_px).mean())
        qual = corr * comp / (corr + comp - corr * comp) if (corr + comp) > 0 else 0
        skel_table.append(dict(buffer_m=buf_m, correctness=round(corr, 4),
                                completeness=round(comp, 4), quality=round(qual, 4)))
    results['skel_table'] = skel_table

    print(f"    Computing road density peaks...")
    t0 = time.time()
    WINDOW = 7
    NMS_RADIUS = 5
    kernel_density = np.ones((WINDOW, WINDOW), dtype=np.float32) / (WINDOW * WINDOW)

    density_pred = ndimage.convolve(pred_bin.astype(np.float32), kernel_density, mode='constant', cval=0)
    density_obs  = ndimage.convolve(obs_bin.astype(np.float32), kernel_density, mode='constant', cval=0)
    density_pred[pred_bin == 0] = 0
    density_obs[obs_bin == 0] = 0

    def find_density_peaks(density, nms_radius=5, threshold_quantile=0.70):
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
    print(f"    Peaks: pred={len(peaks_pred):,}  obs={len(peaks_obs):,}  ({time.time()-t0:.1f}s)")

    results['n_peaks_pred'] = len(peaks_pred)
    results['n_peaks_obs'] = len(peaks_obs)
    results['peak_ratio'] = round(len(peaks_pred) / max(len(peaks_obs), 1), 3)

    if len(peaks_pred) > 0 and len(peaks_obs) > 0:
        tree_po = cKDTree(peaks_obs)
        tree_pp = cKDTree(peaks_pred)
        dp2o, _ = tree_po.query(peaks_pred, k=1)
        do2p, _ = tree_pp.query(peaks_obs, k=1)
        results['peak_mean_nn'] = round(dp2o.mean() * pixel_size, 0)
        results['peak_med_nn'] = round(np.median(dp2o) * pixel_size, 0)

        int_table = []
        for buf_m in [0, 300, 600, 900, 1200, 1500, 2100, 3000]:
            buf_px = buf_m / pixel_size
            mp = int((dp2o <= buf_px).sum()) if buf_px > 0 else int((dp2o == 0).sum())
            mo = int((do2p <= buf_px).sum()) if buf_px > 0 else int((do2p == 0).sum())
            mr = mp / len(peaks_pred)
            dr = mo / len(peaks_obs)
            f1 = 2*mr*dr/(mr+dr) if (mr+dr)>0 else 0
            int_table.append(dict(buffer_m=buf_m, match_rate=round(mr,4),
                                   detect_rate=round(dr,4), f1=round(f1,4),
                                   matched_pred=mp, matched_obs=mo,
                                   n_pred=len(peaks_pred), n_obs=len(peaks_obs)))
        results['int_table'] = int_table
    else:
        results['peak_mean_nn'] = 0
        results['peak_med_nn'] = 0
        results['int_table'] = []

    return results


COUNTRIES = {
    'BRA': {'name': 'Brazil',  'pixel_size': 300},
    'NGA': {'name': 'Nigeria', 'pixel_size': 300},
}

for code, info in COUNTRIES.items():
    name = info['name']
    ps = info['pixel_size']

    print("\n" + "=" * 72)
    print(f"  {name.upper()} ({code})")
    print("=" * 72)

    print(f"\n  [1] Extended pixel-level metrics...")

    all_metrics = []
    for buf_label, buf_file in [('exact', 'exact'), ('300m', '300m'), ('600m', '600m'), ('900m', '900m')]:
        path = f"{OUT_DIR}/{code}_spatial_TPFPFN_{buf_file}_20260607.tif"
        with rasterio.open(path) as src:
            d = src.read(1)

        tp = int((d == 1).sum())
        fp = int((d == 2).sum())
        fn = int((d == 3).sum())
        tn = int(d.size - tp - fp - fn)

        m = compute_all_metrics(tp, fp, fn, tn)
        m['Buffer'] = buf_label
        m['TP'] = tp
        m['FP'] = fp
        m['FN'] = fn
        m['TN'] = tn
        all_metrics.append(m)

    print(f"\n  GRIP 2015  Forecast total network vs observed total network")
    print(f"  SSP1 shown; SSP3/SSP5 within ±0.001\n")
    header = f"  {'Buffer':>7} {'Prec.':>6} {'Recall':>7} {'F1':>6} {'IoU':>6} {'Kappa':>6} {'MCC':>6}   {'FoM':>6} {'QD':>7} {'AD':>7} {'TD':>7} {'OA':>6}"
    print(header)
    print("  " + "-" * len(header))
    for m in all_metrics:
        print(f"  {m['Buffer']:>7} {m['Precision']:>6.3f} {m['Recall']:>7.3f} {m['F1']:>6.3f} {m['IoU']:>6.3f} {m['Kappa']:>6.3f} {m['MCC']:>6.3f}   {m['FoM']:>6.3f} {m['QD']:>7.4f} {m['AD']:>7.4f} {m['TD']:>7.4f} {m['OA']:>6.3f}")

    csv_path = f"{OUT_DIR}/{code}_extended_metrics_20260612.csv"
    with open(csv_path, 'w', newline='') as f:
        fields = ['Buffer', 'Precision', 'Recall', 'F1', 'IoU', 'Kappa', 'MCC', 'FoM', 'QD', 'AD', 'TD', 'OA']
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_metrics)
    print(f"\n  Saved: {csv_path}")

    print(f"\n  [2] Locational accuracy analysis...")

    exact_path = f"{OUT_DIR}/{code}_spatial_TPFPFN_exact_20260607.tif"
    with rasterio.open(exact_path) as src:
        tpfpfn = src.read(1)
    pred_bin = ((tpfpfn == 1) | (tpfpfn == 2)).astype(np.uint8)
    obs_bin  = ((tpfpfn == 1) | (tpfpfn == 3)).astype(np.uint8)

    loc = locational_accuracy(pred_bin, obs_bin, ps, code)

    print(f"\n  Skeleton Distance Metrics:")
    print(f"    Mean symmetric: {loc['skel_mean_sym']:.0f} m")
    print(f"    Median symmetric: {loc['skel_med_sym']:.0f} m")
    print(f"    90th %ile (pred→obs): {loc['skel_p90_p2o']:.0f} m")
    print(f"    90th %ile (obs→pred): {loc['skel_p90_o2p']:.0f} m")
    print(f"\n    {'Buffer':>8} {'Correctness':>12} {'Completeness':>13} {'Quality':>8}")
    print(f"    {'':>8}─┼──────────────┼───────────────┼──────────")
    for r in loc['skel_table']:
        print(f"    {r['buffer_m']:>6}m | {r['correctness']:>12.3f} | {r['completeness']:>13.3f} | {r['quality']:>8.3f}")

    csv_skel = f"{OUT_DIR}/{code}_skeleton_distance_metrics_20260612.csv"
    with open(csv_skel, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=loc['skel_table'][0].keys())
        w.writeheader()
        w.writerows(loc['skel_table'])

    print(f"\n  Convergence Point Matching:")
    print(f"    Predicted: {loc['n_peaks_pred']:,}  Observed: {loc['n_peaks_obs']:,}  Ratio: {loc['peak_ratio']:.3f}")
    print(f"    Mean NN: {loc['peak_mean_nn']:.0f} m  Median NN: {loc['peak_med_nn']:.0f} m")
    if loc['int_table']:
        print(f"\n    {'Buffer':>8} {'MR':>8} {'DR':>8} {'F1':>8} {'Matched/Pred':>14} {'Matched/Obs':>14}")
        print(f"    {'':>8}─┼──────────┼──────────┼──────────┼────────────────┼────────────────")
        for r in loc['int_table']:
            print(f"    {r['buffer_m']:>6}m | {r['match_rate']:>8.3f} | {r['detect_rate']:>8.3f} | {r['f1']:>8.3f} | {r['matched_pred']:>5}/{r['n_pred']:<6} | {r['matched_obs']:>5}/{r['n_obs']:<6}")

    if loc['int_table']:
        csv_int = f"{OUT_DIR}/{code}_intersection_matching_20260612.csv"
        with open(csv_int, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=loc['int_table'][0].keys())
            w.writeheader()
            w.writerows(loc['int_table'])

    summary_path = f"{OUT_DIR}/{code}_locational_accuracy_summary_20260612.txt"
    with open(summary_path, 'w') as f:
        f.write(f"LOCATIONAL ACCURACY — {name}\n{'='*50}\n\n")
        f.write(f"Skeleton Distance:\n")
        f.write(f"  Mean symmetric: {loc['skel_mean_sym']:.0f} m\n")
        f.write(f"  Median symmetric: {loc['skel_med_sym']:.0f} m\n")
        f.write(f"  Mean correctness (pred→obs): {loc['skel_mean_p2o']:.0f} m\n")
        f.write(f"  Mean completeness (obs→pred): {loc['skel_mean_o2p']:.0f} m\n\n")
        f.write(f"Skeleton overlap:\n")
        for r in loc['skel_table']:
            f.write(f"  {r['buffer_m']:>5}m: corr={r['correctness']:.4f}  comp={r['completeness']:.4f}  qual={r['quality']:.4f}\n")
        f.write(f"\nConvergence Points:\n")
        f.write(f"  Predicted: {loc['n_peaks_pred']}  Observed: {loc['n_peaks_obs']}  Ratio: {loc['peak_ratio']}\n")
        f.write(f"  Mean NN: {loc['peak_mean_nn']:.0f} m  Median NN: {loc['peak_med_nn']:.0f} m\n\n")
        if loc['int_table']:
            for r in loc['int_table']:
                f.write(f"  {r['buffer_m']:>5}m: MR={r['match_rate']:.4f}  DR={r['detect_rate']:.4f}  F1={r['f1']:.4f}\n")

    print(f"\n  All outputs saved for {name}.")


print("\n" + "=" * 72)
print("  BANGLADESH (BGD) — Extended metrics only (locational already done)")
print("=" * 72)

all_metrics_bgd = []
for buf_label, buf_file in [('exact', 'exact'), ('300m', '300m'), ('600m', '600m'), ('900m', '900m')]:
    path = f"{OUT_DIR}/BGD_spatial_TPFPFN_{buf_file}_20260607.tif"
    with rasterio.open(path) as src:
        d = src.read(1)
    tp = int((d == 1).sum())
    fp = int((d == 2).sum())
    fn = int((d == 3).sum())
    tn = int(d.size - tp - fp - fn)
    m = compute_all_metrics(tp, fp, fn, tn)
    m['Buffer'] = buf_label
    all_metrics_bgd.append(m)

header = f"  {'Buffer':>7} {'Prec.':>6} {'Recall':>7} {'F1':>6} {'IoU':>6} {'Kappa':>6} {'MCC':>6}   {'FoM':>6} {'QD':>7} {'AD':>7} {'TD':>7} {'OA':>6}"
print(f"\n{header}")
print("  " + "-" * len(header))
for m in all_metrics_bgd:
    print(f"  {m['Buffer']:>7} {m['Precision']:>6.3f} {m['Recall']:>7.3f} {m['F1']:>6.3f} {m['IoU']:>6.3f} {m['Kappa']:>6.3f} {m['MCC']:>6.3f}   {m['FoM']:>6.3f} {m['QD']:>7.4f} {m['AD']:>7.4f} {m['TD']:>7.4f} {m['OA']:>6.3f}")

csv_bgd = f"{OUT_DIR}/BGD_extended_metrics_20260612.csv"
with open(csv_bgd, 'w', newline='') as f:
    fields = ['Buffer', 'Precision', 'Recall', 'F1', 'IoU', 'Kappa', 'MCC', 'FoM', 'QD', 'AD', 'TD', 'OA']
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    w.writerows(all_metrics_bgd)
print(f"\n  Saved: {csv_bgd}")

print("\n" + "=" * 72)
print("ALL DONE.")
print("=" * 72)
