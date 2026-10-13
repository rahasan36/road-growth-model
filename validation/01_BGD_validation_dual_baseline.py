import os
DATA = os.environ.get("ROAD_DATA", "data")
import os, gc, warnings
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import disk
import pandas as pd

warnings.filterwarnings("ignore")

ROADS_DIR  = os.path.join(DATA, "roads")
OUTPUT_DIR = os.path.join(DATA, "output")
DATE       = "20260607"

OSM_2015 = os.path.join(DATA, "bangladesh-150101-free.shp/gis_osm_roads_free_1.shp")
OSM_2026 = os.path.join(DATA, "bangladesh-260606-free.shp/gis_osm_roads_free_1.shp")
GRIP_2015 = f"{ROADS_DIR}/bgd_roads_2015.shp"
REF_TIF   = f"{ROADS_DIR}/BGD_all_final_roads_2030_ssp1.tif"

FORECASTS = {
    'SSP1': f"{ROADS_DIR}/BGD_all_final_roads_2030_ssp1.tif",
    'SSP3': f"{ROADS_DIR}/BGD_all_final_roads_2030_ssp3.tif",
    'SSP5': f"{ROADS_DIR}/BGD_all_final_roads_2030_ssp5.tif",
}

BUFFERS = [0, 300, 600, 900]
TYPE_NAMES = {1: "Main", 2: "Secondary", 3: "Tertiary", 4: "Local"}

OSM_TYPE_MAP = {
    'motorway': 1, 'motorway_link': 1, 'trunk': 1, 'trunk_link': 1,
    'primary': 1, 'primary_link': 1,
    'secondary': 2, 'secondary_link': 2,
    'tertiary': 3, 'tertiary_link': 3,
    'residential': 4, 'unclassified': 4, 'living_street': 4, 'service': 4,
}

def map_gprtp(v, n):
    if n == 5: return {1:1,2:1,3:2,4:3,5:4}.get(v,0)
    return v


def rasterize_osm(path, ref_t, ref_crs, ref_shape, label):
    print(f"  {label}: loading...")
    gdf = gpd.read_file(path)
    gdf = gdf[gdf['fclass'].isin(set(OSM_TYPE_MAP.keys()))].copy()
    gdf['road_type'] = gdf['fclass'].map(OSM_TYPE_MAP)
    if gdf.crs != ref_crs: gdf = gdf.to_crs(ref_crs)

    shapes = [(g, 1) for g in gdf.geometry if g and not g.is_empty]
    r_bin = rio_rasterize(shapes, out_shape=ref_shape, transform=ref_t,
                          fill=0, dtype=np.uint8, all_touched=False)
    shapes_t = [(g, int(t)) for g, t in zip(gdf.geometry, gdf['road_type'])
                if g and not g.is_empty]
    r_typed = rio_rasterize(shapes_t, out_shape=ref_shape, transform=ref_t,
                            fill=0, dtype=np.uint8, all_touched=False)
    km = gdf.geometry.length.sum() / 1000.0
    type_km = {rt: gdf[gdf['road_type']==rt].geometry.length.sum()/1000 for rt in [1,2,3,4]}
    print(f"    {len(gdf):,} features → {int(np.sum(r_bin>0)):,} px, {km:,.0f} km")
    del gdf, shapes, shapes_t; gc.collect()
    return r_bin, r_typed, km, type_km


def rasterize_grip(ref_t, ref_crs, ref_shape):
    print("  GRIP 2015: loading...")
    gdf = gpd.read_file(GRIP_2015)
    if gdf.crs != ref_crs: gdf = gdf.to_crs(ref_crs)
    n_u = len(gdf["GP_RTP"].unique())

    shapes = [(g, 1) for g, v in zip(gdf.geometry, gdf["GP_RTP"])
              if g and not g.is_empty and map_gprtp(int(v), n_u) > 0]
    r_bin = rio_rasterize(shapes, out_shape=ref_shape, transform=ref_t,
                          fill=0, dtype=np.uint8, all_touched=False)
    shapes_t = [(g, map_gprtp(int(v), n_u)) for g, v in zip(gdf.geometry, gdf["GP_RTP"])
                if g and not g.is_empty and map_gprtp(int(v), n_u) > 0]
    r_typed = rio_rasterize(shapes_t, out_shape=ref_shape, transform=ref_t,
                            fill=0, dtype=np.uint8, all_touched=False)
    km = gdf.geometry.length.sum() / 1000.0
    type_km = {}
    for rt in [1,2,3,4]:
        vals = [v for v in range(1,6) if map_gprtp(v, n_u) == rt]
        sub = gdf[gdf['GP_RTP'].isin(vals)]
        type_km[rt] = sub.geometry.length.sum() / 1000.0
    print(f"    {len(gdf):,} features → {int(np.sum(r_bin>0)):,} px, {km:,.0f} km")
    del gdf, shapes, shapes_t; gc.collect()
    return r_bin, r_typed, km, type_km


def compute_metrics(pred, obs, buf, px, tot):
    np_ = int(np.sum(pred)); no = int(np.sum(obs))
    if np_ == 0 and no == 0:
        return dict(N_pred=0, N_obs=0, TP=0, FP=0, FN=0,
                    Precision=1.0, Recall=1.0, F1=1.0, IoU=1.0,
                    Kappa=1.0, MCC=1.0, Commission=0.0, Omission=0.0,
                    FoM=1.0, Quantity_Disagr=0.0, Allocation_Disagr=0.0,
                    Total_Disagr=0.0, Overall_Accuracy=1.0)
    if buf == 0:
        tp = int(np.sum(pred & obs)); fp = np_-tp; fn = no-tp
    else:
        r = max(1, round(buf/px)); s = disk(r)
        tp = int(np.sum(pred & binary_dilation(obs, structure=s))); fp = np_-tp
        fn = no - int(np.sum(obs & binary_dilation(pred, structure=s)))
    tn = tot-tp-fp-fn
    p = tp/(tp+fp) if (tp+fp)>0 else 0
    r_ = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*p*r_/(p+r_) if (p+r_)>0 else 0
    iou = tp/(tp+fp+fn) if (tp+fp+fn)>0 else 0
    oa = (tp+tn)/tot; pe = ((tp+fp)*(tp+fn)+(fn+tn)*(fp+tn))/(tot**2)
    kap = (oa-pe)/(1-pe) if (1-pe)>0 else 0
    d = np.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))
    mcc = (float(tp)*float(tn)-float(fp)*float(fn))/d if d>0 else 0
    fom = tp/(tp+fp+fn) if (tp+fp+fn)>0 else 1.0
    qd = abs(fp-fn)/tot
    ad = 2*min(fp,fn)/tot
    td = (fp+fn)/tot

    return dict(N_pred=np_, N_obs=no, TP=tp, FP=fp, FN=fn,
                Precision=round(p,4), Recall=round(r_,4), F1=round(f1,4),
                IoU=round(iou,4), Kappa=round(kap,4), MCC=round(mcc,4),
                Commission=round(1-p,4), Omission=round(1-r_,4),
                FoM=round(fom,4), Quantity_Disagr=round(qd,6),
                Allocation_Disagr=round(ad,6), Total_Disagr=round(td,6),
                Overall_Accuracy=round(oa,4))


def compute_distances(pred, obs, px):
    if np.sum(obs)==0 or np.sum(pred)==0:
        return {k: np.nan for k in ['mean_d_pred','med_d_pred','p90_d_pred',
                                     'mean_d_obs','med_d_obs','p90_d_obs']}
    d2o = distance_transform_edt(~obs.astype(bool))*px/1000
    pd_ = d2o[pred>0]
    d2p = distance_transform_edt(~pred.astype(bool))*px/1000
    od = d2p[obs>0]
    return dict(mean_d_pred=round(float(np.mean(pd_)),2),
                med_d_pred=round(float(np.median(pd_)),2),
                p90_d_pred=round(float(np.percentile(pd_,90)),2),
                mean_d_obs=round(float(np.mean(od)),2),
                med_d_obs=round(float(np.median(od)),2),
                p90_d_obs=round(float(np.percentile(od,90)),2))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with rasterio.open(REF_TIF) as src:
        ref_t, ref_crs = src.transform, src.crs
        ref_shape = (src.height, src.width)
        px = src.res[0]
    tot = ref_shape[0]*ref_shape[1]
    print(f"Grid: {ref_shape[1]}×{ref_shape[0]}, px={px:.0f}m\n")

    print("="*60)
    print("RASTERIZING BASELINES")
    print("="*60)
    osm15_bin, osm15_typed, osm15_km, osm15_tkm = rasterize_osm(OSM_2015, ref_t, ref_crs, ref_shape, "OSM-2015")
    grip_bin, grip_typed, grip_km, grip_tkm = rasterize_grip(ref_t, ref_crs, ref_shape)
    obs_bin, obs_typed, obs_km, obs_tkm = rasterize_osm(OSM_2026, ref_t, ref_crs, ref_shape, "OSM-2026")

    osm_new = ((obs_bin > 0) & (osm15_bin == 0)).astype(np.uint8)
    grip_new = ((obs_bin > 0) & (grip_bin == 0)).astype(np.uint8)

    print(f"\n  OSM baseline:  {int(np.sum(osm15_bin>0)):>10,} px → new in 2026: {int(np.sum(osm_new)):>10,}")
    print(f"  GRIP baseline: {int(np.sum(grip_bin>0)):>10,} px → new in 2026: {int(np.sum(grip_new)):>10,}")

    summary_rows = []
    for base_label, base_bin, base_km, base_tkm in [
        ("OSM_2015", osm15_bin, osm15_km, osm15_tkm),
        ("GRIP_2015", grip_bin, grip_km, grip_tkm)
    ]:
        n_b = int(np.sum(base_bin>0))
        new_mask = ((obs_bin>0) & (base_bin==0))
        n_new = int(np.sum(new_mask))
        overlap = int(np.sum((base_bin>0) & (obs_bin>0)))
        summary_rows.append({
            'Baseline': base_label,
            'Baseline_pixels': n_b, 'Baseline_km': round(base_km,0),
            'Observed_pixels': int(np.sum(obs_bin>0)), 'Observed_km': round(obs_km,0),
            'Overlap_pixels': overlap, 'Overlap_pct_base': round(overlap/n_b*100,1),
            'New_observed_pixels': n_new,
            'Growth_pct': round(n_new/n_b*100,1),
        })

    baselines = {
        "OSM_2015":  (osm15_bin, osm15_typed, osm_new),
        "GRIP_2015": (grip_bin, grip_typed, grip_new),
    }

    all_rows = []
    dist_rows = []

    for base_label, (base_bin_cur, base_typed_cur, new_obs) in baselines.items():
        n_new_obs = int(np.sum(new_obs))

        for ssp_label, tif_path in FORECASTS.items():
            sc = f"{ssp_label}"
            print(f"\n{'='*60}")
            print(f"  {base_label} → {sc}")
            print(f"{'='*60}")

            with rasterio.open(tif_path) as src:
                fdata = src.read(1)

            fc_new = ((fdata >= 11) & (fdata <= 14)).astype(np.uint8)
            fc_all = (fdata > 0).astype(np.uint8)
            n_fc_new = int(np.sum(fc_new))
            print(f"  Forecast new: {n_fc_new:,} | Observed new: {n_new_obs:,}")

            dm = compute_distances(fc_new, new_obs, px)
            dm.update({'Baseline': base_label, 'SSP': ssp_label,
                       'N_forecast_new': n_fc_new, 'N_observed_new': n_new_obs})
            dist_rows.append(dm)
            print(f"  Dist: mean={dm['mean_d_pred']}km  med={dm['med_d_pred']}km  P90={dm['p90_d_pred']}km")

            for buf in BUFFERS:
                bl = "exact" if buf==0 else f"{buf}m"
                m_new = compute_metrics(fc_new, new_obs, buf, px, tot)
                obs_comb = ((obs_bin>0) | (base_bin_cur>0)).astype(np.uint8)
                m_all = compute_metrics(fc_all, obs_comb, buf, px, tot)

                print(f"  {bl:>5s}: NEW F1={m_new['F1']:.3f} P={m_new['Precision']:.3f} "
                      f"R={m_new['Recall']:.3f} | ALL F1={m_all['F1']:.3f}")

                for comp, m in [('New_roads', m_new), ('All_roads', m_all)]:
                    row = {'Baseline': base_label, 'SSP': ssp_label,
                           'Comparison': comp, 'Buffer_m': buf}
                    row.update(m)
                    all_rows.append(row)

            for rt in [1,2,3,4]:
                fc_rt = (fdata == (rt+10)).astype(np.uint8)
                obs_rt = ((obs_typed == rt) & (base_bin_cur == 0)).astype(np.uint8)
                m_rt = compute_metrics(fc_rt, obs_rt, 300, px, tot)
                row = {'Baseline': base_label, 'SSP': ssp_label,
                       'Comparison': f'New_{TYPE_NAMES[rt]}', 'Buffer_m': 300}
                row.update(m_rt)
                all_rows.append(row)

            del fdata, fc_new, fc_all; gc.collect()

    df = pd.DataFrame(all_rows)
    csv = os.path.join(OUTPUT_DIR, f"validation_dual_baseline_{DATE}.csv")
    df.to_csv(csv, index=False)

    df_d = pd.DataFrame(dist_rows)
    csv_d = os.path.join(OUTPUT_DIR, f"validation_distances_dual_{DATE}.csv")
    df_d.to_csv(csv_d, index=False)

    df_s = pd.DataFrame(summary_rows)
    csv_s = os.path.join(OUTPUT_DIR, f"validation_summary_dual_{DATE}.csv")
    df_s.to_csv(csv_s, index=False)

    print(f"\n\n{'='*100}")
    print("FINAL SUMMARY — New Roads Validation")
    print(f"{'='*100}")
    df_n = df[df['Comparison']=='New_roads']
    print(f"{'Baseline':<12} {'SSP':<6} {'Buffer':>7} {'F1':>7} {'IoU':>7} {'Kappa':>7} "
          f"{'MCC':>7} {'Prec':>7} {'Rec':>7} {'Comm':>7} {'Omis':>7}")
    print("-"*90)
    for _, r in df_n.iterrows():
        b = "exact" if r['Buffer_m']==0 else f"{int(r['Buffer_m'])}m"
        print(f"{r['Baseline']:<12} {r['SSP']:<6} {b:>7} {r['F1']:>7.3f} {r['IoU']:>7.3f} "
              f"{r['Kappa']:>7.3f} {r['MCC']:>7.3f} {r['Precision']:>7.3f} "
              f"{r['Recall']:>7.3f} {r['Commission']:>7.3f} {r['Omission']:>7.3f}")

    print(f"\n{'='*100}")
    print("FINAL SUMMARY — All Roads Validation")
    print(f"{'='*100}")
    df_a = df[df['Comparison']=='All_roads']
    for _, r in df_a.iterrows():
        b = "exact" if r['Buffer_m']==0 else f"{int(r['Buffer_m'])}m"
        print(f"{r['Baseline']:<12} {r['SSP']:<6} {b:>7} F1={r['F1']:.3f} IoU={r['IoU']:.3f} "
              f"Kappa={r['Kappa']:.3f} P={r['Precision']:.3f} R={r['Recall']:.3f}")

    print(f"\nCSV: {csv}")
    print(f"Distances: {csv_d}")
    print(f"Summary: {csv_s}")
    return df


if __name__ == "__main__":
    df = main()
