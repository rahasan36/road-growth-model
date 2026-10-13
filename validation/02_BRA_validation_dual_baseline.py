import os
DATA = os.environ.get("ROAD_DATA", "data")
import os, gc, warnings
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
import pandas as pd

warnings.filterwarnings("ignore")

ROADS_DIR  = os.path.join(DATA, "roads")
OUTPUT_DIR = os.path.join(DATA, "output")
BRA_OSM    = os.path.join(DATA, "BrazilRoads")
DATE       = "20260607"
COUNTRY    = "BRA"

OSM_2015 = f"{BRA_OSM}/2015/gis_osm_roads_free_1.shp"
GRIP_2015 = f"{ROADS_DIR}/bra_roads_2015.shp"
REF_TIF   = f"{ROADS_DIR}/bra_all_final_roads_2030_ssp1.tif"

OSM_2026_REGIONS = [
    f"{BRA_OSM}/2026/norte.gpkg",
    f"{BRA_OSM}/2026/centro-oeste.gpkg",
    f"{BRA_OSM}/2026/nordeste.gpkg",
    f"{BRA_OSM}/2026/sul.gpkg",
    f"{BRA_OSM}/2026/sudeste.gpkg",
]

FORECASTS = {
    'SSP1': f"{ROADS_DIR}/bra_all_final_roads_2030_ssp1.tif",
    'SSP3': f"{ROADS_DIR}/bra_all_final_roads_2030_ssp3.tif",
    'SSP5': f"{ROADS_DIR}/bra_all_final_roads_2030_ssp5.tif",
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


def rasterize_gpkg_chunked(gpkg_path, layer, ref_t, ref_crs, ref_shape, r_bin, r_typed, chunk_read=150000):
    region = os.path.basename(gpkg_path).replace('.gpkg', '')

    attrs = gpd.read_file(gpkg_path, layer=layer, ignore_geometry=True)
    total = len(attrs)
    valid_fclass = set(OSM_TYPE_MAP.keys())
    del attrs; gc.collect()
    print(f"  [{region}]: {total:,} total features", flush=True)

    for start in range(0, total, chunk_read):
        end = min(start + chunk_read, total)
        gdf = gpd.read_file(gpkg_path, layer=layer, rows=slice(start, end))
        gdf = gdf[gdf['fclass'].isin(valid_fclass)].copy()
        if len(gdf) == 0:
            del gdf; gc.collect(); continue
        gdf['road_type'] = gdf['fclass'].map(OSM_TYPE_MAP)
        if gdf.crs != ref_crs:
            gdf = gdf.to_crs(ref_crs)

        shapes = [(g, 1) for g in gdf.geometry if g and not g.is_empty]
        if shapes:
            tmp = rio_rasterize(shapes, out_shape=ref_shape, transform=ref_t,
                                fill=0, dtype=np.uint8, all_touched=False)
            np.maximum(r_bin, tmp, out=r_bin); del tmp
        del shapes

        shapes_t = [(g, int(t)) for g, t in zip(gdf.geometry, gdf['road_type'])
                    if g and not g.is_empty]
        if shapes_t:
            tmp = rio_rasterize(shapes_t, out_shape=ref_shape, transform=ref_t,
                                fill=0, dtype=np.uint8, all_touched=False)
            np.maximum(r_typed, tmp, out=r_typed); del tmp
        del gdf, shapes_t; gc.collect()
        print(f"    {start:,}-{end:,} done", flush=True)


def rasterize_osm_2026(ref_t, ref_crs, ref_shape):
    r_bin = np.zeros(ref_shape, dtype=np.uint8)
    r_typed = np.zeros(ref_shape, dtype=np.uint8)

    for gpkg_path in OSM_2026_REGIONS:
        rasterize_gpkg_chunked(gpkg_path, 'gis_osm_roads_free',
                               ref_t, ref_crs, ref_shape, r_bin, r_typed)

    n = int(np.sum(r_bin > 0))
    print(f"  OSM-2026 total: {n:,} px", flush=True)
    return r_bin, r_typed


def rasterize_osm_2015(ref_t, ref_crs, ref_shape):
    print("  OSM-2015: counting...", flush=True)
    attrs = gpd.read_file(OSM_2015, ignore_geometry=True)
    total = len(attrs)
    del attrs; gc.collect()
    print(f"    {total:,} total features", flush=True)

    r_bin = np.zeros(ref_shape, dtype=np.uint8)
    r_typed = np.zeros(ref_shape, dtype=np.uint8)
    valid_fclass = set(OSM_TYPE_MAP.keys())

    chunk = 150000
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        gdf = gpd.read_file(OSM_2015, rows=slice(start, end))
        gdf = gdf[gdf['fclass'].isin(valid_fclass)].copy()
        if len(gdf) == 0:
            del gdf; gc.collect(); continue
        gdf['road_type'] = gdf['fclass'].map(OSM_TYPE_MAP)
        if gdf.crs != ref_crs:
            gdf = gdf.to_crs(ref_crs)

        shapes = [(g, 1) for g in gdf.geometry if g and not g.is_empty]
        if shapes:
            tmp = rio_rasterize(shapes, out_shape=ref_shape, transform=ref_t,
                                fill=0, dtype=np.uint8, all_touched=False)
            np.maximum(r_bin, tmp, out=r_bin); del tmp
        del shapes

        shapes_t = [(g, int(t)) for g, t in zip(gdf.geometry, gdf['road_type'])
                    if g and not g.is_empty]
        if shapes_t:
            tmp = rio_rasterize(shapes_t, out_shape=ref_shape, transform=ref_t,
                                fill=0, dtype=np.uint8, all_touched=False)
            np.maximum(r_typed, tmp, out=r_typed); del tmp
        del gdf, shapes_t; gc.collect()
        print(f"    {start:,}-{end:,} done", flush=True)

    n = int(np.sum(r_bin > 0))
    print(f"  OSM-2015 total: {n:,} px", flush=True)
    return r_bin, r_typed


def rasterize_grip(ref_t, ref_crs, ref_shape):
    print("  GRIP 2015: loading...", flush=True)
    gdf = gpd.read_file(GRIP_2015)
    if gdf.crs != ref_crs: gdf = gdf.to_crs(ref_crs)
    n_u = len(gdf["GP_RTP"].unique())
    nf = len(gdf)
    print(f"    {nf:,} features, GP_RTP has {n_u} unique values", flush=True)

    r_bin = np.zeros(ref_shape, dtype=np.uint8)
    r_typed = np.zeros(ref_shape, dtype=np.uint8)

    chunk = 200000
    for start in range(0, nf, chunk):
        end = min(start + chunk, nf)
        sub = gdf.iloc[start:end]
        shapes = [(g, 1) for g, v in zip(sub.geometry, sub["GP_RTP"])
                  if g and not g.is_empty and map_gprtp(int(v), n_u) > 0]
        if shapes:
            tmp = rio_rasterize(shapes, out_shape=ref_shape, transform=ref_t,
                                fill=0, dtype=np.uint8, all_touched=False)
            r_bin = np.maximum(r_bin, tmp); del tmp
        del shapes
        shapes_t = [(g, map_gprtp(int(v), n_u)) for g, v in zip(sub.geometry, sub["GP_RTP"])
                    if g and not g.is_empty and map_gprtp(int(v), n_u) > 0]
        if shapes_t:
            tmp = rio_rasterize(shapes_t, out_shape=ref_shape, transform=ref_t,
                                fill=0, dtype=np.uint8, all_touched=False)
            r_typed = np.maximum(r_typed, tmp); del tmp
        del sub, shapes_t; gc.collect()

    del gdf; gc.collect()
    n = int(np.sum(r_bin > 0))
    print(f"  GRIP total: {n:,} px", flush=True)
    return r_bin, r_typed


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
        obs_dil = binary_dilation(obs, structure=s)
        tp = int(np.sum(pred & obs_dil)); fp = np_-tp
        del obs_dil; gc.collect()
        pred_dil = binary_dilation(pred, structure=s)
        fn = no - int(np.sum(obs & pred_dil))
        del pred_dil; gc.collect()
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
    qd = abs(fp-fn)/tot; ad = 2*min(fp,fn)/tot; td = (fp+fn)/tot
    return dict(N_pred=np_, N_obs=no, TP=tp, FP=fp, FN=fn,
                Precision=round(p,4), Recall=round(r_,4), F1=round(f1,4),
                IoU=round(iou,4), Kappa=round(kap,4), MCC=round(mcc,4),
                Commission=round(1-p,4), Omission=round(1-r_,4),
                FoM=round(fom,4), Quantity_Disagr=round(qd,6),
                Allocation_Disagr=round(ad,6), Total_Disagr=round(td,6),
                Overall_Accuracy=round(oa,4))


def run_baseline(base_label, base_bin, base_typed, obs_bin, obs_typed, px, tot):
    new_obs = ((obs_bin > 0) & (base_bin == 0)).astype(np.uint8)
    n_new = int(np.sum(new_obs))
    print(f"\n  {base_label}: new observed = {n_new:,} px", flush=True)
    all_rows = []

    for ssp, tif in FORECASTS.items():
        print(f"\n  --- {base_label} → {ssp} ---", flush=True)
        with rasterio.open(tif) as src:
            fdata = src.read(1)
        fc_new = ((fdata >= 11) & (fdata <= 14)).astype(np.uint8)
        fc_all = (fdata > 0).astype(np.uint8)
        print(f"  Forecast new: {int(np.sum(fc_new)):,} | Obs new: {n_new:,}", flush=True)

        for buf in BUFFERS:
            bl = "exact" if buf==0 else f"{buf}m"
            m_new = compute_metrics(fc_new, new_obs, buf, px, tot)
            obs_comb = ((obs_bin>0) | (base_bin>0)).astype(np.uint8)
            m_all = compute_metrics(fc_all, obs_comb, buf, px, tot)
            del obs_comb; gc.collect()
            print(f"  {bl:>5s}: NEW F1={m_new['F1']:.3f} P={m_new['Precision']:.3f} "
                  f"R={m_new['Recall']:.3f} | ALL F1={m_all['F1']:.3f}", flush=True)
            for comp, m in [('New_roads', m_new), ('All_roads', m_all)]:
                row = {'Country': COUNTRY, 'Baseline': base_label, 'SSP': ssp,
                       'Comparison': comp, 'Buffer_m': buf}
                row.update(m); all_rows.append(row)

        for rt in [1,2,3,4]:
            fc_rt = (fdata == (rt+10)).astype(np.uint8)
            obs_rt = ((obs_typed == rt) & (base_bin == 0)).astype(np.uint8)
            m_rt = compute_metrics(fc_rt, obs_rt, 300, px, tot)
            row = {'Country': COUNTRY, 'Baseline': base_label, 'SSP': ssp,
                   'Comparison': f'New_{TYPE_NAMES[rt]}', 'Buffer_m': 300}
            row.update(m_rt); all_rows.append(row)
            del fc_rt, obs_rt

        del fdata, fc_new, fc_all; gc.collect()

    del new_obs; gc.collect()
    return all_rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with rasterio.open(REF_TIF) as src:
        ref_t, ref_crs = src.transform, src.crs
        ref_shape = (src.height, src.width)
        px = src.res[0]
    tot = ref_shape[0]*ref_shape[1]
    print(f"Grid: {ref_shape[1]}×{ref_shape[0]} = {tot:,} px, px={px:.0f}m\n", flush=True)

    obs_bin, obs_typed = rasterize_osm_2026(ref_t, ref_crs, ref_shape)
    gc.collect()

    all_rows = []

    print("\n" + "="*60 + "\nGRIP 2015 BASELINE\n" + "="*60, flush=True)
    grip_bin, grip_typed = rasterize_grip(ref_t, ref_crs, ref_shape)
    rows = run_baseline("GRIP_2015", grip_bin, grip_typed, obs_bin, obs_typed, px, tot)
    all_rows.extend(rows)
    del grip_bin, grip_typed; gc.collect()

    print("\n" + "="*60 + "\nOSM 2015 BASELINE\n" + "="*60, flush=True)
    osm15_bin, osm15_typed = rasterize_osm_2015(ref_t, ref_crs, ref_shape)
    rows = run_baseline("OSM_2015", osm15_bin, osm15_typed, obs_bin, obs_typed, px, tot)
    all_rows.extend(rows)
    del osm15_bin, osm15_typed, obs_bin, obs_typed; gc.collect()

    df = pd.DataFrame(all_rows)
    csv = os.path.join(OUTPUT_DIR, f"validation_{COUNTRY}_dual_{DATE}.csv")
    df.to_csv(csv, index=False)

    for comp in ['All_roads', 'New_roads']:
        sub = df[df['Comparison']==comp]
        print(f"\n--- {COUNTRY} {comp} ---")
        print(f"{'Base':<12} {'SSP':<6} {'Buf':>5} {'F1':>7} {'FoM':>7} {'Prec':>7} {'Rec':>7}")
        for _, r in sub.iterrows():
            b = "exact" if r['Buffer_m']==0 else f"{int(r['Buffer_m'])}m"
            print(f"{r['Baseline']:<12} {r['SSP']:<6} {b:>5} {r['F1']:>7.3f} {r['FoM']:>7.3f} "
                  f"{r['Precision']:>7.3f} {r['Recall']:>7.3f}")

    print(f"\nCSV: {csv}")
    return df

if __name__ == "__main__":
    df = main()
