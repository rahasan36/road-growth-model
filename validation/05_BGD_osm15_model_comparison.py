import os
DATA = os.environ.get("ROAD_DATA", "data")
import os, gc, warnings
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize as rio_rasterize
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import disk
import pandas as pd

warnings.filterwarnings("ignore")

WORK = DATA
ROADS_DIR = f"{WORK}/mnt/roads"
OUTPUT_DIR = f"{WORK}/mnt/output"

OSM_2015_SHP = f"{WORK}/mnt/bangladesh-150101-free.shp/gis_osm_roads_free_1.shp"
OSM_2026_SHP = f"{WORK}/mnt/bangladesh-260606-free.shp/gis_osm_roads_free_1.shp"
GRIP_SHP = f"{ROADS_DIR}/bgd_roads_2015.shp"
REF_TIF = f"{ROADS_DIR}/BGD_all_final_roads_2030_ssp1.tif"
GRIP_FORECAST = f"{ROADS_DIR}/BGD_all_final_roads_2030_ssp1.tif"
OSM15_MODEL_OUTPUT = f"{WORK}/osm15_model_output/BGD_all_final_roads.tif"

OSM_TYPE_MAP = {
    'motorway': 1, 'motorway_link': 1, 'trunk': 1, 'trunk_link': 1,
    'primary': 1, 'primary_link': 1,
    'secondary': 2, 'secondary_link': 2,
    'tertiary': 3, 'tertiary_link': 3,
    'residential': 4, 'unclassified': 4, 'living_street': 4, 'service': 4,
}

def map_gprtp(v, n):
    if n == 5: return {1:1, 2:1, 3:2, 4:3, 5:4}.get(v, 0)
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
    n = int(np.sum(r_bin > 0))
    print(f"    {len(gdf):,} features → {n:,} px")
    del gdf, shapes; gc.collect()
    return r_bin

def rasterize_grip(ref_t, ref_crs, ref_shape):
    print("  GRIP 2015: loading...")
    gdf = gpd.read_file(GRIP_SHP)
    if gdf.crs != ref_crs: gdf = gdf.to_crs(ref_crs)
    n_u = len(gdf["GP_RTP"].unique())
    shapes = [(g, 1) for g, v in zip(gdf.geometry, gdf["GP_RTP"])
              if g and not g.is_empty and map_gprtp(int(v), n_u) > 0]
    r_bin = rio_rasterize(shapes, out_shape=ref_shape, transform=ref_t,
                          fill=0, dtype=np.uint8, all_touched=False)
    n = int(np.sum(r_bin > 0))
    print(f"    {len(gdf):,} features → {n:,} px")
    del gdf, shapes; gc.collect()
    return r_bin

def compute_metrics(pred, obs, buf, px, tot):
    np_ = int(np.sum(pred)); no = int(np.sum(obs))
    if np_ == 0 and no == 0:
        return dict(N_pred=0, N_obs=0, TP=0, FP=0, FN=0,
                    Precision=1.0, Recall=1.0, F1=1.0)
    if buf == 0:
        tp = int(np.sum(pred & obs)); fp = np_ - tp; fn = no - tp
    else:
        r = max(1, round(buf / px)); s = disk(r)
        tp = int(np.sum(pred & binary_dilation(obs, structure=s))); fp = np_ - tp
        fn = no - int(np.sum(obs & binary_dilation(pred, structure=s)))
    p = tp/(tp+fp) if (tp+fp) > 0 else 0
    r_ = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*p*r_/(p+r_) if (p+r_) > 0 else 0
    return dict(N_pred=np_, N_obs=no, TP=tp, FP=fp, FN=fn,
                Precision=round(p, 4), Recall=round(r_, 4), F1=round(f1, 4))


def main():
    with rasterio.open(REF_TIF) as src:
        ref_t, ref_crs = src.transform, src.crs
        ref_shape = (src.height, src.width)
        px = src.res[0]
    tot = ref_shape[0] * ref_shape[1]
    print(f"Grid: {ref_shape[1]}×{ref_shape[0]}, px={px:.0f}m\n")

    print("="*60)
    print("RASTERIZING BASELINES")
    print("="*60)
    osm15_bin = rasterize_osm(OSM_2015_SHP, ref_t, ref_crs, ref_shape, "OSM-2015")
    osm26_bin = rasterize_osm(OSM_2026_SHP, ref_t, ref_crs, ref_shape, "OSM-2026")
    grip_bin = rasterize_grip(ref_t, ref_crs, ref_shape)

    print("\n" + "="*60)
    print("LOADING FORECASTS")
    print("="*60)

    with rasterio.open(GRIP_FORECAST) as ds:
        grip_fdata = ds.read(1)
    grip_fc_all = (grip_fdata > 0).astype(np.uint8)
    grip_fc_new = ((grip_fdata >= 11) & (grip_fdata <= 14)).astype(np.uint8)
    print(f"  GRIP forecast: {int(np.sum(grip_fc_all)):,} all, {int(np.sum(grip_fc_new)):,} new")

    with rasterio.open(OSM15_MODEL_OUTPUT) as src:
        src_data = src.read(1)
        osm15_model = np.zeros(ref_shape, dtype=np.float32)
        reproject(source=src_data, destination=osm15_model,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_t, dst_crs=ref_crs,
                  resampling=Resampling.nearest)

    with rasterio.open(f"{WORK}/osm15_run1/BGD_road_types.tif") as src:
        src_data = src.read(1)
        osm15_model_baseline = np.zeros(ref_shape, dtype=np.float32)
        reproject(source=src_data, destination=osm15_model_baseline,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_t, dst_crs=ref_crs,
                  resampling=Resampling.nearest)

    osm15_model_all = (osm15_model != 0).astype(np.uint8)
    osm15_model_new = ((osm15_model != 0) & (osm15_model_baseline == 0)).astype(np.uint8)
    print(f"  OSM15 model:   {int(np.sum(osm15_model_all)):,} all, {int(np.sum(osm15_model_new)):,} new")

    BUFFERS = [0, 300, 600, 900]

    print("\n" + "="*60)
    print("(A) GRIP forecast vs (GRIP | OSM_2026) — your table")
    print("="*60)
    obs_grip_all = ((osm26_bin > 0) | (grip_bin > 0)).astype(np.uint8)
    obs_grip_new = ((osm26_bin > 0) & (grip_bin == 0)).astype(np.uint8)
    print(f"  Observed all: {int(np.sum(obs_grip_all)):,} | Observed new: {int(np.sum(obs_grip_new)):,}")

    print(f"\n{'Buffer':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)
    for buf in BUFFERS:
        bl = "exact" if buf == 0 else f"{buf}m"
        m = compute_metrics(grip_fc_all, obs_grip_all, buf, px, tot)
        print(f"{bl:<10} {m['Precision']:>10.3f} {m['Recall']:>10.3f} {m['F1']:>10.3f}")

    print("\n" + "="*60)
    print("(B) OSM15 model vs (OSM_2015 | OSM_2026) = OSM_2026")
    print("="*60)
    obs_osm_all = ((osm26_bin > 0) | (osm15_bin > 0)).astype(np.uint8)
    obs_osm_new = ((osm26_bin > 0) & (osm15_bin == 0)).astype(np.uint8)
    print(f"  Observed all: {int(np.sum(obs_osm_all)):,} | Observed new: {int(np.sum(obs_osm_new)):,}")

    print(f"\n{'Buffer':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)
    for buf in BUFFERS:
        bl = "exact" if buf == 0 else f"{buf}m"
        m = compute_metrics(osm15_model_all, obs_osm_all, buf, px, tot)
        print(f"{bl:<10} {m['Precision']:>10.3f} {m['Recall']:>10.3f} {m['F1']:>10.3f}")

    print("\n" + "="*60)
    print("(C) GRIP forecast vs (OSM_2015 | OSM_2026) = OSM_2026")
    print("="*60)
    print(f"\n{'Buffer':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)
    for buf in BUFFERS:
        bl = "exact" if buf == 0 else f"{buf}m"
        m = compute_metrics(grip_fc_all, obs_osm_all, buf, px, tot)
        print(f"{bl:<10} {m['Precision']:>10.3f} {m['Recall']:>10.3f} {m['F1']:>10.3f}")

    print("\n" + "="*60)
    print("NEW ROADS COMPARISON")
    print("="*60)

    print(f"\n(A) GRIP forecast new vs GRIP-new observed ({int(np.sum(obs_grip_new)):,} px):")
    print(f"{'Buffer':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)
    for buf in BUFFERS:
        bl = "exact" if buf == 0 else f"{buf}m"
        m = compute_metrics(grip_fc_new, obs_grip_new, buf, px, tot)
        print(f"{bl:<10} {m['Precision']:>10.3f} {m['Recall']:>10.3f} {m['F1']:>10.3f}")

    print(f"\n(B) OSM15 model new vs OSM-new observed ({int(np.sum(obs_osm_new)):,} px):")
    print(f"{'Buffer':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)
    for buf in BUFFERS:
        bl = "exact" if buf == 0 else f"{buf}m"
        m = compute_metrics(osm15_model_new, obs_osm_new, buf, px, tot)
        print(f"{bl:<10} {m['Precision']:>10.3f} {m['Recall']:>10.3f} {m['F1']:>10.3f}")

    print("\n" + "="*70)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*70)
    print(f"\nAll Roads:")
    print(f"{'Buffer':<8} {'GRIP→GRIP ref':>16} {'OSM15 model→OSM26':>18} {'GRIP→OSM26':>16}")
    print("-" * 60)
    for buf in BUFFERS:
        bl = "exact" if buf == 0 else f"{buf}m"
        m1 = compute_metrics(grip_fc_all, obs_grip_all, buf, px, tot)
        m2 = compute_metrics(osm15_model_all, obs_osm_all, buf, px, tot)
        m3 = compute_metrics(grip_fc_all, obs_osm_all, buf, px, tot)
        print(f"{bl:<8} {m1['F1']:>16.3f} {m2['F1']:>18.3f} {m3['F1']:>16.3f}")

    print(f"\nNew Roads:")
    print(f"{'Buffer':<8} {'GRIP new':>16} {'OSM15 model new':>18}")
    print("-" * 45)
    for buf in BUFFERS:
        bl = "exact" if buf == 0 else f"{buf}m"
        m1 = compute_metrics(grip_fc_new, obs_grip_new, buf, px, tot)
        m2 = compute_metrics(osm15_model_new, obs_osm_new, buf, px, tot)
        print(f"{bl:<8} {m1['F1']:>16.3f} {m2['F1']:>18.3f}")


if __name__ == "__main__":
    main()
