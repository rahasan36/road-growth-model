import os
DATA = os.environ.get("ROAD_DATA", "data")
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize as rio_rasterize
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings, gc, os
warnings.filterwarnings('ignore')

WORK = DATA
ROADS_DIR = f"{WORK}/mnt/roads"
OUTPUT_DIR = f"{WORK}/mnt/output"

OSM_TYPE_MAP = {
    'motorway':1,'motorway_link':1,'trunk':1,'trunk_link':1,
    'primary':1,'primary_link':1,
    'secondary':2,'secondary_link':2,
    'tertiary':3,'tertiary_link':3,
    'residential':4,'unclassified':4,'living_street':4,'service':4,
}

def map_gprtp(v,n):
    if n==5: return {1:1,2:1,3:2,4:3,5:4}.get(v,0)
    return v

REF_TIF = f"{ROADS_DIR}/BGD_all_final_roads_2030_ssp1.tif"
with rasterio.open(REF_TIF) as src:
    ref_t, ref_crs = src.transform, src.crs
    ref_shape = (src.height, src.width)
    px = src.res[0]
    ref_profile = dict(src.profile)
tot = ref_shape[0]*ref_shape[1]
print(f"Grid: {ref_shape[1]}x{ref_shape[0]}, px={px:.0f}m")

print("Rasterizing OSM 2015...")
gdf = gpd.read_file(f"{WORK}/mnt/bangladesh-150101-free.shp/gis_osm_roads_free_1.shp")
gdf = gdf[gdf['fclass'].isin(OSM_TYPE_MAP.keys())].copy()
if gdf.crs != ref_crs: gdf = gdf.to_crs(ref_crs)
osm15 = rio_rasterize([(g,1) for g in gdf.geometry if g and not g.is_empty],
                       out_shape=ref_shape, transform=ref_t, fill=0, dtype=np.uint8, all_touched=False)
del gdf; gc.collect()

print("Rasterizing OSM 2026...")
gdf = gpd.read_file(f"{WORK}/mnt/bangladesh-260606-free.shp/gis_osm_roads_free_1.shp")
gdf = gdf[gdf['fclass'].isin(OSM_TYPE_MAP.keys())].copy()
if gdf.crs != ref_crs: gdf = gdf.to_crs(ref_crs)
osm26 = rio_rasterize([(g,1) for g in gdf.geometry if g and not g.is_empty],
                       out_shape=ref_shape, transform=ref_t, fill=0, dtype=np.uint8, all_touched=False)
del gdf; gc.collect()

print("Rasterizing GRIP 2015...")
gdf = gpd.read_file(f"{ROADS_DIR}/bgd_roads_2015.shp")
if gdf.crs != ref_crs: gdf = gdf.to_crs(ref_crs)
n_u = len(gdf['GP_RTP'].unique())
grip = rio_rasterize([(g,1) for g,v in zip(gdf.geometry,gdf['GP_RTP'])
                       if g and not g.is_empty and map_gprtp(int(v),n_u)>0],
                      out_shape=ref_shape, transform=ref_t, fill=0, dtype=np.uint8, all_touched=False)
del gdf; gc.collect()

print("Loading forecasts...")
with rasterio.open(f"{ROADS_DIR}/BGD_all_final_roads_2030_ssp1.tif") as ds:
    fdata = ds.read(1)
grip_fc_all = (fdata > 0).astype(np.uint8)

with rasterio.open(f"{WORK}/osm15_model_output/BGD_all_final_roads.tif") as src:
    osm15m = np.zeros(ref_shape, dtype=np.float32)
    reproject(source=src.read(1), destination=osm15m,
              src_transform=src.transform, src_crs=src.crs,
              dst_transform=ref_t, dst_crs=ref_crs, resampling=Resampling.nearest)
osm15_model_all = (osm15m != 0).astype(np.uint8)

obs_grip = ((osm26 > 0) | (grip > 0)).astype(np.uint8)
obs_osm = ((osm26 > 0) | (osm15 > 0)).astype(np.uint8)

print(f"GRIP forecast all: {np.sum(grip_fc_all):,}, obs(GRIP∪OSM26): {np.sum(obs_grip):,}")
print(f"OSM15 model all:   {np.sum(osm15_model_all):,}, obs(OSM15∪OSM26): {np.sum(obs_osm):,}")

def metrics(pred, obs, buf, px, tot):
    np_ = int(np.sum(pred)); no = int(np.sum(obs))
    if buf == 0:
        tp = int(np.sum(pred & obs)); fp = np_ - tp; fn = no - tp
    else:
        r = max(1, round(buf/px)); s = disk(r)
        tp = int(np.sum(pred & binary_dilation(obs, structure=s))); fp = np_ - tp
        fn = no - int(np.sum(obs & binary_dilation(pred, structure=s)))
    p = tp/(tp+fp) if (tp+fp)>0 else 0
    rc = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*p*rc/(p+rc) if (p+rc)>0 else 0
    return p, rc, f1, tp, fp, fn

BUFS = [0, 300, 600, 900]

grip_p, grip_r, grip_f = [], [], []
osm_p, osm_r, osm_f = [], [], []

print("\n=== ALL ROADS METRICS ===")
print(f"{'Buffer':<8} {'':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-"*50)
for buf in BUFS:
    bl = 'exact' if buf==0 else f'{buf}m'
    p1, r1, f1, tp1, fp1, fn1 = metrics(grip_fc_all, obs_grip, buf, px, tot)
    grip_p.append(p1); grip_r.append(r1); grip_f.append(f1)
    print(f"{bl:<8} GRIP    {p1:>10.3f} {r1:>10.3f} {f1:>10.3f}")

    p2, r2, f2, tp2, fp2, fn2 = metrics(osm15_model_all, obs_osm, buf, px, tot)
    osm_p.append(p2); osm_r.append(r2); osm_f.append(f2)
    print(f"{'':8} OSM15   {p2:>10.3f} {r2:>10.3f} {f2:>10.3f}")
    print()

print("Creating all-roads line chart...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

ax = axes[0]
ax.plot(BUFS, grip_p, 'o-', color='#2166ac', linewidth=2, markersize=7, label='GRIP-based forecast')
ax.plot(BUFS, osm_p, 's--', color='#b2182b', linewidth=2, markersize=7, label='OSM 2015-based model')
ax.set_xlabel('Buffer distance (m)', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('(a) Precision', fontsize=12, fontweight='bold')
ax.set_xticks(BUFS)
ax.set_ylim(0.5, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8.5, loc='lower right')

ax = axes[1]
ax.plot(BUFS, grip_r, 'o-', color='#2166ac', linewidth=2, markersize=7, label='GRIP-based forecast')
ax.plot(BUFS, osm_r, 's--', color='#b2182b', linewidth=2, markersize=7, label='OSM 2015-based model')
ax.set_xlabel('Buffer distance (m)', fontsize=11)
ax.set_ylabel('Recall', fontsize=11)
ax.set_title('(b) Recall', fontsize=12, fontweight='bold')
ax.set_xticks(BUFS)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8.5, loc='lower right')

ax = axes[2]
ax.plot(BUFS, grip_f, 'o-', color='#2166ac', linewidth=2, markersize=7, label='GRIP-based forecast')
ax.plot(BUFS, osm_f, 's--', color='#b2182b', linewidth=2, markersize=7, label='OSM 2015-based model')
ax.set_xlabel('Buffer distance (m)', fontsize=11)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('(c) F1 Score', fontsize=12, fontweight='bold')
ax.set_xticks(BUFS)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8.5, loc='lower right')

plt.tight_layout()
chart_svg = os.path.join(OUTPUT_DIR, "fig_BGD_allroads_metrics_20260608.svg")
chart_png = chart_svg.replace('.svg', '.png')
fig.savefig(chart_svg, dpi=300, bbox_inches='tight')
fig.savefig(chart_png, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {chart_svg}")

print("Creating GRIP all-roads spatial map (300m)...")
pred = grip_fc_all.astype(bool)
obs = obs_grip.astype(bool)

buf_px = max(1, round(300 / px))
s = disk(buf_px)
obs_buf = binary_dilation(obs, structure=s)
pred_buf = binary_dilation(pred, structure=s)

spatial = np.zeros(ref_shape, dtype=np.uint8)
spatial[pred & obs_buf] = 1
spatial[pred & ~obs_buf] = 2
spatial[obs & ~pred_buf] = 3

tp_px = int(np.sum(spatial == 1))
fp_px = int(np.sum(spatial == 2))
fn_px = int(np.sum(spatial == 3))
print(f"  GRIP all roads 300m: TP={tp_px:,}  FP={fp_px:,}  FN={fn_px:,}")

rows_nz, cols_nz = np.nonzero(spatial)
if len(rows_nz) > 0:
    r0, r1 = max(0, rows_nz.min()-15), min(ref_shape[0], rows_nz.max()+15)
    c0, c1 = max(0, cols_nz.min()-15), min(ref_shape[1], cols_nz.max()+15)
else:
    r0, r1, c0, c1 = 0, ref_shape[0], 0, ref_shape[1]

crop = spatial[r0:r1, c0:c1]
h, w = crop.shape

img = np.ones((h, w, 4), dtype=np.float32)
img[crop == 1] = [0.2, 0.7, 0.2, 1.0]
img[crop == 2] = [0.85, 0.15, 0.15, 1.0]
img[crop == 3] = [0.15, 0.35, 0.85, 1.0]

fig, ax = plt.subplots(1, 1, figsize=(10, 12))
ax.imshow(img, interpolation='nearest', aspect='equal')
ax.set_axis_off()
legend_elements = [
    Patch(facecolor=(0.2, 0.7, 0.2), edgecolor='none', label=f'True Positive ({tp_px:,} px)'),
    Patch(facecolor=(0.85, 0.15, 0.15), edgecolor='none', label=f'Commission / FP ({fp_px:,} px)'),
    Patch(facecolor=(0.15, 0.35, 0.85), edgecolor='none', label=f'Omission / FN ({fn_px:,} px)'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.9, edgecolor='gray')
ax.set_title('Bangladesh — GRIP-based Forecast (SSP1 2030)\nAll Roads: TP / Commission / Omission at 300 m',
             fontsize=13, fontweight='bold', pad=10)

grip_spatial_path = os.path.join(OUTPUT_DIR, "fig_BGD_allroads_spatial_GRIP_300m_20260608.png")
fig.savefig(grip_spatial_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {grip_spatial_path}")

prof = dict(ref_profile)
prof.update(driver='GTiff', count=1, dtype='uint8', height=ref_shape[0], width=ref_shape[1])
tif_path = os.path.join(OUTPUT_DIR, "BGD_allroads_TPFPFN_GRIP_300m_20260608.tif")
with rasterio.open(tif_path, 'w', **prof) as ds:
    ds.write(spatial, 1)
print(f"  Saved: {tif_path}")

print("Creating OSM15 model all-roads spatial map (300m)...")
pred2 = osm15_model_all.astype(bool)
obs2 = obs_osm.astype(bool)

obs_buf2 = binary_dilation(obs2, structure=s)
pred_buf2 = binary_dilation(pred2, structure=s)

spatial2 = np.zeros(ref_shape, dtype=np.uint8)
spatial2[pred2 & obs_buf2] = 1
spatial2[pred2 & ~obs_buf2] = 2
spatial2[obs2 & ~pred_buf2] = 3

tp2 = int(np.sum(spatial2 == 1))
fp2 = int(np.sum(spatial2 == 2))
fn2 = int(np.sum(spatial2 == 3))
print(f"  OSM15 all roads 300m: TP={tp2:,}  FP={fp2:,}  FN={fn2:,}")

crop2 = spatial2[r0:r1, c0:c1]

img2 = np.ones((h, w, 4), dtype=np.float32)
img2[crop2 == 1] = [0.2, 0.7, 0.2, 1.0]
img2[crop2 == 2] = [0.85, 0.15, 0.15, 1.0]
img2[crop2 == 3] = [0.15, 0.35, 0.85, 1.0]

fig, ax = plt.subplots(1, 1, figsize=(10, 12))
ax.imshow(img2, interpolation='nearest', aspect='equal')
ax.set_axis_off()
legend2 = [
    Patch(facecolor=(0.2, 0.7, 0.2), edgecolor='none', label=f'True Positive ({tp2:,} px)'),
    Patch(facecolor=(0.85, 0.15, 0.15), edgecolor='none', label=f'Commission / FP ({fp2:,} px)'),
    Patch(facecolor=(0.15, 0.35, 0.85), edgecolor='none', label=f'Omission / FN ({fn2:,} px)'),
]
ax.legend(handles=legend2, loc='lower left', fontsize=10, framealpha=0.9, edgecolor='gray')
ax.set_title('Bangladesh — OSM 2015-based Model (SSP1 2030)\nAll Roads: TP / Commission / Omission at 300 m',
             fontsize=13, fontweight='bold', pad=10)

osm_spatial_path = os.path.join(OUTPUT_DIR, "fig_BGD_allroads_spatial_OSM15_300m_20260608.png")
fig.savefig(osm_spatial_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {osm_spatial_path}")

tif2_path = os.path.join(OUTPUT_DIR, "BGD_allroads_TPFPFN_OSM15_300m_20260608.tif")
with rasterio.open(tif2_path, 'w', **prof) as ds:
    ds.write(spatial2, 1)
print(f"  Saved: {tif2_path}")

print("Creating side-by-side figure...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

ax1.imshow(img, interpolation='nearest', aspect='equal')
ax1.set_axis_off()
ax1.set_title('(a) GRIP-based Forecast', fontsize=13, fontweight='bold')
ax1.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.9, edgecolor='gray')

ax2.imshow(img2, interpolation='nearest', aspect='equal')
ax2.set_axis_off()
ax2.set_title('(b) OSM 2015-based Model', fontsize=13, fontweight='bold')
ax2.legend(handles=legend2, loc='lower left', fontsize=9, framealpha=0.9, edgecolor='gray')

fig.suptitle('Bangladesh SSP1 2030 — All Roads Validation at 300 m Buffer\nGreen: TP  |  Red: Commission  |  Blue: Omission',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
sidebyside_path = os.path.join(OUTPUT_DIR, "fig_BGD_allroads_spatial_comparison_300m_20260608.png")
fig.savefig(sidebyside_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {sidebyside_path}")

print("\nDone!")
