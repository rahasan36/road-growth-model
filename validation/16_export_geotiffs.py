import os
DATA = os.environ.get("ROAD_DATA", "data")
import numpy as np, rasterio, gc, os, sys
OUT_DIR = os.path.join(DATA, "output")
EXPORT_DIR = os.path.join(DATA, "RoadModel_SeeForData/geospatial_exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

code = sys.argv[1]
name = {'BGD':'Bangladesh','BRA':'Brazil','NGA':'Nigeria'}[code]
print(f"Exporting {name} ({code})...")

tpfpfn_path = f"{OUT_DIR}/{code}_spatial_TPFPFN_exact_20260607.tif"
with rasterio.open(tpfpfn_path) as src:
    d = src.read(1)
    profile = src.profile.copy()
    print(f"  Shape: {d.shape}, CRS: {src.crs}")

pred = ((d==1)|(d==2)).astype(np.uint8)
obs  = ((d==1)|(d==3)).astype(np.uint8)
print(f"  Pred: {pred.sum():,}  Obs: {obs.sum():,}")

profile.update(dtype='uint8', count=1, compress='lzw', nodata=255)

pred_path = f"{EXPORT_DIR}/{code}_predicted_roads_2030SSP1.tif"
with rasterio.open(pred_path, 'w', **profile) as dst:
    dst.write(pred, 1)
print(f"  {pred_path}")
del pred; gc.collect()

obs_path = f"{EXPORT_DIR}/{code}_observed_roads_OSM2026.tif"
with rasterio.open(obs_path, 'w', **profile) as dst:
    dst.write(obs, 1)
print(f"  {obs_path}")
del obs; gc.collect()

tpfpfn_export = f"{EXPORT_DIR}/{code}_TPFPFN_exact.tif"
profile.update(dtype=d.dtype, nodata=0)
with rasterio.open(tpfpfn_export, 'w', **profile) as dst:
    dst.write(d, 1)
print(f"  {tpfpfn_export}")
del d; gc.collect()
print(f"  Done: {name}")
