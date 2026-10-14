"""Assembles the model from the modules in core, with faster implementations of the slowest steps and compressed GeoTIFF output."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'core')); sys.path.insert(0, HERE)
import numpy as np
import matlab_numerics; matlab_numerics.apply()
import speedups, grown_roads_fast
import matlab_compat
def geotiffwrite_compressed(path, A, R):
    import rasterio
    A = np.asarray(A, dtype=np.float32)
    prof = dict(R["profile"]); prof.update(driver="GTiff", count=1, dtype="float32", height=A.shape[0], width=A.shape[1],
                                           compress="deflate", predictor=1, zlevel=1, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    prof["transform"] = R["transform"]; prof["crs"] = R["crs"]
    with rasterio.open(path, "w", **prof) as ds: ds.write(A, 1)
matlab_compat.geotiffwrite = geotiffwrite_compressed
matlab_compat.label2idx = speedups.label2idx_fast
matlab_compat.bwconncomp = speedups.bwconncomp_fast
import make_road_fast, make_radial_growth_fast, make_urban_clusters_fast
import run_single_scenario as rss
make_radial_growth_fast.get_radial_info = speedups.get_radial_info_fast
make_road_fast.get_upgraded_points = speedups.get_upgraded_points_fast
make_road_fast.get_grown_roads = grown_roads_fast.get_grown_roads_lean
make_road_fast.geotiffwrite = geotiffwrite_compressed
make_urban_clusters_fast.geotiffwrite = geotiffwrite_compressed
rss.make_road = make_road_fast.make_road
rss.make_urban_clusters = make_urban_clusters_fast.make_urban_clusters
run_single_scenario = rss.run_single_scenario
