import os
DATA = os.environ.get("ROAD_DATA", "data")
import argparse
import numpy as np
import rasterio
from rasterio.transform import xy
from skimage.morphology import skeletonize
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OBSERVED_DIR = os.path.join(DATA, "roads")
RASTER_BASE = os.path.join(DATA, "forecast_runs")

COUNTRIES = ['bgd', 'nga']
SCENARIOS = ['ssp1', 'ssp3', 'ssp5']
SSP_MAP = {'ssp1': 1, 'ssp3': 3, 'ssp5': 5}
YEARS = [2030, 2040, 2050]

VISUALIZATION_DICT = {
    'bgd': [['Dhaka', 17], ['Chattogram', 9]],
    'nga': [['Kano Municipal', 412], ['Abuja Municipal', 11]],
}


def find_raster_path(base, country, year, ssp_num):
    cu = country.upper()
    candidates = [
        os.path.join(base, f"{year} New 08282024", f"{cu} - {year} New",
                     f"scenario_{ssp_num}", f"{cu}_all_final_roads.tif"),
        os.path.join(base, f"{year} New 08282024", f"{cu} - {year}",
                     f"scenario_{ssp_num}", f"{cu}_all_final_roads.tif"),
        os.path.join(base, f"{year} New 08152024", f"{cu}-{year}new",
                     f"scenario_{ssp_num}", f"{cu}_all_final_roads.tif"),
        os.path.join(base, f"{year} New 08152024", f"{cu}-{year}",
                     f"scenario_{ssp_num}", f"{cu}_all_final_roads.tif"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def raster_to_lines(tif_path):
    with rasterio.open(tif_path) as src:
        band = src.read(1)
        transform = src.transform
        crs = src.crs

    if crs is None or 'Mollweide' in (crs.to_wkt() if crs else ''):
        from pyproj import CRS as ProjCRS
        crs = ProjCRS.from_authority('ESRI', 54009)

    road_mask = (band != 0).astype(np.uint8)
    print(f"  Road pixels: {road_mask.sum():,}")

    skeleton = skeletonize(road_mask).astype(np.uint8)
    skel_count = skeleton.sum()
    print(f"  Skeleton pixels: {skel_count:,}")

    rows, cols = np.where(skeleton == 1)
    if len(rows) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    pixel_set = set(zip(rows.tolist(), cols.tolist()))

    segments = []
    offsets = [(0, 1), (1, 0), (1, 1), (-1, 1)]

    for r, c in pixel_set:
        x1, y1 = xy(transform, r, c)
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if (nr, nc) in pixel_set:
                x2, y2 = xy(transform, nr, nc)
                segments.append(LineString([(x1, y1), (x2, y2)]))

    print(f"  Line segments: {len(segments):,}")
    return gpd.GeoDataFrame(geometry=segments, crs=crs)


def compare_and_plot(observed_gdf, predicted_gdf, boundary_geom,
                     buffer_m=50, custom_title=None, save_svg=None):

    obs_clip = gpd.clip(observed_gdf, boundary_geom)
    pred_clip = gpd.clip(predicted_gdf, boundary_geom)

    print(f"  Observed (clipped): {len(obs_clip):,}")
    print(f"  Predicted (clipped): {len(pred_clip):,}")

    if len(obs_clip) == 0 and len(pred_clip) == 0:
        print("  [SKIP] No segments in this boundary.")
        return

    obs_buffer = (obs_clip.geometry.buffer(buffer_m).union_all()
                  if len(obs_clip) > 0 else None)
    pred_buffer = (pred_clip.geometry.buffer(buffer_m).union_all()
                   if len(pred_clip) > 0 else None)

    if obs_buffer is not None and len(pred_clip) > 0:
        mask = pred_clip.intersects(obs_buffer)
        consistent = pred_clip[mask]
        spurious = pred_clip[~mask]
    else:
        consistent = pred_clip.iloc[0:0]
        spurious = pred_clip

    if pred_buffer is not None and len(obs_clip) > 0:
        missing = obs_clip[~obs_clip.intersects(pred_buffer)]
    else:
        missing = obs_clip

    n_con, n_mis, n_spu = len(consistent), len(missing), len(spurious)
    print(f"  Consistent (green): {n_con:,}")
    print(f"  Missing (red):      {n_mis:,}")
    print(f"  Spurious (orange):  {n_spu:,}")

    fig, ax = plt.subplots(figsize=(15, 12))
    gpd.GeoSeries([boundary_geom]).plot(
        ax=ax, facecolor='lightgray', edgecolor='black', linewidth=2, alpha=0.3)

    if n_mis > 0:
        missing.plot(ax=ax, color='red', linewidth=1.2, alpha=0.8, zorder=2)
    if n_spu > 0:
        spurious.plot(ax=ax, color='orange', linewidth=1.2, alpha=0.8, zorder=3)
    if n_con > 0:
        consistent.plot(ax=ax, color='green', linewidth=1.2, alpha=0.8, zorder=4)

    legend_lines = [
        Line2D([0], [0], color='green', lw=2, label=f'Consistent ({n_con:,})'),
        Line2D([0], [0], color='red', lw=2, label=f'Missing ({n_mis:,})'),
        Line2D([0], [0], color='orange', lw=2, label=f'Spurious ({n_spu:,})'),
    ]
    ax.legend(handles=legend_lines, loc='upper right', fontsize=11)

    minx, miny, maxx, maxy = boundary_geom.bounds
    margin = 0.05 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - margin, maxx + margin)
    ax.set_ylim(miny - margin, maxy + margin)
    ax.set_aspect('equal')

    ax.set_title(custom_title or "Predicted vs Observed", fontsize=14,
                 fontweight='bold')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_svg:
        plt.rcParams['svg.fonttype'] = 'none'
        fig.savefig(save_svg, format='svg', bbox_inches='tight', dpi=300)
        print(f"  Saved: {save_svg}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Compare predicted road rasters vs observed shapefiles')
    parser.add_argument('--country', type=str, nargs='+', default=COUNTRIES,
                        help='Country code(s) (default: bgd nga)')
    parser.add_argument('--observed_dir', type=str, default=OBSERVED_DIR,
                        help='Directory with observed road shapefiles and admin shapefiles')
    parser.add_argument('--raster_base', type=str, default=RASTER_BASE,
                        help='Base directory for predicted rasters')
    parser.add_argument('--buffer', type=float, default=50,
                        help='Buffer distance in metres (default: 50)')
    parser.add_argument('--out_dir', type=str, default='./output',
                        help='Output directory')
    args = parser.parse_args()

    for country in args.country:
        cu = country.upper()
        country_out = os.path.join(args.out_dir, country, 'raster_comparison')
        os.makedirs(country_out, exist_ok=True)

        obs_shp = os.path.join(args.observed_dir, f"{country}_roads_2015.shp")
        if not os.path.exists(obs_shp):
            print(f"[SKIP] Observed shapefile not found: {obs_shp}")
            continue
        print(f"\nObserved: {obs_shp}")
        observed_gdf = gpd.read_file(obs_shp)
        road_crs = observed_gdf.crs

        admin_shp = os.path.join(args.observed_dir, f"{country}_dist.shp")
        admin_gdf = gpd.read_file(admin_shp)
        if admin_gdf.crs != road_crs:
            print(f"Reprojecting admin from {admin_gdf.crs} to {road_crs}")
            admin_gdf = admin_gdf.to_crs(road_crs)

        viz_list = VISUALIZATION_DICT.get(country, [])
        if not viz_list:
            print(f"[SKIP] No visualization boundaries defined for {cu}")
            continue

        for year in YEARS:
            for scenario in SCENARIOS:
                ssp_num = SSP_MAP[scenario]
                raster_path = find_raster_path(args.raster_base, country,
                                               year, ssp_num)
                if raster_path is None:
                    print(f"\n[SKIP] Raster not found: {cu} {year} {scenario}")
                    continue

                print(f"\n{'='*70}")
                print(f"{cu} | 2015 observed vs {year} {scenario.upper()} (raster)")
                print(f"{'='*70}")
                print(f"Raster: {raster_path}")

                predicted_gdf = raster_to_lines(raster_path)

                if predicted_gdf.crs != road_crs:
                    print(f"  Reprojecting predicted from {predicted_gdf.crs} "
                          f"to {road_crs}")
                    predicted_gdf = predicted_gdf.to_crs(road_crs)

                for admin_name, admin_id in viz_list:
                    print(f"\n  {admin_name} (admin_id={admin_id}):")
                    boundary_geom = admin_gdf.iloc[admin_id].geometry

                    svg_path = os.path.join(
                        country_out,
                        f"raster_cmp_{admin_name}_{year}_{scenario}.svg")
                    compare_and_plot(
                        observed_gdf, predicted_gdf, boundary_geom,
                        buffer_m=args.buffer,
                        custom_title=(f"{admin_name}, {cu} — "
                                      f"2015 observed vs {year} "
                                      f"{scenario.upper()} raster "
                                      f"({int(args.buffer)}m buffer)"),
                        save_svg=svg_path,
                    )


if __name__ == '__main__':
    main()
