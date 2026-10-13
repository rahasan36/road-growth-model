import os
DATA = os.environ.get("ROAD_DATA", "data")
import argparse
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


VISUALIZATION_DICT = {
    'bgd': [['Dhaka', 17], ['Chattogram', 9]],
    'nga': [['Kano Municipal', 412], ['Abuja Municipal', 11]],
    'bra': [['Manaus', 111], ['São Paulo', 5012]],
}

SCENARIOS = ['ssp1', 'ssp3', 'ssp5']
YEARS = [2030, 2040, 2050]


def compare_and_plot(observed_gdf, predicted_gdf, admin_gdf, admin_id,
                     buffer_m=50, custom_title=None, save_svg=None):
    boundary = admin_gdf.iloc[admin_id]
    boundary_geom = boundary.geometry

    obs_clip = gpd.clip(observed_gdf, boundary_geom)
    pred_clip = gpd.clip(predicted_gdf, boundary_geom)

    print(f"  Observed segments (clipped): {len(obs_clip)}")
    print(f"  Predicted segments (clipped): {len(pred_clip)}")

    if len(obs_clip) == 0 and len(pred_clip) == 0:
        print("  [SKIP] No segments in this boundary.")
        return None, None

    obs_buffer = obs_clip.geometry.buffer(buffer_m).union_all() if len(obs_clip) > 0 else None
    pred_buffer = pred_clip.geometry.buffer(buffer_m).union_all() if len(pred_clip) > 0 else None

    if obs_buffer is not None and len(pred_clip) > 0:
        consistent = pred_clip[pred_clip.intersects(obs_buffer)]
        spurious = pred_clip[~pred_clip.intersects(obs_buffer)]
    else:
        consistent = pred_clip.iloc[0:0]
        spurious = pred_clip

    if pred_buffer is not None and len(obs_clip) > 0:
        missing = obs_clip[~obs_clip.intersects(pred_buffer)]
    else:
        missing = obs_clip

    print(f"  Consistent (green): {len(consistent)}")
    print(f"  Missing (red):      {len(missing)}")
    print(f"  Spurious (orange):  {len(spurious)}")

    fig, ax = plt.subplots(figsize=(15, 12))
    gpd.GeoSeries([boundary_geom]).plot(
        ax=ax, facecolor='lightgray', edgecolor='black', linewidth=2, alpha=0.3)

    if len(missing) > 0:
        missing.plot(ax=ax, color='red', linewidth=1.2, alpha=0.8, zorder=2)
    if len(spurious) > 0:
        spurious.plot(ax=ax, color='orange', linewidth=1.2, alpha=0.8, zorder=3)
    if len(consistent) > 0:
        consistent.plot(ax=ax, color='green', linewidth=1.2, alpha=0.8, zorder=4)

    legend_lines = [
        Line2D([0], [0], color='green', linewidth=2,
               label=f'Consistent ({len(consistent)})'),
        Line2D([0], [0], color='red', linewidth=2,
               label=f'Missing ({len(missing)})'),
        Line2D([0], [0], color='orange', linewidth=2,
               label=f'Spurious ({len(spurious)})'),
    ]
    ax.legend(handles=legend_lines, loc='upper right', fontsize=11)

    minx, miny, maxx, maxy = boundary_geom.bounds
    margin = 0.05 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - margin, maxx + margin)
    ax.set_ylim(miny - margin, maxy + margin)
    ax.set_aspect('equal')

    title = custom_title or "Predicted vs Observed Road Network"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_svg is not None:
        plt.rcParams['svg.fonttype'] = 'none'
        fig.savefig(save_svg, format='svg', bbox_inches='tight', dpi=300)
        print(f"  Saved: {save_svg}")

    plt.close(fig)
    return fig, ax


def process_country(country, input_dir, buffer_m, out_dir):
    country_out_dir = os.path.join(out_dir, country)
    os.makedirs(country_out_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# COUNTRY: {country.upper()}")
    print(f"{'#'*70}")

    admin_shp = os.path.join(input_dir, f"{country}_dist.shp")
    admin_gdf = gpd.read_file(admin_shp)

    obs_shp = os.path.join(input_dir, f"{country}_roads_2015.shp")
    print(f"Observed: {obs_shp}")
    observed_gdf = gpd.read_file(obs_shp)

    if admin_gdf.crs != observed_gdf.crs:
        print(f"Reprojecting admin from {admin_gdf.crs} to {observed_gdf.crs}")
        admin_gdf = admin_gdf.to_crs(observed_gdf.crs)

    viz_list = VISUALIZATION_DICT.get(country, [])

    for year in YEARS:
        for scenario in SCENARIOS:
            pred_shp = os.path.join(
                input_dir, f"{country}_roads_{year}_{scenario}.shp")
            if not os.path.exists(pred_shp):
                print(f"[SKIP] {pred_shp} not found")
                continue

            predicted_gdf = gpd.read_file(pred_shp)
            print(f"\n{'='*70}")
            print(f"Comparing: {country.upper()} | 2015 (observed) vs {year} {scenario.upper()} (predicted)")
            print(f"{'='*70}")

            for admin_name, admin_id in viz_list:
                print(f"\n  {admin_name} (admin_id={admin_id}):")
                svg_path = os.path.join(
                    country_out_dir,
                    f"comparison_{admin_name}_{year}_{scenario}.svg")
                compare_and_plot(
                    observed_gdf, predicted_gdf, admin_gdf,
                    admin_id=admin_id,
                    buffer_m=buffer_m,
                    custom_title=(f"{admin_name}, {country.upper()} — "
                                  f"2015 vs {year} {scenario.upper()} "
                                  f"({int(buffer_m)}m buffer)"),
                    save_svg=svg_path,
                )


def main():
    parser = argparse.ArgumentParser(
        description='Compare predicted vs observed road networks')
    parser.add_argument('--country', type=str, nargs='+', default=['nga'],
                        help='Country code(s) to process (e.g. --country bgd nga bra)')
    parser.add_argument('--input_dir', type=str,
                        default=os.path.join(DATA, 'roads'),
                        help='Input directory for shapefiles')
    parser.add_argument('--buffer', type=float, default=50,
                        help='Buffer distance in metres (default: 50)')
    parser.add_argument('--out_dir', type=str, default='./output',
                        help='Output directory (default: ./output)')
    args = parser.parse_args()

    for country in args.country:
        process_country(country, args.input_dir, args.buffer, args.out_dir)


if __name__ == '__main__':
    main()
