import json
import os
import argparse
import networkx as nx
import networkit as nk
import geopandas as gpd
import momepy

from .shp_utils import (
    construct_road_graph_from_shp,
    list_admin_boundaries,
    assign_admin_to_nodes_batch,
    get_selected_admin_stats,
    calculate_fractal_dimension,
    visualize_graph_on_admin_boundary,
)

def analyze_graph_metrics(G):

    density = nx.density(G)
    print(f"Mesh Density: {density:.10f}")

    avg_clustering = nx.average_clustering(G)
    print(f"Average Clustering Coefficient: {avg_clustering:.4f}")

    transitivity = nx.transitivity(G)
    print(f"Global Clustering Coefficient (Transitivity): {transitivity:.4f}")


    if nx.is_connected(G):
        g_nk = nk.nxadapter.nx2nk(G, weightAttr='length')
        diam = nk.distance.Diameter(g_nk, algo=0)
        diam.run()
        diameter = diam.getDiameter()[0]

    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_largest = G.subgraph(largest_cc).copy()
        print("Graph is not connected, taking largest connected subgraph")
        print(f"{G.number_of_nodes()}-->{G_largest.number_of_nodes()}")
        g_nk = nk.nxadapter.nx2nk(G_largest, weightAttr='length')

        diam = nk.distance.Diameter(g_nk, algo=0)
        diam.run()
        diameter = diam.getDiameter()[0]

    print(f"Approximate Diameter: {diameter} (unit of length)")

    return density, avg_clustering, transitivity, diameter


VISUALIZATION_DICT = {'bgd': [['Dhaka', 17],
                            ['Chattogram', 9]],
                    'nga': [['Kano Municipal', 412],
                            ['Abuja Municipal', 11]],
                    'bra': [['Manaus',111],
                            ['São Paulo', 5012]],
                            }

def main():
    parser = argparse.ArgumentParser(description='Road network analysis')
    parser.add_argument('--country', type=str, default='bgd', help='Country code (default: bgd)')
    parser.add_argument('--input_dir_road', type=str, default='./data', help='Input directory for road shapefiles (default: ./data)')
    parser.add_argument('--input_dir_admin', type=str, default='./data', help='Input directory for admin shapefiles (default: ./data)')
    parser.add_argument('--out_dir', type=str, default='./output', help='Output directory (default: ./output)')
    args = parser.parse_args()

    country = args.country
    input_dir_road = args.input_dir_road
    input_dir_admin = args.input_dir_admin
    out_dir = args.out_dir

    country_out_dir = os.path.join(out_dir, country)
    os.makedirs(country_out_dir, exist_ok=True)
    print(f"Output directory created: {country_out_dir}")

    '''
    input arguments

    road_shp_file = "./data/bgd_roads.shp"
    admin_shp_file = "./data/bgd_dist.shp"
    '''
    road_shp_file = os.path.join(input_dir_road, f"{country}_roads.shp")
    admin_shp_file = os.path.join(input_dir_admin, f"{country}_dist.shp")

    print(f"Road shapefile: {road_shp_file}")
    print(f"Admin shapefile: {admin_shp_file}")


    print("\nConstructing road graph from shapefile...")

    gdf = gpd.read_file(road_shp_file)
    gdf_fixed = momepy.remove_false_nodes(gdf)
    G = momepy.gdf_to_nx(gdf_fixed, approach="primal", multigraph=False)

    G.remove_edges_from(nx.selfloop_edges(G))
    print(f"After removing self-loops: {G.number_of_edges()} edges")

    print("\nAnalyzing graph metrics...")

    density, avg_clustering, transitivity, diameter = analyze_graph_metrics(G)


    admin_gdf = list_admin_boundaries(admin_shp_file, name_field='NAME')

    G, admin_gdf = assign_admin_to_nodes_batch(G, admin_shp_file)

    df, total_inter_edges = get_selected_admin_stats(G, admin_gdf)

    print(f"Total inter-admin edges: {total_inter_edges}")

    df['intersect_density_1'] =  df['nodes_degree_gt2']/df['area_sqkm']
    print(f"Intersection density (no dead-end): {df['intersect_density_1'].mean():.4f} ± {df['intersect_density_1'].std():.4f}")

    df['intersect_density_2'] =  (df['nodes_degree_gt2']+df['nodes_degree_1'])/df['area_sqkm']
    print(f"Intersection density (with dead-end): {df['intersect_density_2'].mean():.4f} ± {df['intersect_density_2'].std():.4f}")

    df.to_csv(os.path.join(country_out_dir, 'admin_statistics.csv'), index=False)
    print(f"\nStatistics saved to 'admin_statistics.csv'")


    D, R2 = calculate_fractal_dimension(road_shp_file)
    print(f"D = {D:.4f}, R² = {R2:.4f}")


    results = {
        "input_files": {
            "road_shp_file": road_shp_file,
            "admin_shp_file": admin_shp_file
        },
        "graph_metrics": {
            "density": float(density),
            "avg_clustering": float(avg_clustering),
            "transitivity": float(transitivity),
            "diameter": int(diameter) if diameter != float('inf') else None
        },
        "total_inter_admin_edges": int(total_inter_edges),
        "intersection_density_no_dead_end": {
            "mean": float(df['intersect_density_1'].mean()),
            "std": float(df['intersect_density_1'].std())
        },
        "intersection_density_with_dead_end": {
            "mean": float(df['intersect_density_2'].mean()),
            "std": float(df['intersect_density_2'].std())
        },
        "fractal_dimension": {
            "D": float(D),
            "R2": float(R2)
        }
    }

    with open(os.path.join(country_out_dir, 'analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print("Results saved to 'analysis_results.json'")


    for admin_data in VISUALIZATION_DICT[country]:
        print(f"saving {admin_data[0]} ({country})")

        _, _, _ = visualize_graph_on_admin_boundary(
            G,
            admin_shp_file,
            admin_id=admin_data[1],
            clip_to_boundary=True,
            roads_gdf=gdf_fixed,
            custom_title=f"Road Network - {admin_data[0]}, {country.upper()}",
            save_svg=os.path.join(country_out_dir, f"{admin_data[0]}_{country.upper()}.svg")
        )

if __name__ == '__main__':
    main()
