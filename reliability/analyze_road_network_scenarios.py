import json
import os
DATA = os.environ.get("ROAD_DATA", "data")
import argparse
from collections import defaultdict
import pandas as pd
import networkx as nx
import networkit as nk
import geopandas as gpd
import momepy

from .shp_utils import (
    assign_admin_to_nodes_batch,
    get_selected_admin_stats,
    calculate_fractal_dimension,
    visualize_graph_on_admin_boundary,
)

SCENARIOS = ['ssp1', 'ssp3', 'ssp5']
YEARS = [2030, 2040, 2050]

VISUALIZATION_DICT = {
    'bgd': [['Dhaka', 17],
            ['Chattogram', 9]],
    'nga': [['Kano Municipal', 412],
            ['Abuja Municipal', 11]],
    'bra': [['Manaus', 111],
            ['São Paulo', 5012]],
}

ASPL_NODE_LIMIT = 3000


def analyze_graph_metrics(G):
    density = nx.density(G)
    print(f"Mesh Density: {density:.10f}")

    avg_clustering = nx.average_clustering(G)
    print(f"Average Clustering Coefficient: {avg_clustering:.4f}")

    transitivity = nx.transitivity(G)
    print(f"Global Clustering Coefficient (Transitivity): {transitivity:.4f}")

    if nx.is_connected(G):
        G_for_diam = G
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_for_diam = G.subgraph(largest_cc).copy()
        print("Graph is not connected, taking largest connected subgraph")
        print(f"{G.number_of_nodes()}-->{G_for_diam.number_of_nodes()}")

    g_nk = nk.nxadapter.nx2nk(G_for_diam, weightAttr='length')
    diam = nk.distance.Diameter(g_nk, algo=1)
    diam.run()
    diameter = diam.getDiameter()[0]

    print(f"Approximate Diameter: {diameter} (unit of length)")

    return density, avg_clustering, transitivity, diameter


def calculate_additional_metrics(G, total_area_sqkm=None):
    n = G.number_of_nodes()
    e = G.number_of_edges()

    if n == 0 or e == 0:
        print("[WARNING] Empty graph — skipping additional metrics.")
        return {}

    M_m = sum(d.get('length', 0) for _, _, d in G.edges(data=True))
    M = M_m / 1000

    nx2nk_map = {node: i for i, node in enumerate(G.nodes())}
    g_nk = nk.Graph(n, weighted=True)
    for u, v, d in G.edges(data=True):
        g_nk.addEdge(nx2nk_map[u], nx2nk_map[v], d.get('length', 1.0))

    degrees = [deg for _, deg in G.degree()]
    C_D_mean = sum(degrees) / n

    bc = nk.centrality.EstimateBetweenness(g_nk, nSamples=min(n, 500), normalized=False)
    bc.run()
    bc_scores = bc.scores()
    C_B_mean = sum(bc_scores) / n

    cc = nk.centrality.ApproxCloseness(g_nk, nSamples=min(n, 500))
    cc.run()
    cc_scores = cc.scores()
    C_C_mean = sum(cc_scores) / n

    sample_size = min(n, 500)
    sample_nodes = list(range(sample_size))
    eff_sum = 0.0
    for src in sample_nodes:
        sssp = nk.distance.Dijkstra(g_nk, src)
        sssp.run()
        dists = sssp.getDistances()
        eff_sum += sum(1.0 / d for d in dists if d > 0 and d < float('inf'))
    E_global = eff_sum / (sample_size * (n - 1)) if n > 1 else 0

    print(f"C_D (mean degree): {C_D_mean:.4f}")
    print(f"C_B (mean betweenness, approx): {C_B_mean:.4f}")
    print(f"C_C (mean closeness, approx): {C_C_mean:.6f}")
    print(f"E (global efficiency, approx): {E_global:.4f}")


    alpha = (e - n + 1) / (2 * n - 5) if (2 * n - 5) != 0 else None

    beta = e / n

    gamma = e / (3 * (n - 2)) if n > 2 else None

    eta = M / e

    D_net = M / total_area_sqkm if (total_area_sqkm and total_area_sqkm > 0) else None

    print(f"Alpha: {alpha:.4f}" if alpha is not None else "Alpha: N/A")
    print(f"Beta: {beta:.4f}")
    print(f"Gamma: {gamma:.4f}" if gamma is not None else "Gamma: N/A")
    print(f"Eta: {eta:.4f} km/link")
    print(f"D_net: {D_net:.4f} km/km²" if D_net is not None else "D_net: N/A (no area in admin shapefile)")

    aspl_sample_size = min(n, 500)
    aspl_sum = 0.0
    aspl_count = 0
    for src in range(aspl_sample_size):
        sssp = nk.distance.Dijkstra(g_nk, src)
        sssp.run()
        dists = sssp.getDistances()
        for d in dists:
            if 0 < d < float('inf'):
                aspl_sum += d
                aspl_count += 1
    L_avg = aspl_sum / aspl_count if aspl_count > 0 else None
    if L_avg is not None:
        print(f"Average Shortest Path Length (approx): {L_avg:.4f}")
    else:
        print("[NOTE] Could not compute ASPL — graph may be disconnected.")

    return {
        'C_D_mean': round(C_D_mean, 6),
        'C_B_mean': round(C_B_mean, 6),
        'C_C_mean': round(C_C_mean, 6),
        'E_global': round(E_global, 6),
        'alpha': round(alpha, 6) if alpha is not None else None,
        'beta': round(beta, 6),
        'gamma': round(gamma, 6) if gamma is not None else None,
        'eta': round(eta, 4),
        'D_net': round(D_net, 6) if D_net is not None else None,
        'L_avg': round(L_avg, 4) if L_avg is not None else None,
    }


def analyze_scenario(country, year, scenario, road_shp_file, admin_shp_file, country_out_dir):
    tag = f"{year}_{scenario}"
    print(f"\n{'='*70}")
    print(f"Processing: {country.upper()} | {year} | {scenario.upper()}")
    print(f"{'='*70}")
    print(f"Road shapefile: {road_shp_file}")

    if not os.path.exists(road_shp_file):
        print(f"[SKIP] Road shapefile not found: {road_shp_file}")
        return None

    gdf = gpd.read_file(road_shp_file)
    gdf_fixed = momepy.remove_false_nodes(gdf)
    G = momepy.gdf_to_nx(gdf_fixed, approach="primal", multigraph=False)
    G.remove_edges_from(nx.selfloop_edges(G))
    print(f"After removing self-loops: {G.number_of_edges()} edges")

    for _, _, data in G.edges(data=True):
        if 'geometry' in data:
            data['length'] = data['geometry'].length
        elif 'length' not in data:
            data['length'] = 0.0

    density, avg_clustering, transitivity, diameter = analyze_graph_metrics(G)

    G, admin_gdf = assign_admin_to_nodes_batch(G, admin_shp_file, road_crs=gdf.crs)

    if 'area_sqkm' not in admin_gdf.columns:
        admin_gdf = admin_gdf.copy()
        admin_gdf['area_sqkm'] = admin_gdf.geometry.area / 1e6
        print("[INFO] 'area_sqkm' not found in admin shapefile — computed from geometry.")

    df, total_inter_edges = get_selected_admin_stats(G, admin_gdf)

    print(f"Total inter-admin edges: {total_inter_edges}")

    if df['area_sqkm'].isna().any():
        df['area_sqkm'] = df['area_sqkm'].fillna(
            admin_gdf['area_sqkm'].reindex(df['admin_id']).values
        )
    df['intersect_density_1'] = df['nodes_degree_gt2'] / df['area_sqkm']
    id1_mean = df['intersect_density_1'].mean()
    id1_std  = df['intersect_density_1'].std()
    id1_var  = df['intersect_density_1'].var()
    print(f"Intersection density (no dead-end):   mean={id1_mean:.4f}, std={id1_std:.4f}, var={id1_var:.4f}")

    df['intersect_density_2'] = (df['nodes_degree_gt2'] + df['nodes_degree_1']) / df['area_sqkm']
    id2_mean = df['intersect_density_2'].mean()
    id2_std  = df['intersect_density_2'].std()
    id2_var  = df['intersect_density_2'].var()
    print(f"Intersection density (with dead-end): mean={id2_mean:.4f}, std={id2_std:.4f}, var={id2_var:.4f}")

    csv_path = os.path.join(country_out_dir, f'admin_statistics_{tag}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Statistics saved to '{csv_path}'")

    D_frac, R2 = calculate_fractal_dimension(road_shp_file)
    print(f"D = {D_frac:.4f}, R² = {R2:.4f}")

    total_area_sqkm = admin_gdf['area_sqkm'].sum()
    print(f"\nCalculating additional metrics...")
    extra = calculate_additional_metrics(G, total_area_sqkm=total_area_sqkm)

    results = {
        "input_files": {
            "road_shp_file": road_shp_file,
            "admin_shp_file": admin_shp_file
        },
        "scenario": scenario,
        "year": year,
        "graph_metrics": {
            "D_mesh": float(density),
            "C_avg": float(avg_clustering),
            "T": float(transitivity),
            "diam_G": int(diameter) if diameter != float('inf') else None,
            "C_D_mean": extra.get('C_D_mean'),
            "C_B_mean": extra.get('C_B_mean'),
            "C_C_mean": extra.get('C_C_mean'),
            "E_global": extra.get('E_global'),
            "alpha": extra.get('alpha'),
            "beta": extra.get('beta'),
            "gamma": extra.get('gamma'),
            "eta": extra.get('eta'),
            "D_net": extra.get('D_net'),
            "L_avg": extra.get('L_avg'),
        },
        "total_inter_admin_edges": int(total_inter_edges),
        "intersection_density_no_dead_end": {
            "mean": float(id1_mean),
            "std":  float(id1_std),
            "var":  float(id1_var),
        },
        "intersection_density_with_dead_end": {
            "mean": float(id2_mean),
            "std":  float(id2_std),
            "var":  float(id2_var),
        },
        "fractal_dimension": {
            "D": float(D_frac),
            "R2": float(R2)
        }
    }

    json_path = os.path.join(country_out_dir, f'analysis_results_{tag}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to '{json_path}'")

    metrics_row = {'year': year, 'scenario': scenario, **results['graph_metrics'],
                   'id1_mean': id1_mean, 'id1_std': id1_std, 'id1_var': id1_var,
                   'id2_mean': id2_mean, 'id2_std': id2_std, 'id2_var': id2_var,
                   'D_frac': D_frac, 'R2_frac': R2}
    pd.DataFrame([metrics_row]).to_csv(
        os.path.join(country_out_dir, f'analysis_results_{tag}.csv'), index=False
    )
    print(f"Analysis results CSV saved to 'analysis_results_{tag}.csv'")

    if country in VISUALIZATION_DICT:
        for admin_data in VISUALIZATION_DICT[country]:
            print(f"Saving {admin_data[0]} ({country}) [{tag}]")
            svg_path = os.path.join(country_out_dir, f"{admin_data[0]}_{country.upper()}_{tag}.svg")
            _, _, _ = visualize_graph_on_admin_boundary(
                G,
                admin_shp_file,
                admin_id=admin_data[1],
                clip_to_boundary=True,
                roads_gdf=gdf_fixed,
                custom_title=f"Road Network - {admin_data[0]}, {country.upper()} ({year} {scenario.upper()})",
                save_svg=svg_path,
                road_crs=gdf.crs
            )

    return {
        'year': year,
        'D_mesh': density,
        'C_avg': avg_clustering,
        'T': transitivity,
        'diam_G': diameter if diameter != float('inf') else None,
        **extra,
        'D_frac':   round(D_frac, 6),
        'R2_frac':  round(R2, 6),
        'id1_mean': round(id1_mean, 6),
        'id1_std':  round(id1_std, 6),
        'id1_var':  round(id1_var, 6),
        'id2_mean': round(id2_mean, 6),
        'id2_std':  round(id2_std, 6),
        'id2_var':  round(id2_var, 6),
    }


SUMMARY_METRICS = [
    'D_mesh', 'C_avg', 'T', 'diam_G',
    'C_D_mean', 'C_B_mean', 'C_C_mean', 'E_global',
    'alpha', 'beta', 'gamma', 'eta', 'D_net', 'L_avg',
    'D_frac', 'R2_frac',
    'id1_mean', 'id1_std', 'id1_var',
    'id2_mean', 'id2_std', 'id2_var',
]

METRIC_LABELS = {
    'D_mesh':   'Mesh density',
    'C_avg':    'Avg clustering coeff',
    'T':        'Transitivity',
    'diam_G':   'Network diameter',
    'C_D_mean': 'Degree centrality',
    'C_B_mean': 'Betweenness centrality',
    'C_C_mean': 'Closeness centrality',
    'E_global': 'Network efficiency',
    'alpha':    'Alpha index',
    'beta':     'Beta index',
    'gamma':    'Gamma index',
    'eta':      'Eta index',
    'D_net':    'Network density',
    'L_avg':    'Avg shortest path length',
    'D_frac':   'Fractal dimension (D)',
    'R2_frac':  'Fractal dimension (R²)',
    'id1_mean': 'Intersect density no dead-end (mean)',
    'id1_std':  'Intersect density no dead-end (std)',
    'id1_var':  'Intersect density no dead-end (var)',
    'id2_mean': 'Intersect density with dead-end (mean)',
    'id2_std':  'Intersect density with dead-end (std)',
    'id2_var':  'Intersect density with dead-end (var)',
}

BASE_YEAR   = 2015
TARGET_YEAR = 2050


def _cagr(v0, v1, n_years):
    if v0 is None or v1 is None or v0 == 0:
        return None
    return round(((v1 / v0) ** (1 / n_years) - 1) * 100, 2)


def generate_summary_table(country, base_row, scenario_metrics, country_out_dir):
    n_years = TARGET_YEAR - BASE_YEAR
    records = []

    if base_row is not None:
        base_record = {
            'country':  country.upper(),
            'year':     BASE_YEAR,
            'scenario': 'Existing',
            'row_type': 'value',
        }
        for m in SUMMARY_METRICS:
            base_record[METRIC_LABELS[m]] = base_row.get(m)
        records.append(base_record)

    for scenario in SCENARIOS:
        year_rows = scenario_metrics.get(scenario, [])
        year_map = {r['year']: r for r in year_rows}

        if TARGET_YEAR not in year_map:
            print(f"[WARNING] Missing {TARGET_YEAR} data for {scenario} — skipping summary rows.")
            continue

        if base_row is None:
            print(f"[WARNING] No 2015 baseline — CAGR cannot be computed for {scenario}.")
            continue

        target = year_map[TARGET_YEAR]

        target_record = {
            'country':  country.upper(),
            'year':     TARGET_YEAR,
            'scenario': scenario.upper(),
            'row_type': 'value',
        }
        for m in SUMMARY_METRICS:
            target_record[METRIC_LABELS[m]] = target.get(m)
        records.append(target_record)

        cagr_record = {
            'country':  country.upper(),
            'year':     TARGET_YEAR,
            'scenario': scenario.upper(),
            'row_type': f'cagr% ({BASE_YEAR}-{TARGET_YEAR})',
        }
        for m in SUMMARY_METRICS:
            cagr_record[METRIC_LABELS[m]] = _cagr(base_row.get(m), target.get(m), n_years)
        records.append(cagr_record)

    df = pd.DataFrame(records)
    csv_path = os.path.join(country_out_dir, f'summary_table_{TARGET_YEAR}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSummary table saved to '{csv_path}'")
    return df


def process_country(country, input_dir_road, input_dir_admin, out_dir):
    country_out_dir = os.path.join(out_dir, country)
    os.makedirs(country_out_dir, exist_ok=True)
    print(f"\n{'#'*70}")
    print(f"# COUNTRY: {country.upper()}")
    print(f"{'#'*70}")
    print(f"Output directory: {country_out_dir}")

    admin_shp_file = os.path.join(input_dir_admin, f"{country}_dist.shp")
    print(f"Admin shapefile: {admin_shp_file}")

    base_road_shp = os.path.join(input_dir_road, f"{country}_roads_2015.shp")
    base_row = analyze_scenario(country, 2015, 'existing', base_road_shp, admin_shp_file, country_out_dir)

    scenario_metrics = defaultdict(list)
    for year in YEARS:
        for scenario in SCENARIOS:
            road_shp_file = os.path.join(input_dir_road, f"{country}_roads_{year}_{scenario}.shp")
            row = analyze_scenario(country, year, scenario, road_shp_file, admin_shp_file, country_out_dir)
            if row is not None:
                scenario_metrics[scenario].append(row)

    CSV_COLUMNS = ['year'] + SUMMARY_METRICS
    for scenario, rows in scenario_metrics.items():
        all_rows = ([base_row] + rows) if base_row is not None else rows
        metrics_df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
        metrics_csv = os.path.join(country_out_dir, f'graph_metrics_{scenario}.csv')
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Graph metrics saved to '{metrics_csv}'")

    generate_summary_table(country, base_row, scenario_metrics, country_out_dir)


def main():
    parser = argparse.ArgumentParser(description='Road network analysis across SSP scenarios and years')
    parser.add_argument('--country', type=str, nargs='+', default=['bgd'],
                        help='Country code(s) to process (e.g. --country bgd nga bra)')
    parser.add_argument('--input_dir_road', type=str, default=os.path.join(DATA, 'roads'),
                        help='Input directory for road shapefiles')
    parser.add_argument('--input_dir_admin', type=str, default=os.path.join(DATA, 'roads'),
                        help='Input directory for admin shapefiles')
    parser.add_argument('--out_dir', type=str, default='./output',
                        help='Output directory (default: ./output)')
    args = parser.parse_args()

    for country in args.country:
        process_country(country, args.input_dir_road, args.input_dir_admin, args.out_dir)


if __name__ == '__main__':
    main()
