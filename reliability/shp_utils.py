import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.geometry import Point, LineString, box as Box
import numpy as np
import time
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def construct_road_graph_from_shp(shp_file_path):

    gdf = gpd.read_file(shp_file_path)

    G = nx.Graph()

    node_dict = {}
    node_id = 0

    def get_or_create_node(coord):
        nonlocal node_id
        coord_key = (round(coord[0], 6), round(coord[1], 6))

        if coord_key not in node_dict:
            node_dict[coord_key] = node_id
            G.add_node(node_id, pos=coord_key, x=coord[0], y=coord[1])
            node_id += 1

        return node_dict[coord_key]

    for idx, row in gdf.iterrows():
        geometry = row.geometry

        if geometry.geom_type == 'LineString':
            coords = list(geometry.coords)

            start_node = get_or_create_node(coords[0])
            end_node = get_or_create_node(coords[-1])

            length = geometry.length

            edge_attrs = {
                'length': length,
                'geometry': geometry,
                'edge_id': idx
            }

            for col in gdf.columns:
                if col != 'geometry':
                    edge_attrs[col] = row[col]

            G.add_edge(start_node, end_node, **edge_attrs)

        elif geometry.geom_type == 'MultiLineString':
            for line in geometry.geoms:
                coords = list(line.coords)
                start_node = get_or_create_node(coords[0])
                end_node = get_or_create_node(coords[-1])

                length = line.length
                edge_attrs = {
                    'length': length,
                    'geometry': line,
                    'edge_id': idx
                }

                for col in gdf.columns:
                    if col != 'geometry':
                        edge_attrs[col] = row[col]

                G.add_edge(start_node, end_node, **edge_attrs)

    print(f"Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    return G, gdf


def list_admin_boundaries(admin_shp_path, name_field=None):
    admin_gdf = gpd.read_file(admin_shp_path)

    print(f"\nTotal boundaries: {len(admin_gdf)}")
    print("\nAvailable columns:", list(admin_gdf.columns))


    return admin_gdf


def assign_admin_to_nodes_batch(G, admin_shp_path, batch_size=10000, road_crs=None):


    first_node = next(iter(G.nodes), None)
    use_pos = first_node is not None and 'pos' in G.nodes[first_node]

    start_time = time.time()

    admin_gdf = gpd.read_file(admin_shp_path)

    if road_crs is not None and admin_gdf.crs != road_crs:
        print(f"Reprojecting admin boundaries from {admin_gdf.crs} to {road_crs}")
        admin_gdf = admin_gdf.to_crs(road_crs)

    print(f"Assigning admin IDs to {G.number_of_nodes()} nodes in batches...")

    all_nodes = list(G.nodes())
    nodes_without_admin = 0

    for i in range(0, len(all_nodes), batch_size):
        batch_nodes = all_nodes[i:i+batch_size]

        node_data = []
        node_ids = []

        for node in batch_nodes:
            node_pos = G.nodes[node]['pos'] if use_pos else node
            node_data.append({'geometry': Point(node_pos)})
            node_ids.append(node)

        nodes_gdf = gpd.GeoDataFrame(node_data, crs=admin_gdf.crs)
        nodes_gdf['node_id'] = node_ids

        joined = gpd.sjoin(nodes_gdf, admin_gdf, how='left', predicate='intersects')
        node_admin_mapping = joined.groupby('node_id')['index_right'].apply(list).to_dict()

        for node in batch_nodes:
            admin_ids = node_admin_mapping.get(node, [])
            admin_ids = [int(aid) for aid in admin_ids if pd.notna(aid)]

            G.nodes[node]['admin_ids'] = admin_ids

            if len(admin_ids) == 0:
                nodes_without_admin += 1

        if (i + batch_size) % 50000 == 0:
            print(f"  Processed {min(i + batch_size, len(all_nodes))}/{len(all_nodes)} nodes...")

    elapsed_time = time.time() - start_time

    print(f"Nodes assigned to admin boundaries: {G.number_of_nodes() - nodes_without_admin}")
    print(f"Nodes without admin assignment: {nodes_without_admin}")
    print(f"Time taken: {elapsed_time:.2f} seconds")

    return G, admin_gdf

def get_selected_admin_stats(G, admin_gdf):


    total_inter_admin_edges = 0
    for u, v in G.edges():
        u_admins = set(G.nodes[u].get('admin_ids', []))
        v_admins = set(G.nodes[v].get('admin_ids', []))

        if len(u_admins) == 0 or len(v_admins) == 0:
            continue

        if len(u_admins.intersection(v_admins)) == 0:
            total_inter_admin_edges += 1

    admin_stats = []

    for admin_idx, admin_row in admin_gdf.iterrows():
        nodes_in_admin = [n for n in G.nodes()
                         if admin_idx in G.nodes[n].get('admin_ids', [])]

        inter_edges = 0
        for u, v in G.edges():
            u_admins = set(G.nodes[u].get('admin_ids', []))
            v_admins = set(G.nodes[v].get('admin_ids', []))

            if admin_idx in u_admins or admin_idx in v_admins:
                if admin_idx in u_admins and admin_idx not in v_admins:
                    inter_edges += 1
                elif admin_idx in v_admins and admin_idx not in u_admins:
                    inter_edges += 1

        nodes_degree_1 = 0
        nodes_degree_2 = 0
        nodes_degree_gt2 = 0

        for node in nodes_in_admin:
            degree = G.degree(node)
            if degree == 1:
                nodes_degree_1 += 1
            elif degree == 2:
                nodes_degree_2 += 1
            elif degree > 2:
                nodes_degree_gt2 += 1

        admin_stats.append({
            'admin_id': admin_idx,
            'admin_name': admin_row.get('NAME', f'Admin {admin_idx}'),
            'area_sqkm': admin_row.get('area_sqkm', None),
            'inter_admin_edges': inter_edges,
            'nodes_degree_gt2': nodes_degree_gt2,
            'nodes_degree_1': nodes_degree_1,
            'nodes_degree_2': nodes_degree_2
        })

    df = pd.DataFrame(admin_stats)

    print("="*70)
    print("SELECTED STATISTICS")
    print("="*70)
    print(f"Total inter-admin edges in graph: {total_inter_admin_edges}")
    print("\nPer-admin statistics:")
    print(df.to_string(index=False))
    print("="*70)

    return df, total_inter_admin_edges


def calculate_fractal_dimension(shapefile_path, min_boxes=10, max_boxes=1000, n_sizes=15):
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs and not gdf.crs.is_projected:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())

    minx, miny, maxx, maxy = gdf.total_bounds
    max_extent = max(maxx - minx, maxy - miny)

    epsilons = np.logspace(
        np.log10(max_extent / np.sqrt(max_boxes)),
        np.log10(max_extent / np.sqrt(min_boxes)),
        n_sizes
    )

    box_counts = []

    for eps in epsilons:
        x_edges = np.arange(minx, maxx + eps, eps)
        y_edges = np.arange(miny, maxy + eps, eps)

        occupied_boxes = set()

        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue

            geom_bounds = geom.bounds

            i_min = int((geom_bounds[0] - minx) / eps)
            i_max = int((geom_bounds[2] - minx) / eps) + 1
            j_min = int((geom_bounds[1] - miny) / eps)
            j_max = int((geom_bounds[3] - miny) / eps) + 1

            for i in range(max(0, i_min), min(len(x_edges)-1, i_max)):
                for j in range(max(0, j_min), min(len(y_edges)-1, j_max)):
                    box_geom = Box(x_edges[i], y_edges[j],
                                   x_edges[i+1], y_edges[j+1])

                    if geom.intersects(box_geom):
                        occupied_boxes.add((i, j))

        box_counts.append(len(occupied_boxes))

    log_eps = np.log10(epsilons)
    log_count = np.log10(box_counts)
    slope, _, r_value, _, _ = stats.linregress(log_eps, log_count)

    return -slope, r_value**2


def visualize_graph_on_admin_boundary(G, admin_shp_path, admin_id=None,
                                      admin_name_field=None, admin_name_value=None,
                                      clip_to_boundary=True, roads_gdf=None,
                                      custom_title=None, save_svg=None,
                                      road_crs=None):

    admin_gdf = gpd.read_file(admin_shp_path)
    if road_crs is not None and admin_gdf.crs != road_crs:
        admin_gdf = admin_gdf.to_crs(road_crs)
    print(f"Administrative shapefile contains {len(admin_gdf)} boundaries")

    if admin_name_field and admin_name_value:
        admin_boundary = admin_gdf[admin_gdf[admin_name_field] == admin_name_value]
        if len(admin_boundary) == 0:
            print(f"No boundary found with {admin_name_field} = {admin_name_value}")
            print(f"Available values: {admin_gdf[admin_name_field].unique()}")
            return None, None, None
        admin_boundary = admin_boundary.iloc[0]
        title = f"Road Network - {admin_name_value}"
    elif admin_id is not None:
        if admin_id >= len(admin_gdf):
            print(f"Admin ID {admin_id} out of range. Max index: {len(admin_gdf)-1}")
            return None, None, None
        admin_boundary = admin_gdf.iloc[admin_id]
        title = f"Road Network - Admin Boundary {admin_id}"
    else:
        admin_boundary = admin_gdf.iloc[0]
        title = "Road Network - Admin Boundary"
        print("No specific boundary selected. Using first boundary.")

    if custom_title is not None:
        title = custom_title

    boundary_geom = admin_boundary.geometry

    if clip_to_boundary:
        G_clipped = G.copy()
        nodes_to_remove = []
        for node in G_clipped.nodes():
            point = Point(node)
            if not boundary_geom.contains(point):
                nodes_to_remove.append(node)
        G_clipped.remove_nodes_from(nodes_to_remove)
        print(f"Clipped graph: {G_clipped.number_of_nodes()} nodes and {G_clipped.number_of_edges()} edges")
        G_display = G_clipped
    else:
        G_display = G

    G_display.remove_edges_from(nx.selfloop_edges(G_display))

    fig, ax = plt.subplots(figsize=(15, 12))

    gpd.GeoSeries([boundary_geom]).plot(ax=ax, facecolor='lightgray',
                                         edgecolor='black', linewidth=2,
                                         alpha=0.3, label='Admin Boundary')


    pos = {node: node for node in G_display.nodes()}

    print(f"Number of nodes: {len(G_display.nodes())}")
    print(f"Number of edges: {len(G_display.edges())}")

    if len(pos) > 0:
        for u, v, data in G_display.edges(data=True):
            x_coords = [pos[u][0], pos[v][0]]
            y_coords = [pos[u][1], pos[v][1]]
            ax.plot(x_coords, y_coords, color='blue', linewidth=1.5,
                   alpha=0.7, zorder=2)

        edge_line = Line2D([0], [0], color='blue', linewidth=1.5,
                          alpha=0.7, label='Graph Edges')

        node_collection = ax.scatter([pos[node][0] for node in G_display.nodes()],
                                     [pos[node][1] for node in G_display.nodes()],
                                     s=30, c='#505050', alpha=0.8, zorder=3,
                                     label='Graph Nodes')
    else:
        print("Warning: No nodes found within the boundary!")

    minx, miny, maxx, maxy = boundary_geom.bounds
    margin = 0.05 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - margin, maxx + margin)
    ax.set_ylim(miny - margin, maxy + margin)

    ax.set_aspect('equal')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    handles, labels = ax.get_legend_handles_labels()
    if len(pos) > 0 and len(G_display.edges()) > 0:
        handles.append(edge_line)
        labels.append('Graph Edges')

    ax.legend(handles=handles, labels=labels, loc='upper right')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_svg is not None:
        plt.rcParams['svg.fonttype'] = 'none'
        fig.savefig(save_svg, format='svg', bbox_inches='tight', dpi=300)
        print(f"Figure saved as: {save_svg}")

    return fig, ax, G_display if clip_to_boundary else None
