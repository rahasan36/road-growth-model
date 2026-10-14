import argparse
import contextlib
import gc
import io
import json
import os
import sys

import geopandas as gpd
import networkit as nk
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import shapely
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from reliability.analyze_road_network_scenarios import calculate_additional_metrics
from raster_to_network import vectorize

EXTRACTS = {'BGD': [17, 9], 'NGA': [11]}


def to_graph(gdf):
    geoms = gdf.geometry.values
    xy = shapely.get_coordinates(geoms)
    n = shapely.get_num_coordinates(geoms)
    end = np.cumsum(n) - 1
    beg = end - n + 1
    a = [tuple(p) for p in np.round(xy[beg], 3)]
    b = [tuple(p) for p in np.round(xy[end], 3)]
    G = nx.Graph()
    for u, v, l in zip(a, b, shapely.length(geoms)):
        if u != v:
            G.add_edge(u, v, length=float(l))
    return G


def nk_graph(G):
    idx = {v: i for i, v in enumerate(G.nodes())}
    g = nk.Graph(len(idx), weighted=True)
    for u, v, d in G.edges(data=True):
        g.addEdge(idx[u], idx[v], d['length'])
    return g


def sampled_paths(g, n_src=500):
    n = g.numberOfNodes()
    k = min(n, n_src)
    s, c, e = 0.0, 0, 0.0
    for src in range(k):
        sp = nk.distance.Dijkstra(g, src)
        sp.run()
        d = np.asarray(sp.getDistances())
        d = d[(d > 0) & (d < 1e300)]
        s += d.sum()
        c += len(d)
        e += float((1.0 / d).sum())
    return s / c if c else None, e / (k * (n - 1))


def diameter(g, starts=5, sweeps=4, seed=42):
    rng = np.random.default_rng(seed)
    best = 0.0
    for src in rng.integers(g.numberOfNodes(), size=starts):
        src = int(src)
        for _ in range(sweeps):
            sp = nk.distance.Dijkstra(g, src)
            sp.run()
            d = np.asarray(sp.getDistances())
            d[d > 1e300] = -1
            src = int(d.argmax())
            best = max(best, float(d[src]))
    return best


def admin_stats(G, admin, crs):
    adm = admin.to_crs(crs).reset_index(drop=True)
    nodes = list(G.nodes())
    pts = gpd.GeoDataFrame({'deg': [G.degree(x) for x in nodes]},
                           geometry=gpd.points_from_xy([p[0] for p in nodes], [p[1] for p in nodes]), crs=crs)
    j = gpd.sjoin(pts, adm[['geometry']], how='inner', predicate='intersects')
    df = pd.DataFrame({'admin_id': adm.index, 'area_sqkm': admin['area_sqkm'].values})
    df['nodes_degree_1'] = df.admin_id.map(j[j.deg == 1].groupby('index_right').size()).fillna(0).astype(int)
    df['nodes_degree_gt2'] = df.admin_id.map(j[j.deg > 2].groupby('index_right').size()).fillna(0).astype(int)
    df['id1'] = df.nodes_degree_gt2 / df.area_sqkm
    df['id2'] = (df.nodes_degree_gt2 + df.nodes_degree_1) / df.area_sqkm
    return df


def fractal(gdf, min_boxes=10, max_boxes=1000, n_sizes=15):
    tree = shapely.STRtree(gdf.geometry.values)
    minx, miny, maxx, maxy = gdf.total_bounds
    ext = max(maxx - minx, maxy - miny)
    eps = np.logspace(np.log10(ext / np.sqrt(max_boxes)), np.log10(ext / np.sqrt(min_boxes)), n_sizes)
    counts = []
    for e in eps:
        X0, Y0 = np.meshgrid(np.arange(minx, maxx + e, e)[:-1], np.arange(miny, maxy + e, e)[:-1])
        boxes = shapely.box(X0.ravel(), Y0.ravel(), X0.ravel() + e, Y0.ravel() + e)
        counts.append(len(np.unique(tree.query(boxes, predicate='intersects')[0])))
    s, _, r, _, _ = stats.linregress(np.log10(eps), np.log10(counts))
    return -s, r ** 2


def networks(inputs, outputs, C):
    yield 2015, 'existing', os.path.join(inputs, C, 'SSP1', '2030', f'{C}_road_types.tif')
    for y in (2030, 2040, 2050):
        for s in (1, 3, 5):
            yield y, f'ssp{s}', os.path.join(outputs, C, f'SSP{s}', str(y), f'{C}_all_final_roads.tif')


def main():
    ap = argparse.ArgumentParser(description='Network metrics of the existing and forecast road networks.')
    ap.add_argument('country')
    ap.add_argument('--inputs', default=os.path.join(REPO, 'data', 'model_inputs'))
    ap.add_argument('--outputs', default=os.path.join(REPO, 'data', 'model_outputs'))
    ap.add_argument('--admin', required=True)
    ap.add_argument('--extract_crs', default=None)
    ap.add_argument('--out', default=os.path.join(HERE, 'output'))
    a = ap.parse_args()
    C = a.country
    out = os.path.join(a.out, C)
    os.makedirs(out, exist_ok=True)
    with rasterio.open(os.path.join(a.inputs, C, 'SSP1', '2030', f'{C}_slope.tif')) as s:
        mask = s.read(1) > 0
        T, crs = s.transform, s.crs
    admin = gpd.read_file(a.admin)
    rows = []
    for y, sc, path in networks(a.inputs, a.outputs, C):
        with rasterio.open(path) as s:
            code = s.read(1)
        code[~mask] = 0
        code = code.astype(np.uint8)
        row = dict(country=C, year=y, scenario=sc, cells=int((code > 0).sum()))
        for c in (1, 2, 3, 4):
            Gc = to_graph(vectorize((code % 10 == c) & (code > 0), T, crs))
            row[f'nodes_c{c}'], row[f'edges_c{c}'] = Gc.number_of_nodes(), Gc.number_of_edges()
        lines = vectorize(code > 0, T, crs)
        G = to_graph(lines)
        row.update(nodes=G.number_of_nodes(), edges=G.number_of_edges(), km=round(float(lines.length.sum() / 1000), 1))
        row.update(D_mesh=nx.density(G), C_avg=nx.average_clustering(G), T=nx.transitivity(G))
        H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        h = nk_graph(H)
        row['lcc_nodes'] = H.number_of_nodes()
        dm = nk.distance.Diameter(nk.nxadapter.nx2nk(H), algo=1)
        dm.run()
        row['diam_links'] = dm.getDiameter()[0]
        row['diam_km'] = round(diameter(h) / 1000, 1)
        nk.engineering.setSeed(42, False)
        cc = nk.centrality.ApproxCloseness(h, nSamples=min(h.numberOfNodes(), 500))
        cc.run()
        row['C_C_mean'] = float(np.nanmean(cc.scores()))
        df = admin_stats(G, admin, crs)
        df.to_csv(f'{out}/admin_{y}_{sc}.csv', index=False)
        for k in ('id1', 'id2'):
            row.update({f'{k}_mean': df[k].mean(), f'{k}_var': df[k].var(), f'{k}_cv': df[k].std() / df[k].mean(), f'{k}_median': df[k].median()})
        row['D_frac'], row['R2_frac'] = fractal(lines)
        nk.engineering.setSeed(42, False)
        with contextlib.redirect_stdout(io.StringIO()):
            extra = calculate_additional_metrics(G, total_area_sqkm=float(admin['area_sqkm'].sum()))
        row.update({k: extra[k] for k in ('C_D_mean', 'C_B_mean', 'alpha', 'beta', 'gamma', 'eta', 'D_net')})
        row['L_avg_km'], row['E_global'] = sampled_paths(nk_graph(G))
        row['L_avg_km'] /= 1000
        if C in EXTRACTS and y == 2050 and a.extract_crs:
            split = vectorize(code, T, crs, split_classes=True)
            adm = admin.to_crs(crs)
            ex = pd.concat([split.iloc[split.sindex.query(adm.geometry.iloc[i], predicate='intersects')] for i in EXTRACTS[C]]).drop_duplicates()
            gpd.GeoDataFrame(ex, geometry='geometry', crs=crs).to_crs(gpd.read_file(a.extract_crs, rows=1).crs).to_file(f'{out}/{C.lower()}_roads_{y}_{sc}.shp')
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
        del G, H, lines, code
        gc.collect()
    pd.DataFrame(rows).to_csv(f'{out}/network_metrics_{C}.csv', index=False)


if __name__ == '__main__':
    main()
