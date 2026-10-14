import argparse
import glob
import os
import shutil

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy import signal
from skimage.graph import MCP_Geometric

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CLS = {'motorway': 1, 'motorway_link': 1, 'trunk': 1, 'trunk_link': 1, 'primary': 1, 'primary_link': 1,
       'secondary': 2, 'secondary_link': 2, 'tertiary': 3, 'tertiary_link': 3,
       'residential': 4, 'unclassified': 4, 'living_street': 4, 'service': 4}


def quantile_classes(v, mask, n=5, increasing=True):
    out = np.zeros(v.shape, np.float32)
    x = v[mask].astype(np.float64)
    if not increasing:
        x = -x
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    cum = (np.cumsum(cnt) - cnt / 2.0) / x.size
    out[mask] = np.minimum((cum * n).astype(int) + 1, n)[inv].astype(np.float32)
    return out


def cost_distance(src, cost, res_m):
    d, _ = MCP_Geometric(np.where(np.isfinite(cost), cost, 1e6).astype(np.float64), fully_connected=True).find_costs(np.argwhere(src))
    return (d * res_m).astype(np.float32)


def line_density(road, r_px, res_km=0.3):
    y, x = np.ogrid[-r_px:r_px + 1, -r_px:r_px + 1]
    k = ((x * x + y * y) <= r_px * r_px).astype(np.float32)
    c = np.clip(np.rint(signal.fftconvolve((road != 0).astype(np.float32), k, mode='same')), 0, None)
    return (c * res_km / (k.sum() * res_km ** 2)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description='Model inputs with OSM 2015 roads in place of GRIP for Bangladesh.')
    ap.add_argument('--inputs', default=os.path.join(REPO, 'data', 'model_inputs', 'BGD', 'SSP1', '2030'))
    ap.add_argument('--slope_deg', required=True)
    ap.add_argument('--osm2015', default=os.path.join(REPO, 'data', 'figures', 'osm', 'bangladesh-150101-free.shp', 'gis_osm_roads_free_1.shp.zip'))
    ap.add_argument('--out', default=os.path.join(REPO, 'data', 'model_inputs_osm2015', 'BGD', 'SSP1', '2030'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in glob.glob(os.path.join(a.inputs, '*')):
        shutil.copy(f, a.out)
    with rasterio.open(os.path.join(a.inputs, 'BGD_slope.tif')) as r:
        mask = r.read(1) > 0
        prof, T, crs = r.profile, r.transform, r.crs
    with rasterio.open(a.slope_deg) as r:
        sd = r.read(1).astype(np.float32)
        sd = np.where(sd == 65535, 0, sd) * float(r.tags().get('scale', 1))
    g = gpd.read_file('zip://' + a.osm2015 if a.osm2015.endswith('.zip') else a.osm2015)
    g = g[g.fclass.isin(list(CLS))].to_crs(crs)
    g['k'] = g.fclass.map(CLS)
    g = g.sort_values('k', ascending=False)
    rt = rasterize(((x, int(k)) for x, k in zip(g.geometry, g.k)), out_shape=mask.shape, transform=T, fill=0, dtype='uint8')
    rt[~mask] = 0
    rd = line_density(rt, 33)
    rd[~mask] = 0
    layers = dict(road_types=rt.astype(np.float32),
                  dist_roads=quantile_classes(cost_distance(rt > 0, 1.0 + sd / 10.0, 300.0), mask, n=5, increasing=False),
                  road_density=quantile_classes(rd, mask, n=5, increasing=True), road_density_total=rd)
    p = dict(prof)
    p.update(dtype='float32', nodata=None, compress='deflate', tiled=True, blockxsize=512, blockysize=512)
    for n, v in layers.items():
        with rasterio.open(os.path.join(a.out, f'BGD_{n}.tif'), 'w', **p) as w:
            w.write(v.astype(np.float32), 1)
    print('OSM 2015 road cells', int((rt > 0).sum()), 'by class', [int((rt == k).sum()) for k in (1, 2, 3, 4)])


if __name__ == '__main__':
    main()
