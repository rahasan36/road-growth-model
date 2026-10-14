import argparse
import gc
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validation_metrics as V

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BTM = 'ESRI:102953'
REGIONS = [('Padma Bridge corridor', 23.46, 90.26, 20), ('Rangpur-Dinajpur', 26.04, 89.55, 25), ('Chittagong Hill Tracts', 23.50, 92.20, 30)]


def layer(path, g):
    with rasterio.open(path) as s:
        if s.shape != g['shape'] or s.transform != g['T']:
            raise SystemExit(f'{path} is not on the model grid')
        return s.read(1)


def osm_layer(path, g):
    if path.endswith('.tif'):
        return layer(path, g) > 0
    parts = sorted(glob.glob(os.path.join(path, 'gis_osm_roads_free_1*.shp.zip')))
    return V.rasterize_geofabrik(['zip://' + p for p in parts], g)[0]


def write_map(path, m, g):
    p = g['profile'].copy()
    p.update(dtype='uint8', count=1, nodata=0, compress='deflate')
    with rasterio.open(path, 'w', **p) as w:
        w.write(m, 1)


def main():
    ap = argparse.ArgumentParser(description='Validate the 2030 SSP1 forecast against OpenStreetMap.')
    ap.add_argument('country')
    ap.add_argument('--inputs', default=os.path.join(REPO, 'data', 'model_inputs'))
    ap.add_argument('--outputs', default=os.path.join(REPO, 'data', 'model_outputs'))
    ap.add_argument('--osm2015', required=True)
    ap.add_argument('--osm2026', required=True)
    ap.add_argument('--osm15_inputs', default=None)
    ap.add_argument('--osm15_outputs', default=None)
    ap.add_argument('--out', default=os.path.join(HERE, 'output'))
    a = ap.parse_args()
    C = a.country
    out = os.path.join(a.out, C)
    os.makedirs(out, exist_ok=True)
    inp = os.path.join(a.inputs, C, 'SSP1', '2030')
    res = os.path.join(a.outputs, C, 'SSP1', '2030')
    g = V.read_grid(f'{inp}/{C}_slope.tif')
    px = g['px']
    mask = layer(f'{inp}/{C}_slope.tif', g) > 0
    tot = int(mask.sum())
    grip = (layer(f'{inp}/{C}_road_types.tif', g) > 0) & mask
    fc = layer(f'{res}/{C}_all_final_roads.tif', g)
    pred = (fc > 0) & mask
    new = (fc >= 11) & (fc <= 14) & mask
    del fc
    o26 = osm_layer(a.osm2026, g) & mask
    o15 = osm_layer(a.osm2015, g) & mask
    info = dict(country=C, cell_m=px, domain_cells=tot, grip_cells=int(grip.sum()), forecast_cells=int(pred.sum()),
                forecast_new_cells=int(new.sum()), osm2026_cells=int(o26.sum()), osm2015_cells=int(o15.sum()))

    pd.DataFrame([dict(Buffer_m=b, **V.two_sided(grip, o15, b, px, tot)) for b in V.BUFFERS]).to_csv(f'{out}/grip_vs_osm2015.csv', index=False)

    cases = [('GRIP 2015', pred, o26 | grip)]
    osm15 = a.osm15_inputs is not None and a.osm15_outputs is not None
    if osm15:
        ob = (layer(os.path.join(a.osm15_inputs, C, 'SSP1', '2030', f'{C}_road_types.tif'), g) > 0) & mask
        of = layer(os.path.join(a.osm15_outputs, C, 'SSP1', '2030', f'{C}_all_final_roads.tif'), g)
        opred = (of > 0) & mask
        onew = (of >= 11) & (of <= 14) & mask
        del of
        info.update(osm15_init_baseline_cells=int(ob.sum()), osm15_init_forecast_cells=int(opred.sum()), osm15_init_new_cells=int(onew.sum()))
        cases.append(('OSM 2015', opred, o26 | ob))
    else:
        cases.append(('OSM 2015', pred, o26 | o15))

    rows, maps300 = [], {}
    for name, p, ref in cases:
        for b in V.BUFFERS:
            m = V.percell_map(p, ref, b, px, mask)
            rows.append(dict(Baseline=name, Buffer_m=b, **V.map_metrics(m, mask)))
            if b == 300:
                maps300[name] = m
            else:
                del m
            gc.collect()
    st = pd.DataFrame(rows)
    st.to_csv(f'{out}/percell_metrics.csv', index=False)
    lab = {0: 'exact', 300: '300m', 600: '600m', 900: '900m'}
    cols = ['Buffer', 'Precision', 'Recall', 'F1', 'IoU', 'Kappa', 'MCC', 'FoM', 'QD', 'AD', 'TD', 'OA']
    for name, suf in (('GRIP 2015', ''), ('OSM 2015', '_OSM15')):
        d = st[st.Baseline == name].copy()
        d['Buffer'] = d.Buffer_m.map(lab)
        d[cols].to_csv(f'{out}/{C}_extended_metrics{suf}.csv', index=False)
    for name, m in maps300.items():
        write_map(f'{out}/{C}_spatial_TPFPFN_300m_{"GRIP" if name.startswith("GRIP") else "OSM15"}.tif', m, g)

    pd.DataFrame([dict(Baseline=n, Buffer_m=b, **V.two_sided(p, ref, b, px, tot)) for n, p, ref in cases for b in V.BUFFERS]).to_csv(
        f'{out}/two_sided_allroads.csv', index=False)

    ref = o26 | grip
    if g['shape'][0] * g['shape'][1] > 1e8:
        lp, lo, lpx, bufs = V.maxpool(pred, 3), V.maxpool(ref, 3), px * 3, [900, 1800, 2700, 3600]
    else:
        lp, lo, lpx, bufs = pred, ref, px, [300, 600, 900, 1500]
    la = V.locational_accuracy(lp, lo, lpx, bufs)
    del lp, lo
    json.dump(la, open(f'{out}/locational_accuracy.json', 'w'), indent=1, default=float)

    rows = []
    for b in V.BUFFERS:
        rows.append(dict(Buffer_m=b, Network='Forecast', **V.two_sided(pred, o26, b, px, tot)))
        rows.append(dict(Buffer_m=b, Network='No growth', **V.two_sided(grip, o26, b, px, tot)))
    pd.DataFrame(rows).to_csv(f'{out}/nogrowth_vs_osm2026.csv', index=False)

    rows = [dict(Baseline='GRIP', Buffer_m=b, **V.two_sided(new, o26 & ~grip, b, px, tot)) for b in V.BUFFERS]
    if osm15:
        rows += [dict(Baseline='OSM', Buffer_m=b, **V.two_sided(onew, o26 & ~ob, b, px, tot)) for b in V.BUFFERS]
    pd.DataFrame(rows).to_csv(f'{out}/new_roads.csv', index=False)
    info['osm_growth_fold'] = round(int(o26.sum()) / max(int(o15.sum()), 1), 1)

    if C == 'BGD':
        m = maps300['GRIP 2015']
        nr = V.near(o26 | grip, 300, px) & new
        rr, cc = np.nonzero(mask)
        cx, cy = Transformer.from_crs(g['crs'], BTM, always_xy=True).transform(g['T'].c + (cc + 0.5) * g['T'].a, g['T'].f + (rr + 0.5) * g['T'].e)
        tf = Transformer.from_crs('EPSG:4326', BTM, always_xy=True)
        rows = []
        for name, lat, lon, rad in REGIONS:
            x, y = tf.transform(lon, lat)
            R = rad * 1000.0
            w = (cx >= x - R) & (cx < x + R) & (cy >= y - R) & (cy < y + R)
            r, c = rr[w], cc[w]
            mv = m[r, c]
            tp, fp, fn = (int((mv == k).sum()) for k in (1, 2, 3))
            nn = int(new[r, c].sum())
            rows.append(dict(Region=name, half_width_km=rad, TP=tp, Commission=fp, Omission=fn, New_pred=nn,
                             All_precision=round(tp / (tp + fp), 3) if tp + fp else None,
                             New_precision=round(float(nr[r, c].sum()) / nn, 3) if nn else None,
                             OSM2015_cells=int(o15[r, c].sum()), OSM2026_cells=int(o26[r, c].sum())))
        pd.DataFrame(rows).to_csv(f'{out}/regions.csv', index=False)
        v = (distance_transform_edt(~(o26 & ~grip)) * px / 1000)[new]
        info['new_road_distance_km'] = dict(mean=round(float(v.mean()), 2), median=round(float(np.median(v)), 2), p90=round(float(np.percentile(v, 90)), 2))
    json.dump(info, open(f'{out}/summary.json', 'w'), indent=1)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
