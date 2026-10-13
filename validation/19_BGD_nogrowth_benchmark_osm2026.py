import os
DATA = os.environ.get("ROAD_DATA", "data")
import sys, numpy as np, rasterio, pandas as pd
import importlib.util
spec = importlib.util.spec_from_file_location("v", os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_BGD_validation_dual_baseline.py"))
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.morphology import disk

with rasterio.open(v.REF_TIF) as s:
    fc = s.read(1); T = s.transform; crs = s.crs; shp = fc.shape; px = abs(T.a); nod = s.nodata
vals, cnt = np.unique(fc, return_counts=True); print("forecast values:", dict(zip(vals.tolist(), cnt.tolist())), "px", px)
valid = fc != nod if nod is not None else np.ones_like(fc, bool)
pred_all = valid & (fc > 0)
pred_new = valid & (fc >= 11) & (fc <= 14)
pred_base = pred_all & ~pred_new
osm26, _, km26, _ = v.rasterize_osm(v.OSM_2026, T, crs, shp, "OSM 2026"); osm26 = osm26 > 0
grip, _, kmg, _ = v.rasterize_grip(T, crs, shp); grip = grip > 0
tot = int(valid.sum())
rows = []
for b in v.BUFFERS:
    for name, p in [("Forecast 2030 (baseline + growth)", pred_all), ("No-growth: retained baseline", pred_base), ("No-growth: GRIP 2015 rasterized", grip)]:
        m = v.compute_metrics(p, osm26, b, px, tot); rows.append(dict(Assessment="All roads", Model=name, Buffer=b, **m))

obs_new = osm26 & ~binary_dilation(grip, structure=disk(1))
land = valid
dist = distance_transform_edt(~pred_base) * px
n_new = int(pred_new.sum()); rng = np.random.default_rng(42)
pred_new_dist = dist[pred_new]
cand_any = np.flatnonzero(land & ~pred_base)
def sample_like(nrep=5):
    bins = np.array([0,300,600,900,1500,3000,6000,1e9]); out=[]
    cand_d = dist.ravel()[cand_any]
    h,_ = np.histogram(pred_new_dist, bins)
    for r in range(nrep):
        sel=[]
        for i in range(len(bins)-1):
            pool = cand_any[(cand_d>=bins[i])&(cand_d<bins[i+1])]
            if h[i]>0 and len(pool)>0: sel.append(rng.choice(pool, min(h[i],len(pool)), replace=False))
        m=np.zeros(fc.size,bool); m[np.concatenate(sel)]=True; out.append(m.reshape(fc.shape))
    return out
nulls = {"Null: random over land": [ (lambda m: (m.__setitem__(rng.choice(cand_any, n_new, replace=False), True), m.reshape(fc.shape))[1])(np.zeros(fc.size,bool)) for _ in range(5)],
         "Null: random, distance-matched to forecast": sample_like()}
for b in v.BUFFERS:
    for ref_name, ref in [("OSM 2026 all roads", osm26), ("OSM 2026 roads absent from GRIP 2015", obs_new)]:
        m = v.compute_metrics(pred_new, ref, b, px, tot); rows.append(dict(Assessment=f"New roads vs {ref_name}", Model="Forecast new roads", Buffer=b, **m))
        for nn, reps in nulls.items():
            ms = [v.compute_metrics(r, ref, b, px, tot) for r in reps]
            avg = {k: round(float(np.mean([x[k] for x in ms])),4) for k in ms[0]}
            rows.append(dict(Assessment=f"New roads vs {ref_name}", Model=nn+" (mean of 5)", Buffer=b, **avg))
df = pd.DataFrame(rows); df.to_csv("BGD_nogrowth_null_benchmarks.csv", index=False)
pd.set_option("display.width", 250)
print(df[["Assessment","Model","Buffer","N_pred","Precision","Recall","F1","Kappa"]].to_string(index=False))
print("n_new px", n_new, "baseline px", int(pred_base.sum()), "OSM26 px", int(osm26.sum()), "obs_new px", int(obs_new.sum()))
