import os
DATA = os.environ.get("ROAD_DATA", "data")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(DATA, "output")
EFF = 900

C_CORR = '#2166AC'; C_COMP = '#B2182B'; C_QUAL = '#984EA3'
C_MR = '#2166AC'; C_DR = '#B2182B'; C_F1 = '#4DAF4A'

skel = np.load(f"{OUT}/BRA_skeleton_distances_raw.npz")
dp2o_m = skel['dp2o_m']; do2p_m = skel['do2p_m']

conv = np.load(f"{OUT}/BRA_convergence_raw.npz")
pp = conv['pp']; po = conv['po']
int_bufs = conv['int_bufs'].tolist()
int_mr = conv['int_mr'].tolist()
int_dr = conv['int_dr'].tolist()
int_f1 = conv['int_f1'].tolist()
shape = tuple(conv['shape'])

skel_bufs = [0, 900, 1800, 2700, 3600, 4500]
dp2o_pix = dp2o_m / EFF; do2p_pix = do2p_m / EFF
skel_corr = [float((dp2o_pix <= b/EFF).mean()) for b in skel_bufs]
skel_comp = [float((do2p_pix <= b/EFF).mean()) for b in skel_bufs]
skel_qual = [c*cm/(c+cm-c*cm) if (c+cm)>0 else 0 for c,cm in zip(skel_corr, skel_comp)]

print(f"  Skeleton distances loaded: {len(dp2o_m):,} pred, {len(do2p_m):,} obs")
print(f"  Convergence points loaded: {len(pp):,} pred, {len(po):,} obs")
print(f"  Shape: {shape}")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

ax = axes[0,0]
bins_p = np.sort(dp2o_m); bins_o = np.sort(do2p_m)
cdf_p = np.arange(1, len(bins_p)+1)/len(bins_p)
cdf_o = np.arange(1, len(bins_o)+1)/len(bins_o)
ax.plot(bins_p, cdf_p, color=C_CORR, linewidth=1.5, label='Correctness (Pred→Obs)')
ax.plot(bins_o, cdf_o, color=C_COMP, linewidth=1.5, label='Completeness (Obs→Pred)')
ax.set_xlim(0, 8000); ax.set_ylim(0, 1.05)
ax.set_xlabel('Distance (m)'); ax.set_ylabel('Cumulative fraction')
ax.set_title('(a) Skeleton distance CDF', fontweight='bold', fontsize=12)
ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.2)
ax.text(0.02, 0.02, f'pred: {len(dp2o_m):,}\nobs: {len(do2p_m):,}\n~900m effective res.',
        transform=ax.transAxes, fontsize=7, va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax = axes[0,1]
ax.plot(skel_bufs, skel_corr, 'o-', color=C_CORR, linewidth=2, markersize=6, label='Correctness')
ax.plot(skel_bufs, skel_comp, 's-', color=C_COMP, linewidth=2, markersize=6, label='Completeness')
ax.plot(skel_bufs, skel_qual, '^-', color=C_QUAL, linewidth=2, markersize=6, label='Quality')
ax.axhline(0.90, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_xlim(-50, 5000); ax.set_ylim(0, 1.05)
ax.set_xlabel('Buffer tolerance (m)'); ax.set_ylabel('Fraction')
ax.set_title('(b) Skeleton overlap vs. buffer', fontweight='bold', fontsize=12)
ax.legend(fontsize=8, loc='center right'); ax.grid(True, alpha=0.2)
for i, b in enumerate(skel_bufs):
    if b in [0, 900, 2700]:
        ax.annotate(f'{skel_corr[i]:.3f}', (b, skel_corr[i]), textcoords="offset points",
                    xytext=(5, 8), fontsize=6, color=C_CORR)

ax = axes[1,0]
ax.plot(int_bufs, int_mr, 'o-', color=C_MR, linewidth=2, markersize=6, label='Match rate (precision)')
ax.plot(int_bufs, int_dr, 's-', color=C_DR, linewidth=2, markersize=6, label='Detection rate (recall)')
ax.plot(int_bufs, int_f1, '^-', color=C_F1, linewidth=2, markersize=6, label='F1 score')
ax.set_ylim(0, 1.05)
ax.set_xlabel('Buffer tolerance (m)'); ax.set_ylabel('Rate')
ax.set_title('(c) Intersection matching vs. buffer', fontweight='bold', fontsize=12)
ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.2)
for i, b in enumerate(int_bufs):
    if b in [1800, 5400]:
        ax.annotate(f'F1={int_f1[i]:.3f}', (b, int_f1[i]), textcoords="offset points",
                    xytext=(5, -15), fontsize=7, color=C_F1)

ax = axes[1,1]
pp_km_x = pp[:,1] * EFF / 1000; pp_km_y = pp[:,0] * EFF / 1000
po_km_x = po[:,1] * EFF / 1000; po_km_y = po[:,0] * EFF / 1000
ax.scatter(po_km_x, po_km_y, s=1.5, c=C_COMP, alpha=0.4, linewidths=0, label=f'Observed ({len(po):,})')
ax.scatter(pp_km_x, pp_km_y, s=1.5, c=C_CORR, alpha=0.4, linewidths=0, label=f'Predicted ({len(pp):,})')
ax.set_xlim(0, shape[1]*EFF/1000); ax.set_ylim(shape[0]*EFF/1000, 0)
ax.set_xlabel('Easting (km)'); ax.set_ylabel('Northing (km)')
ax.set_title('(d) Intersection density peaks', fontweight='bold', fontsize=12)
ax.legend(fontsize=8, loc='upper right', markerscale=4); ax.set_aspect('equal')

plt.suptitle('Brazil — Locational Accuracy Assessment (~900 m resolution)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(pad=1.5)
out = f"{OUT}/fig_BRA_locational_accuracy_20260612.png"
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
out_hi = f"{OUT}/fig_BRA_locational_accuracy_20260612_hires.png"
fig.savefig(out_hi, dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {out}")
print(f"Saved: {out_hi}")
