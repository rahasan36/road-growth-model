import os
DATA = os.environ.get("ROAD_DATA", "data")
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(DATA, "output")
DATE = "20260607"

bgd = pd.read_csv(f"{OUTPUT_DIR}/validation_dual_baseline_{DATE}.csv")
if 'Country' not in bgd.columns: bgd['Country'] = 'BGD'
nga = pd.read_csv(f"{OUTPUT_DIR}/validation_NGA_dual_{DATE}.csv")
bra = pd.read_csv(f"{OUTPUT_DIR}/validation_BRA_dual_{DATE}.csv")
df = pd.concat([bgd, nga, bra], ignore_index=True)

metrics_cols = ['Precision','Recall','F1','IoU','Kappa','MCC','Commission','Omission',
                'FoM','Quantity_Disagr','Allocation_Disagr','Total_Disagr','Overall_Accuracy']
avg = df.groupby(['Country','Baseline','Comparison','Buffer_m'])[metrics_cols].mean().reset_index()

COUNTRIES = ['BGD', 'NGA', 'BRA']
COUNTRY_NAMES = {'BGD': 'Bangladesh', 'NGA': 'Nigeria', 'BRA': 'Brazil'}
BASELINES = ['GRIP_2015', 'OSM_2015']
BUFFERS = [0, 300, 600, 900]

C_GRIP = '#1b9e77'
C_OSM = '#1f77b4'
C_COUNTRY = {'BGD': '#e41a1c', 'NGA': '#377eb8', 'BRA': '#4daf4a'}

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10})

print("Figure 1: F1 vs Buffer (all countries)...")
fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey='row')

for j, country in enumerate(COUNTRIES):
    for i, (comp, title_suffix) in enumerate([('All_roads', 'All roads'), ('New_roads', 'New roads only')]):
        ax = axes[i, j]
        for base, color, ls, marker, lbl in [
            ('GRIP_2015', C_GRIP, '-', 'o', 'GRIP 2015'),
            ('OSM_2015', C_OSM, '--', 's', 'OSM 2015'),
        ]:
            sub = avg[(avg['Country']==country) & (avg['Baseline']==base) &
                      (avg['Comparison']==comp)].sort_values('Buffer_m')
            if len(sub) == 0: continue
            ax.plot(sub['Buffer_m'], sub['F1'], color=color, linestyle=ls,
                    marker=marker, markersize=5, linewidth=2, label=lbl)
            v300 = sub[sub['Buffer_m']==300]['F1'].values
            if len(v300) > 0:
                ax.annotate(f'{v300[0]:.3f}', xy=(300, v300[0]),
                            xytext=(10, 8), textcoords='offset points',
                            fontsize=7.5, color=color, fontweight='bold')

        ax.set_xticks([0, 300, 600, 900])
        ax.set_xticklabels(['0', '300', '600', '900'])
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if i == 0:
            ax.set_title(COUNTRY_NAMES[country], fontsize=12, fontweight='bold')
        if j == 0:
            ax.set_ylabel(f'F1 — {title_suffix}', fontsize=10)
        if i == 1:
            ax.set_xlabel('Buffer distance (m)', fontsize=10)

axes[0, 0].legend(fontsize=8, loc='lower right')
plt.tight_layout()
out1 = f"{OUTPUT_DIR}/fig_validation_f1_buffer_all_{DATE}.svg"
fig.savefig(out1, format='svg', bbox_inches='tight')
plt.close(fig)
print(f"  {out1}")


print("Figure 2: Commission/Omission (all countries)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

x = np.arange(len(COUNTRIES))
w = 0.35
sub = avg[(avg['Comparison']=='All_roads') & (avg['Buffer_m']==300)]

for ax, metric, title in [
    (ax1, 'Commission', 'Commission error (FP rate) — All roads, 300 m'),
    (ax2, 'Omission', 'Omission error (FN rate) — All roads, 300 m'),
]:
    grip_vals = [sub[(sub['Country']==c) & (sub['Baseline']=='GRIP_2015')][metric].values[0]
                 for c in COUNTRIES]
    osm_vals = [sub[(sub['Country']==c) & (sub['Baseline']=='OSM_2015')][metric].values[0]
                for c in COUNTRIES]
    b1 = ax.bar(x - w/2, grip_vals, w, label='GRIP 2015', color=C_GRIP, alpha=0.85,
                edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + w/2, osm_vals, w, label='OSM 2015', color=C_OSM, alpha=0.85,
                edgecolor='white', linewidth=0.5)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([COUNTRY_NAMES[c] for c in COUNTRIES])
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8)

plt.tight_layout()
out2 = f"{OUTPUT_DIR}/fig_validation_comm_omis_all_{DATE}.svg"
fig.savefig(out2, format='svg', bbox_inches='tight')
plt.close(fig)
print(f"  {out2}")


print("Figure 3: Multi-metric (all countries, GRIP baseline)...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

met_list = ['F1', 'IoU', 'Kappa', 'MCC', 'Precision', 'Recall']
x = np.arange(len(met_list))
w = 0.25

for ax, comp, title in [
    (ax1, 'All_roads', 'All roads — GRIP baseline, 300 m'),
    (ax2, 'New_roads', 'New roads — GRIP baseline, 300 m'),
]:
    for k, country in enumerate(COUNTRIES):
        sub = avg[(avg['Country']==country) & (avg['Baseline']=='GRIP_2015') &
                  (avg['Comparison']==comp) & (avg['Buffer_m']==300)]
        if len(sub) == 0: continue
        vals = [sub[m].values[0] for m in met_list]
        ax.bar(x + k*w, vals, w, label=COUNTRY_NAMES[country],
               color=C_COUNTRY[country], alpha=0.85, edgecolor='white', linewidth=0.3)

    ax.set_xticks(x + w)
    ax.set_xticklabels(met_list, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8)

plt.tight_layout()
out3 = f"{OUTPUT_DIR}/fig_validation_metrics_all_{DATE}.svg"
fig.savefig(out3, format='svg', bbox_inches='tight')
plt.close(fig)
print(f"  {out3}")


print("Figure 4: Pontius spatial metrics (all countries)...")
fig, axes = plt.subplots(1, 3, figsize=(13, 5))

for j, country in enumerate(COUNTRIES):
    ax = axes[j]
    sub_grip = avg[(avg['Country']==country) & (avg['Baseline']=='GRIP_2015') &
                   (avg['Comparison']=='All_roads')].sort_values('Buffer_m')
    sub_osm = avg[(avg['Country']==country) & (avg['Baseline']=='OSM_2015') &
                  (avg['Comparison']=='All_roads')].sort_values('Buffer_m')

    x = np.arange(len(BUFFERS))
    w = 0.35

    for sub, offset, color, label in [
        (sub_grip, -w/2, C_GRIP, 'GRIP'),
        (sub_osm, w/2, C_OSM, 'OSM'),
    ]:
        qd = sub['Quantity_Disagr'].values
        ad = sub['Allocation_Disagr'].values
        ax.bar(x + offset, qd, w, label=f'{label} QD', color=color, alpha=0.9,
               edgecolor='white', linewidth=0.3)
        ax.bar(x + offset, ad, w, bottom=qd, label=f'{label} AD', color=color,
               alpha=0.5, edgecolor='white', linewidth=0.3, hatch='///')

    ax.set_xticks(x)
    ax.set_xticklabels(['Exact', '300m', '600m', '900m'])
    ax.set_xlabel('Buffer distance')
    ax.set_title(COUNTRY_NAMES[country], fontsize=11, fontweight='bold')
    if j == 0:
        ax.set_ylabel('Disagreement')
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)

    for k, buf in enumerate(BUFFERS):
        for sub, offset, color in [(sub_grip, -w/2, C_GRIP), (sub_osm, w/2, C_OSM)]:
            row = sub[sub['Buffer_m']==buf]
            if len(row) > 0:
                oa = row['Overall_Accuracy'].values[0]
                td = row['Total_Disagr'].values[0]
                ax.text(k + offset, td + 0.003, f'OA={oa:.3f}',
                        ha='center', va='bottom', fontsize=6.5, color=color, rotation=90)

axes[0].legend(fontsize=7, loc='upper right', ncol=2)

plt.tight_layout()
out4 = f"{OUTPUT_DIR}/fig_validation_spatial_all_{DATE}.svg"
fig.savefig(out4, format='svg', bbox_inches='tight')
plt.close(fig)
print(f"  {out4}")


print("Figure 5: Cross-country F1 summary...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, comp, title in [
    (ax1, 'All_roads', 'All roads (total network)'),
    (ax2, 'New_roads', 'New roads only'),
]:
    for country in COUNTRIES:
        for base, ls, alpha in [('GRIP_2015', '-', 1.0), ('OSM_2015', '--', 0.6)]:
            sub = avg[(avg['Country']==country) & (avg['Baseline']==base) &
                      (avg['Comparison']==comp)].sort_values('Buffer_m')
            if len(sub) == 0: continue
            lbl = f'{COUNTRY_NAMES[country]} ({base.split("_")[0]})'
            ax.plot(sub['Buffer_m'], sub['F1'], color=C_COUNTRY[country],
                    linestyle=ls, marker='o', markersize=4, linewidth=1.8,
                    alpha=alpha, label=lbl)

    ax.set_xticks([0, 300, 600, 900])
    ax.set_xticklabels(['0\n(exact)', '300', '600', '900'])
    ax.set_xlabel('Buffer distance (m)')
    ax.set_ylabel('F1 Score')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7, loc='best', ncol=2)

plt.tight_layout()
out5 = f"{OUTPUT_DIR}/fig_validation_f1_summary_all_{DATE}.svg"
fig.savefig(out5, format='svg', bbox_inches='tight')
plt.close(fig)
print(f"  {out5}")


print(f"\nAll 5 SVGs saved to {OUTPUT_DIR}/")
