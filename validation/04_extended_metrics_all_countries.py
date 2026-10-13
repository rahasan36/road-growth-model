import os
DATA = os.environ.get("ROAD_DATA", "data")
import numpy as np, rasterio, csv, math, os

OUT_DIR = os.path.join(DATA, "output")

def compute_all_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    pred = tp + fp
    obs = tp + fn
    prec = tp / pred if pred > 0 else 0
    rec  = tp / obs  if obs > 0  else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    iou  = tp/(tp+fp+fn) if (tp+fp+fn)>0 else 0
    po   = (tp+tn)/total
    pe   = ((tp+fp)*(tp+fn)+(fp+tn)*(fn+tn))/(total*total)
    kappa= (po-pe)/(1-pe) if (1-pe)>0 else 0
    num  = tp*tn - fp*fn
    denom= math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)>0 else 1
    mcc  = num/denom
    fom  = iou
    qd   = abs(pred-obs)/total
    ad   = 2*min(fp,fn)/total
    td   = qd + ad
    oa   = (tp+tn)/total
    return dict(Precision=round(prec,3), Recall=round(rec,3), F1=round(f1,3),
                IoU=round(iou,3), Kappa=round(kappa,3), MCC=round(mcc,3),
                FoM=round(fom,3), QD=round(qd,4), AD=round(ad,4),
                TD=round(td,4), OA=round(oa,3))

for code, name in [('BGD','Bangladesh'), ('BRA','Brazil'), ('NGA','Nigeria')]:
    print(f"\n{'='*72}")
    print(f"  {name} ({code})  —  Forecast total network vs observed total network")
    print(f"  GRIP 2015 baseline; SSP1 shown")
    print(f"{'='*72}\n")

    rows = []
    for buf_label, buf_file in [('exact','exact'),('300m','300m'),('600m','600m'),('900m','900m')]:
        path = f"{OUT_DIR}/{code}_spatial_TPFPFN_{buf_file}_20260607.tif"
        with rasterio.open(path) as src:
            d = src.read(1)
        tp=int((d==1).sum()); fp=int((d==2).sum()); fn=int((d==3).sum())
        tn=int(d.size-tp-fp-fn)
        m = compute_all_metrics(tp, fp, fn, tn)
        m['Buffer'] = buf_label
        rows.append(m)
        del d

    hdr = f"  {'Buffer':>7} {'Prec.':>6} {'Recall':>7} {'F1':>6} {'IoU':>6} {'Kappa':>6} {'MCC':>6}   {'FoM':>6} {'QD':>7} {'AD':>7} {'TD':>7} {'OA':>6}"
    print(hdr)
    print("  "+"-"*len(hdr))
    for m in rows:
        print(f"  {m['Buffer']:>7} {m['Precision']:>6.3f} {m['Recall']:>7.3f} {m['F1']:>6.3f} {m['IoU']:>6.3f} {m['Kappa']:>6.3f} {m['MCC']:>6.3f}   {m['FoM']:>6.3f} {m['QD']:>7.4f} {m['AD']:>7.4f} {m['TD']:>7.4f} {m['OA']:>6.3f}")

    csv_path = f"{OUT_DIR}/{code}_extended_metrics_20260612.csv"
    with open(csv_path, 'w', newline='') as f:
        fields = ['Buffer','Precision','Recall','F1','IoU','Kappa','MCC','FoM','QD','AD','TD','OA']
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    print(f"\n  Saved: {csv_path}")

    print(f"  FoM = Figure of Merit (Pontius et al. 2008). QD = Quantity Disagreement.")
    print(f"  AD = Allocation Disagreement. TD = QD + AD. Pontius & Millones (2011).")

print("\nDONE — all extended metrics computed.")
