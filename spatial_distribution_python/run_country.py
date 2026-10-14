import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "core"))
sys.path.insert(0, os.path.join(HERE, "fast"))
from model import run_single_scenario

REPO = os.path.dirname(HERE)


def run_country(country, ssps=(1, 3, 5), years=(2030, 2040, 2050),
                inputs_dir=os.path.join(REPO, "data", "model_inputs"),
                demand_file=os.path.join(REPO, "data", "road_demand_published.csv"),
                output_root=os.path.join(HERE, "output_data"), seed=42):
    demand = pd.read_csv(demand_file)
    for s in ssps:
        for y in years:
            row = demand[(demand.country == country) & (demand.ssp == f"SSP{s}") & (demand.year == y)]
            if len(row) != 1:
                raise SystemExit(f"No road demand for {country} SSP{s} {y} in {demand_file}")
            km = row[["main_km", "secondary_km", "tertiary_km", "local_km"]].iloc[0].astype(float).tolist()
            input_dir = os.path.join(inputs_dir, country, f"SSP{s}", str(y))
            if not os.path.isdir(input_dir):
                raise SystemExit(f"Input folder {input_dir} not found")
            output_dir = os.path.join(output_root, country, f"SSP{s}", str(y))
            os.makedirs(output_dir, exist_ok=True)
            print(f"{country} SSP{s} {y}: " + " ".join(f"{v:.1f}" for v in km) + " km", flush=True)
            run_single_scenario(input_dir, output_dir, km, country, "dist_only", 0, 0, 0, 0, seed=seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the spatial allocation model for one country.")
    ap.add_argument("country")
    ap.add_argument("--ssp", default="1,3,5")
    ap.add_argument("--years", default="2030,2040,2050")
    ap.add_argument("--inputs_dir", default=os.path.join(REPO, "data", "model_inputs"))
    ap.add_argument("--demand_file", default=os.path.join(REPO, "data", "road_demand_published.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "output_data"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run_country(a.country, [int(x) for x in a.ssp.split(",")], [int(x) for x in a.years.split(",")],
                a.inputs_dir, a.demand_file, a.out, a.seed)
