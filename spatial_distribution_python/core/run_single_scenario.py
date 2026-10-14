"""run_single_scenario.py  -- port of run_single_scenario.m

Runs the urban clustering script followed by the road growth script.

Usage (CLI):
    python run_single_scenario.py \
        --input_dir DIR --output_dir DIR \
        --predicted_road_amount 142.89,181.5,20.21,25 \
        --country_name NAME [--sel_mode dist_only] \
        --base_year 0 --skip_radial 0 --use_poi 0 --debug 0 [--seed 42]

Seed behaviour: if --seed is given the run is deterministic; otherwise the
random ordering steps behave randomly (matching MATLAB's randomized=1).
"""

import argparse

from matlab_compat import set_seed
from make_urban_clusters import make_urban_clusters
from make_road import make_road


def run_single_scenario(input_dir, output_dir, predicted_road_amount_scenario,
                        country_name, sel_mode, base_year, skip_radial, use_poi,
                        debug, seed=None):
    set_seed(seed)

    print("Making urban clusters: starting")
    make_urban_clusters(input_dir, output_dir, country_name, base_year)
    print("Making urban clusters: completed")

    print("Growing roads: starting")
    make_road(input_dir, output_dir, predicted_road_amount_scenario, country_name,
              sel_mode, base_year, skip_radial, use_poi, debug)
    print("Growing roads: completed")


def _parse_args():
    p = argparse.ArgumentParser(description="Road growth single scenario (Python port)")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--predicted_road_amount", required=True,
                   help="comma-separated list, e.g. 142.89,181.5,20.21,25")
    p.add_argument("--country_name", required=True)
    p.add_argument("--sel_mode", default="dist_only")
    p.add_argument("--base_year", type=int, default=0)
    p.add_argument("--skip_radial", type=int, default=0)
    p.add_argument("--use_poi", type=int, default=0)
    p.add_argument("--debug", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pra = [float(x) for x in args.predicted_road_amount.split(",")]
    run_single_scenario(args.input_dir, args.output_dir, pra, args.country_name,
                        args.sel_mode, args.base_year, args.skip_radial,
                        args.use_poi, args.debug, args.seed)
