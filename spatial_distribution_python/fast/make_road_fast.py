"""make_road.py  -- port of make_road.m

Simulates radial + linear road growth and writes the resulting GeoTIFFs and
statistics.  Faithful port of the MATLAB ``make_road`` (including its
variable-persistence behaviour across the per-road-type loop).
"""

import os
import numpy as np

from matlab_compat import (geotiffread, geotiffwrite, load_mat, save_mat,
                           sub2ind, ind2sub, matlab_round, find)
from make_radial_growth_fast import make_radial_growth
from matlab_numerics import matlab_class_div
from get_upgraded_points import get_upgraded_points
from get_grown_roads import get_grown_roads


def _last_is_one(s):
    try:
        return float(s[-1]) == 1
    except (ValueError, IndexError):
        return False


def make_road(input_dir, output_dir, predicted_road_amount_scenario, country_name,
              sel_mode, base_year, skip_radial, use_poi, debug):
    # ------------------------------------------------------------------
    # Configurations
    # ------------------------------------------------------------------
    upgrade_percent = np.array([0.75, 0.65, 0.5, 0])
    res = 300
    search_rad = 150
    factor_2 = 3
    theta_r = 10
    factor_sector_mask = 1 / 4
    radial_r = round(10000 / res)
    randomized = 1

    road_spacing = round(3 * 1000 / res)
    N_iter = 10
    search_radius_for_urban = 0
    min_number_road_pixel = 15

    # ------------------------------------------------------------------
    # Process input data
    # ------------------------------------------------------------------
    temp_dir = None
    if base_year and not _last_is_one(input_dir):
        temp_dir = input_dir[:-1] + "1"

    data_dir = input_dir
    file_list = ["road_types.tif", "tp1.tif", "slope.tif", "dist_urban.tif",
                 "pop_density.tif", "dist_roads.tif", "road_density.tif", "urban.tif"]
    road_density_file = "road_density_total.tif"

    weight = [1, 1, 1, 1, 1, 1, 1]

    urban_center_file = os.path.join(output_dir, f"{country_name}_uc_data.mat")
    poi_file = "poi.tif" if use_poi == 1 else ""
    uc_poi_weight_ratio = 2

    S = load_mat(urban_center_file)
    uc_map = np.asarray(S["urban_center"], dtype=float)

    if poi_file:
        if sel_mode != "dist_only":
            raise ValueError("POI file can be used only with selection mode: dist only")
        src = data_dir if temp_dir is None else temp_dir
        A, _ = geotiffread(os.path.join(src, f"{country_name}_{poi_file}"))
        a, b, _ = find(A)
        poi_map = np.column_stack((a, b)).astype(float)
        uc_map = (uc_map, poi_map, uc_poi_weight_ratio)

    src = data_dir if temp_dir is None else temp_dir
    A, R = geotiffread(os.path.join(src, f"{country_name}_{file_list[0]}"))
    A_road = np.asarray(A).astype(np.uint8)   # integer road codes 0-4 (11-14 after radial)
    assert np.array_equal(A_road, np.asarray(A))
    del A
    N_r, N_c = A_road.shape
    score = np.zeros((N_r, N_c))

    for idx in range(1, len(file_list) - 1):
        if file_list[idx] != "pop_density.tif" and temp_dir is not None:
            src = temp_dir
        else:
            src = data_dir
        A, _ = geotiffread(os.path.join(src, f"{country_name}_{file_list[idx]}"))
        score += matlab_class_div(np.asarray(A), weight[idx])
        del A

    src = data_dir if temp_dir is None else temp_dir
    A, _ = geotiffread(os.path.join(src, f"{country_name}_{file_list[-1]}"))
    score_2 = score * (A_road == 0)
    _t = matlab_class_div(np.asarray(A), 1); del A
    np.subtract(1, _t, out=_t)
    score_2 *= _t
    del _t

    road_types = np.unique(A_road)
    road_types = road_types[1:]  # drop first (background = 0)

    # ------------------------------------------------------------------
    # Make radial growth
    # ------------------------------------------------------------------
    pra = np.asarray(predicted_road_amount_scenario, dtype=float)
    total_length_extend_upgrade = matlab_round(pra * 1000 / 300)
    new_total_length = float(np.sum(total_length_extend_upgrade))

    print("Radial Growth")
    added_road, updated_road, num_new_road, updated_road_1_5 = make_radial_growth(
        (A_road != 0).astype(np.uint8), A_road, uc_map, score, search_rad, new_total_length,
        factor_2, theta_r, factor_sector_mask, radial_r, randomized,
        output_dir, country_name, skip_radial, debug)

    if num_new_road > 0:
        for road_idx in range(1, 5):  # hard assumption: 5 road growth types
            extended_length = (int(np.count_nonzero(updated_road_1_5 == road_idx)) - int(np.count_nonzero(A_road == road_idx)))
            total_length_extend_upgrade[road_idx - 1] = (
                total_length_extend_upgrade[road_idx - 1] - extended_length)

    geotiffwrite(os.path.join(output_dir, f"{country_name}_after_radial.tif"),
                 updated_road_1_5, R)

    # ------------------------------------------------------------------
    A_road = updated_road_1_5
    score_2 *= (A_road == 0)
    del score
    upgrade_length = matlab_round(total_length_extend_upgrade * upgrade_percent)
    upgrade_length[-1] = 0
    growth_length = total_length_extend_upgrade - upgrade_length

    src = data_dir if temp_dir is None else temp_dir
    X, _ = geotiffread(os.path.join(src, f"{country_name}_{road_density_file}"))
    _rd_path = os.path.join(src, f"{country_name}_{road_density_file}")
    del X
    def _road_density():   # loaded when needed; values read through .astype(float) downstream
        return np.asarray(geotiffread(_rd_path)[0], dtype=np.float32)

    min_upgrade_per_intersect = 15
    max_radius = 50

    final_road = A_road.copy()

    remove_iteration = 30
    min_road_sep_1 = 12
    min_road_sep_2 = 12
    score_map = score_2

    growth_road_stats = np.zeros((len(road_types), 7))

    # persistent variables across the loop (matching MATLAB scoping)
    idx_to_be_upgraded = np.array([], dtype=np.int64)
    road_demand = 0.0
    road_grown = 0.0
    added_road_without_preprocess = np.zeros((N_r, N_c))

    for road_idx in range(1, len(road_types) + 1):
        rt = road_types[road_idx - 1]
        print(f"*********************Road type {rt:g}*********************")
        A1 = np.zeros((N_r, N_c), np.uint8)
        A1_and_high = np.zeros((N_r, N_c), np.uint8)
        A2 = np.zeros((N_r, N_c), np.uint8)
        A1[A_road == rt] = 1
        if road_idx == 1:
            A1_and_high = A1.copy()
            if len(road_types) > 1:   # one road type in the domain (e.g. BHR): no lower type, A2 stays empty
                A2[A_road == road_types[road_idx]] = 1
        elif road_idx == len(road_types):
            A1_and_high[(A_road == rt) | (A_road == road_types[road_idx - 2])] = 1
            A2 = None
        else:
            A1_and_high[(A_road == rt) | (A_road == road_types[road_idx - 2])] = 1
            A2[A_road == road_types[road_idx]] = 1

        A_updated_road = A1.copy()

        if (A2 is not None) and upgrade_length[road_idx - 1] > 0:
            to_be_upgraded, _ = get_upgraded_points(
                A1, A2, _road_density(), max_radius,
                upgrade_length[road_idx - 1], min_upgrade_per_intersect)
            if to_be_upgraded.size:
                lin = sub2ind((N_r, N_c), to_be_upgraded[:, 0], to_be_upgraded[:, 1])
                idx_to_be_upgraded = np.unique(lin)
                rr, cc = ind2sub((N_r, N_c), idx_to_be_upgraded)
                A_updated_road[rr - 1, cc - 1] = 2
                print(f"Road types {rt:g} upgraded: {len(idx_to_be_upgraded)}, "
                      f"required: {upgrade_length[road_idx - 1]:g}")
                growth_length[road_idx] = growth_length[road_idx] + len(idx_to_be_upgraded)
                road_demand = (growth_length[road_idx - 1] - len(idx_to_be_upgraded)
                               + upgrade_length[road_idx - 1])
            else:
                print("no feasible intersection points for upgrade")
                print("Warning: this case is treated the same way we handle for "
                      "zero upgrade length / the highest order road")
        else:
            road_demand = growth_length[road_idx - 1]
            print("no need to upgrade")

        # grow roads
        if rt == 10:
            start_road = A1
        else:
            start_road = A1_and_high

        if road_demand != 0:
            processed_road_2, new_rd, road_grown, added_road_without_preprocess = get_grown_roads(
                remove_iteration, min_road_sep_1, min_road_sep_2, start_road,
                final_road != 0, uc_map, score_map, road_demand, road_spacing,
                search_radius_for_urban, min_number_road_pixel, sel_mode, N_iter, debug)
            del processed_road_2
        else:
            new_rd = np.zeros_like(start_road)

        score_map[new_rd == 1] = 0
        print(f"Road types {rt:g} grown: {road_grown:g}, "
              f"required: {growth_length[road_idx - 1]:g}, adjusted:{road_demand:g}")
        if road_demand > road_grown:
            print("Warning: cannot meet the required growth")

        A_updated_road_dummy = A_updated_road.copy()
        A_updated_road_dummy[added_road_without_preprocess == 1] = 3
        A_updated_road[new_rd == 1] = 3

        growth_road_stats[road_idx - 1, 0] = rt
        growth_road_stats[road_idx - 1, 1] = len(idx_to_be_upgraded)
        growth_road_stats[road_idx - 1, 2] = upgrade_length[road_idx - 1]
        growth_road_stats[road_idx - 1, 3] = road_grown
        growth_road_stats[road_idx - 1, 4] = growth_length[road_idx - 1]
        growth_road_stats[road_idx - 1, 5] = road_demand
        growth_road_stats[road_idx - 1, 6] = (total_length_extend_upgrade[road_idx - 1]
                                              - len(idx_to_be_upgraded) - road_grown)

        geotiffwrite(os.path.join(output_dir, f"{country_name}_type_{rt:g}_before_process.tif"),
                     A_updated_road, R)
        geotiffwrite(os.path.join(output_dir, f"{country_name}_type_{rt:g}.tif"),
                     A_updated_road_dummy, R)

        final_road[(A_updated_road != 0) & ((final_road == 0) | ((final_road > rt) & (final_road < 10)))] = rt

    geotiffwrite(os.path.join(output_dir, f"{country_name}_all_final_roads.tif"),
                 final_road, R)
    save_mat(os.path.join(output_dir, f"{country_name}_upgrade_and_growth_info.mat"),
             growth_road_stats=growth_road_stats)
