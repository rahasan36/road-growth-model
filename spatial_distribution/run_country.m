function run_country(country_name, scenario_list, prediction_years, inputs_dir, demand_file, output_root)

    if nargin < 2 || isempty(scenario_list)
        scenario_list = [1, 3, 5];
    end
    if nargin < 3 || isempty(prediction_years)
        prediction_years = [2030, 2040, 2050];
    end
    if nargin < 4 || isempty(inputs_dir)
        inputs_dir = fullfile('..', 'data', 'model_inputs');
    end
    if nargin < 5 || isempty(demand_file)
        demand_file = fullfile('..', 'data', 'road_demand_published.csv');
    end
    if nargin < 6 || isempty(output_root)
        output_root = 'output_data';
    end

    sel_mode = 'dist_only';
    base_year = 0;
    skip_radial = 0;
    use_poi = 0;
    debug = 0;
    seed = 42;

    here = fileparts(mfilename('fullpath'));
    addpath(here);

    demand = readtable(demand_file, 'TextType', 'string');

    for scenario_type = scenario_list
        for prediction_year = prediction_years
            ssp = "SSP" + string(scenario_type);
            row = demand.country == string(country_name) & demand.ssp == ssp & demand.year == prediction_year;
            if sum(row) ~= 1
                error('No road demand for %s %s %d in %s', country_name, ssp, prediction_year, demand_file);
            end
            predicted_road_amount_scenario = [demand.main_km(row), demand.secondary_km(row), ...
                demand.tertiary_km(row), demand.local_km(row)];

            input_dir = fullfile(inputs_dir, country_name, char(ssp), num2str(prediction_year));
            if ~exist(input_dir, 'dir')
                error('Input folder %s not found', input_dir);
            end
            output_dir = fullfile(output_root, country_name, char(ssp), num2str(prediction_year));
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end

            fprintf('%s %s %d: %.1f %.1f %.1f %.1f km\n', country_name, ssp, prediction_year, predicted_road_amount_scenario);
            rng(seed);
            run_single_scenario(input_dir, output_dir, predicted_road_amount_scenario, country_name, ...
                sel_mode, base_year, skip_radial, use_poi, debug);
        end
    end

end
