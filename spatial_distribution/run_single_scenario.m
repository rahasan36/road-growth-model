function run_single_scenario(input_dir,output_dir, predicted_road_amount_scenario, country_name,...
                            sel_mode, base_year, skip_radial, use_poi, debug)

disp('Making urban clusters: starting')
make_urban_clusters(input_dir,output_dir,country_name, base_year)
disp('Making urban clusters: completed')

disp('Growing roads: starting')
make_road(input_dir,output_dir, predicted_road_amount_scenario, country_name, sel_mode, ...
        base_year, skip_radial, use_poi, debug)
disp('Growing roads: completed')

end
