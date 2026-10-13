function make_road(input_dir,output_dir, predicted_road_amount_scenario, country_name,...
                    sel_mode,base_year, skip_radial, use_poi, debug)

    upgrade_percent = [0.75 0.65 0.5 0];
    res = 300;
    search_rad = 150;
    factor_2 = 3;
    theta_r = 10;
    factor_sector_mask = 1/4;
    radial_r = round(10000/res);
    randomized = 1;

    road_spacing = round(3*1000/res);
    N_iter = 10;
    search_radius_for_urban = 0;
    min_number_road_pixel = 15;

    temp_dir = [];
    if base_year && str2num(input_dir(end))~=1
        temp_dir = input_dir;
        temp_dir(end) = '1';
    end

    data_dir = input_dir;
    file_list = {'road_types.tif', 'tp1.tif','slope.tif','dist_urban.tif',...
        'pop_density.tif','dist_roads.tif','road_density.tif','urban.tif'};
    road_density_file = 'road_density_total.tif';

    weight = [1,1,1,1,1,1,1];

    urban_center_file = [output_dir, filesep, country_name,'_uc_data.mat'];
    if use_poi==1
        poi_file = 'poi.tif';
    else
        poi_file = '';
    end

    uc_poi_weight_ratio = 2;

    S = load(urban_center_file);
    uc_map = S.urban_center;

    if ~isempty(poi_file)
        if ~strcmp(sel_mode, 'dist_only')
            error 'POI file can be used only with selection mode: dist only'
        end
        if isempty(temp_dir)
            [A,~] = geotiffread([data_dir,filesep,country_name,'_', poi_file]);
        else
            [A,~] = geotiffread([temp_dir,filesep,country_name,'_', poi_file]);
        end

        [a,b] = find(A);
        poi_map = [a,b];

        uc_map = {uc_map, poi_map, uc_poi_weight_ratio};
    end

    if isempty(temp_dir)
        [A,R] = geotiffread([data_dir,filesep,country_name,'_', file_list{1}]);
    else
        [A,R] = geotiffread([temp_dir,filesep,country_name,'_', file_list{1}]);
    end

    A = double(A);
    A_road = A;
    A_init_road = A_road;
    [N_r, N_c] = size(A_road);
    score = sparse(zeros(size(A_road)));

    for idx=2:length(file_list)-1
        if ~strcmp(file_list{idx} , 'pop_density.tif') && ~isempty(temp_dir)
            [A,~] = geotiffread( [temp_dir,filesep,country_name,'_',file_list{idx}]);
        else
            [A,~] = geotiffread( [data_dir,filesep,country_name,'_',file_list{idx}]);
        end

        score = score + double(A*weight(idx)/max(max(A)));
    end

    if isempty(temp_dir)
        [A,~] = geotiffread( [data_dir,filesep,country_name,'_',file_list{end}]);
    else
        [A,~] = geotiffread( [temp_dir,filesep,country_name,'_',file_list{end}]);
    end

    score_2 = score.*(A_road==0).*(1-double(A/max(max(A))));

    road_types = unique(A_road);
    road_types = road_types(2:end);
    if isempty(temp_dir)
        info = geotiffinfo([data_dir,filesep,country_name,'_',file_list{1}]);
    else
        info = geotiffinfo([temp_dir,filesep,country_name,'_',file_list{1}]);
    end

    total_length_extend_upgrade = predicted_road_amount_scenario*1000/(300);
    total_length_extend_upgrade = round(total_length_extend_upgrade);

    new_total_length = sum(total_length_extend_upgrade);

    disp('Radial Growth')
    [added_road,updated_road,num_new_road, updated_road_1_5] = make_radial_growth((A_road~=0)*1.0,A_road,uc_map, score, ...
                                            search_rad,new_total_length, factor_2,...
                                            theta_r,factor_sector_mask,radial_r,randomized,...
                                            output_dir, country_name,skip_radial,debug);
    if num_new_road>0
        for road_idx = 1:4
            extended_length = sum(sum(updated_road_1_5==road_idx)) - sum(sum(A_road==road_idx));
            total_length_extend_upgrade(road_idx) = total_length_extend_upgrade(road_idx) - extended_length;
        end
    end

    info.GeoTIFFTags.GeoKeyDirectoryTag.GTModelTypeGeoKey = 1;
    info.GeoTIFFTags.GeoKeyDirectoryTag.ProjectedCSTypeGeoKey = 32767;
    geotiffwrite([output_dir,filesep,country_name,'_after_radial.tif'],single(full(updated_road_1_5)),R,'GeoKeyDirectoryTag' ,info.GeoTIFFTags.GeoKeyDirectoryTag)

    A_road = updated_road_1_5;
    score_2 = score_2.*(A_road==0);
    upgrade_length = round(total_length_extend_upgrade.*upgrade_percent);
    upgrade_length(end) = 0;
    growth_length = total_length_extend_upgrade - upgrade_length;

    if isempty(temp_dir)
        [X,~] = geotiffread([data_dir,filesep,country_name,'_',road_density_file]);
    else
        [X,~] = geotiffread([temp_dir,filesep,country_name,'_',road_density_file]);
    end

    road_density_map = double(X);
    min_upgrade_per_intersect = 15;
    max_radius = 50;

    final_road = A_road;

    remove_iteration = 30;
    min_road_sep_1 = 12;
    min_road_sep_2 = 12;
    score_map = score_2;

    growth_road_stats = zeros(length(road_types), 7);

    for road_idx = 1:length(road_types)
        disp(['*********************Road type ',num2str(road_types(road_idx)),'*********************'])
        A1 = zeros(size(A_road));
        A1_and_high = A1;
        A2 = A1;
        A1(A_road==road_types(road_idx) ) = 1;
        if road_idx==1
            A1_and_high = A1;
            A2(A_road==road_types(road_idx+1)) = 1;

        elseif road_idx==length(road_types)
            A1_and_high(A_road==road_types(road_idx) | A_road==road_types(road_idx-1)) = 1;
            A2 = [];
        else
            A1_and_high(A_road==road_types(road_idx) | A_road==road_types(road_idx-1)) = 1;
            A2(A_road==road_types(road_idx+1)) = 1;
        end
        A_updated_road = A1;
        if ~isempty(A2) && upgrade_length(road_idx)>0
            [to_be_upgraded, ~] = get_upgraded_points(A1, A2, road_density_map,max_radius,upgrade_length(road_idx),min_upgrade_per_intersect);
            if ~isempty(to_be_upgraded)
                idx_to_be_upgraded = unique(sub2ind([N_r, N_c], to_be_upgraded(:,1), to_be_upgraded(:,2)));
                A_updated_road(idx_to_be_upgraded) = 2;
                disp(['Road types ',num2str(road_types(road_idx)),...
                    ' upgraded: ',num2str(length(idx_to_be_upgraded)), ', required: ', num2str(upgrade_length(road_idx))])
                growth_length(road_idx+1) = growth_length(road_idx+1) + length(idx_to_be_upgraded);
                road_demand = growth_length(road_idx) - length(idx_to_be_upgraded)+ upgrade_length(road_idx);

            else
                disp('no feasible intersection points for upgrade')
                warning('this case is treated the same way we handle for zero upgrade length/ the highest order road')
            end

        else
            road_demand = growth_length(road_idx);
            disp('no need to upgrade')
        end

        if road_types(road_idx) == 10
            start_road = A1;
        else
            start_road = A1_and_high;
        end

        if road_demand ~=0
            [processed_road_2, new_rd, road_grown, added_road_without_preprocess] = get_grown_roads(remove_iteration,min_road_sep_1, min_road_sep_2,start_road,...
                                      final_road~=0,uc_map, score_map,road_demand, road_spacing, search_radius_for_urban, ...
                                      min_number_road_pixel,sel_mode, N_iter,debug);

        else

            new_rd = zeros(size(start_road));

        end

        score_map(new_rd == 1) = 0;
        disp(['Road types ',num2str(road_types(road_idx)),...
            ' grown: ',num2str(road_grown), ', required: ', num2str(growth_length(road_idx)), ', adjusted:', num2str(road_demand)])
        if road_demand > road_grown
            warning('cannot meet the required growth')
        end

        A_updated_road_dummy = A_updated_road;
        A_updated_road_dummy(added_road_without_preprocess ==1) = 3;

        A_updated_road(new_rd==1) = 3;

        growth_road_stats(road_idx,1) = road_types(road_idx);
        growth_road_stats(road_idx,2) = length(idx_to_be_upgraded);
        growth_road_stats(road_idx,3) = upgrade_length(road_idx);
        growth_road_stats(road_idx,4) = road_grown;
        growth_road_stats(road_idx,5) = growth_length(road_idx);
        growth_road_stats(road_idx,6) = road_demand;
        growth_road_stats(road_idx,7) = total_length_extend_upgrade(road_idx) - length(idx_to_be_upgraded) - road_grown ;

        geotiffwrite([output_dir,filesep,country_name,'_type_',num2str(road_types(road_idx)),'_before_process.tif'],single(full(A_updated_road)),R,'GeoKeyDirectoryTag' ,info.GeoTIFFTags.GeoKeyDirectoryTag)
        geotiffwrite([output_dir,filesep,country_name,'_type_',num2str(road_types(road_idx)),'.tif'],single(full(A_updated_road_dummy)),R,'GeoKeyDirectoryTag' ,info.GeoTIFFTags.GeoKeyDirectoryTag)

        to_set = A_updated_road~=0 & (final_road==0 | (final_road>road_types(road_idx) & final_road<10));
        final_road(to_set) = road_types(road_idx);

    end

    geotiffwrite([output_dir,filesep,country_name,'_all_final_roads.tif'],single(full(final_road)),R,'GeoKeyDirectoryTag' ,info.GeoTIFFTags.GeoKeyDirectoryTag)

    save([output_dir,filesep,country_name,'_upgrade_and_growth_info.mat'], 'growth_road_stats');

end
