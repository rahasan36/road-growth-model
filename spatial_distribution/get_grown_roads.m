function [processed_road_2, new_rd, road_grown, added_road] = get_grown_roads(remove_iteration,min_road_sep_1, min_road_sep_2,start_road,...
                                  all_high_roads,uc_map,score_map,road_demand, road_spacing, search_radius_for_urban ,...
                                  min_number_road_pixel,sel_mode, N_iter,debug)

    A1 = start_road;
    score_2 = score_map;
    [N_r, N_c] = size(A1);
    road_grown = 0;
    old_cand_map = [];
    path_list = [];
    old_path_length = [];
    for remove_iter=1:remove_iteration

        [added_road,updated_road,num_new_road, path_list,old_path_length, old_cand_map,~] = make_linear_growth(start_road, uc_map, score_map, ...
                                            road_demand-road_grown,road_spacing, search_radius_for_urban, ...
                                            min_number_road_pixel,sel_mode, N_iter,...
                                            old_cand_map,path_list,old_path_length,debug);

        P = path_list;
        nn_dist = zeros(1, length(P));
        idx_removed_paths = [];
        for idx=1:length(P)
            if ~isempty(P{idx})
                road_idx_mask = reshape(P{idx},2,[])';

                bd_up_limit = min(max(road_idx_mask)+min_road_sep_1,[N_r, N_c]);
                bd_low_limit = max(min(road_idx_mask)-min_road_sep_1,[1,1]);

                A_temp = all_high_roads(bd_low_limit(1):bd_up_limit(1),bd_low_limit(2):bd_up_limit(2));

                nearest_rd = find(A_temp);

                [x,y] = size(A_temp);
                [nearest_rdx, nearest_rdy] = ind2sub([x,y],nearest_rd);
                avg_dist_nn = mean(pdist2([nearest_rdx,nearest_rdy]+bd_low_limit-1, road_idx_mask, 'euclidean','Smallest',1));
                nn_dist(idx) = avg_dist_nn;

                if avg_dist_nn<min_road_sep_1
                    path_list{idx} = [];
                    old_path_length(idx) = 0;
                    idx_removed_paths = [idx_removed_paths, idx];
                end
            end

        end

        P = path_list;

        for idx=1:length(P)-1
            for idx2 = idx+1:length(P)
                if ~isempty(P{idx}) && ~isempty(P{idx2})
                    if length(P{idx})> length(P{idx2})
                        b = idx;
                        a = idx2;
                    else
                        a = idx;
                        b = idx2;
                    end
                    road_idx_mask = reshape(P{a},2,[])';
                    road_idx2_mask = reshape(P{b},2,[])';

                    avg_dist_nn = mean(pdist2(road_idx2_mask, road_idx_mask, 'euclidean','Smallest',1));
                    if avg_dist_nn<min_road_sep_2
                        P{a} = [];
                        path_list{a} = [];
                        old_path_length(a) = 0;
                        idx_removed_paths = [idx_removed_paths, a];
                    end
                end

            end

        end

        new_rd = zeros(N_r, N_c);
        for idx=1:length(P)
            if ~isempty(P{idx})
                for idx1 = 1:2:length(P{idx})-1
                    coord= P{idx}(idx1: idx1+1);
                    new_rd(coord(1),coord(2)) = 1;
                end
            end
        end

        processed_road_2 = A1+new_rd;
        updated_score = score_2.*(1-processed_road_2);

        for idx=idx_removed_paths
            old_cand_map(old_cand_map(:,4)==idx,3) = 0;
        end

        for idx=1:length(path_list)
            if ~isempty(path_list{idx})
                temp_mask = old_cand_map(:,4) == idx;
                temp_info = old_cand_map(temp_mask,:);
                old_cand_map(temp_mask,3) = updated_score(temp_info(1),temp_info(2));
            end
        end

        road_grown = sum(sum(processed_road_2)) - sum(sum(A1));
        start_road = processed_road_2;
        score_map = updated_score;

        disp(['road grown: ',num2str(road_grown), ' iteration: ', num2str(remove_iter)])

        if road_grown >= road_demand
            disp('stopping growth process')
            break
        end

        if max(old_cand_map(:,3))<= 0
            warning('max score becomes zero; terminating the growth process')
            break
        end

        if sum(old_cand_map(:,1)==0) || sum(old_cand_map(:,2)==0)
            warning('candidate index is zero')
        end
    end

end
