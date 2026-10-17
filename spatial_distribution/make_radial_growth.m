function [added_road,updated_road,num_new_road, A_road_update] = make_radial_growth(A_road, A_road_1_5,uc_map, score, ...
                                        search_rad,new_total_length, ...
                                factor_2,theta_r,factor_sector_mask,radial_r,randomized,...
                                output_dir, country_name,skip_radial,debug)
    A_road = full(A_road);
    [N_r, N_c] = size(A_road);
    A_road_update = A_road_1_5;
    if skip_radial == 0

        if iscell(uc_map)
            uc_map = uc_map{1};
        end

        [~, sorted_idx] = sort(uc_map(:,3),'descend');

        R_ind = [];
        R_s = [];

        C_ind = [];
        C_s = [];

        candidate_node_list = cell(0);
        current_px_count = [];
        if debug
            sorted_idx = sorted_idx(1:2);
        end
        for uc_idx=sorted_idx'
            temp = uc_map(uc_idx,:);
            cx = temp(1);
            cy = temp(2);

            ROI_r = round(max(1,cx-search_rad):min(N_r,cx+search_rad));
            ROI_c = round(max(1,cy-search_rad):min(N_c,cy+search_rad));

            cropped_A = A_road(ROI_r,ROI_c);
            cropped_score = score(ROI_r, ROI_c);
            [current_radial,candidate_node] = get_radial_info(cropped_A,cropped_score,radial_r,theta_r,factor_sector_mask , 0);

            if ~isempty(current_radial)
                ROI_r = ROI_r - 1;
                ROI_c = ROI_c - 1;
                candidate_node_list = [candidate_node_list, cellfun(@(x) x+ [ROI_r(1), ROI_c(1)], candidate_node(3,:), 'UniformOutput', false)];

                radial_adjusted_subs = reshape([current_radial{1,:}],2,[])'+ [ROI_r(1), ROI_c(1)];
                radial_stats = [current_radial{2,:}];
                current_px_count = [current_px_count, current_radial{3,:}];

                cand_adjusted_subs = reshape([candidate_node{1,:}],2,[])'+ [ROI_r(1), ROI_c(1)];
                cand_stats = [candidate_node{2,:}];
                R_ind = [R_ind; sub2ind([N_r, N_c],radial_adjusted_subs(:,1), radial_adjusted_subs(:,2))];
                R_s = [R_s,radial_stats ];

                C_ind = [C_ind; sub2ind([N_r, N_c],cand_adjusted_subs(:,1), cand_adjusted_subs(:,2))];
                C_s = [C_s,cand_stats ];
            end

        end
        total_len = sum(sum(A_road));
        [~, id] = unique(R_ind);
        current_px_count = current_px_count(id);

        new_radial = round(length(id)*(new_total_length/total_len));
        new_radial_length = round(sum(current_px_count)*(new_total_length/total_len));
        disp(['Current number of radial centers:', num2str(length(id))])
        disp(['Current length of radial centers:', num2str(sum(current_px_count))])

        disp(['Estimated number of new radial centers:', num2str(new_radial)])
        disp(['Estimated length of new radial centers:', num2str(new_radial_length)])

        disp(['Length of existing roads:', num2str(total_len)])
        disp(['Length of new additional roads to grow:', num2str(new_total_length)])

        [~, id] = unique(C_ind);
        C_s = C_s(id);

        [~, sorted_idx] = sort(C_s,'descend');
        picked_cand = id(sorted_idx);
        if randomized
            picked_cand = picked_cand(randperm(length(picked_cand)));
        end

        path_list = [];
        [A1, A2] = ind2sub([N_r, N_c], C_ind(picked_cand));
        valid_cand = zeros(size(picked_cand));
        radial_left = new_radial;
        radial_left = new_radial_length;
        disp(['Creating ', num2str(new_radial), ' radial centers'])

        for grow_idx = 1:length(picked_cand)
            if radial_left<=0
                break
            end
            if valid_cand(grow_idx)
                continue
            else

                a1 = A1(grow_idx);
                a2 = A2(grow_idx);
                road_type = A_road_1_5(a1,a2);
                if road_type == 0
                    error('type 0!!')
                end

                end_idx = candidate_node_list{picked_cand(grow_idx)};
                r_max_squared = max(sum((end_idx - [a1, a2]).^2,2));
                valid_cand(grow_idx+1:end) = valid_cand(grow_idx+1:end) + (sum(([A1(grow_idx+1:end,:), A2(grow_idx+1:end,:)]-[a1, a2]).^2,2) <= (factor_2)^2*r_max_squared);

                for idx=1:numel(candidate_node_list{picked_cand(grow_idx)})/2
                    x = get_next_best_cand_radial([a1, a2], end_idx(idx,:), score);
                    path_list = [path_list;x];
                    A_road_update(sub2ind([N_r, N_c], x(:,1), x(:,2))) = road_type+10;

                end

                radial_left = new_radial_length - sum(sum(A_road_update~=0)) + sum(sum(A_road_1_5~=0));
            end
        end

        if radial_left>0
            disp([num2str(radial_left),' radial lengths cannot be added'])
        end
    else
        path_list = [];
    end

    if ~isempty(path_list)
        added_road = zeros(N_r, N_c);
        added_road(sub2ind([N_r, N_c], path_list(:,1), path_list(:,2))) = 1;

        updated_road = A_road;
        updated_road(sub2ind([N_r, N_c], path_list(:,1), path_list(:,2))) = 1;
        num_new_road = sum(sum(updated_road)) - sum(sum(A_road));

        save_filename = [output_dir, filesep, country_name,'_radial_current_and_new.mat'];
        save(save_filename, 'current_px_count','num_new_road');

    else
        disp('no new road is added')
        added_road = zeros(N_r, N_c);
        updated_road = A_road;
        num_new_road = 0;
    end

end
