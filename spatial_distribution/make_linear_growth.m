function [added_road,updated_road,num_new_road, path_list, path_length,cand_map, score] = make_linear_growth(road_data, uc_map, score, ...
                                        new_total_length, road_spacing, search_radius_for_urban, ...
                                        min_number_road_pixel,sel_mode, N_iter, old_cand_map, old_path_list,old_path_length,debug)

    [N_r, N_c] = size(road_data);
    if iscell(uc_map)
        uc_map_1 = uc_map{1};
    else
        uc_map_1 = uc_map;
    end

    if isempty(old_cand_map)
        disp('Generating candidates')
        cand_A = 1*(conv2(full(road_data), ones(3),'same')>=1);

        cand_A = cand_A - road_data;

        [a,b,v] = find(sparse(score.*cand_A));
        cand_map = zeros(length(a),4);
        cand_map(:,1:2) = [a,b];
        cand_map(:,3) = v;
        sparse_cand_map = sparse(cand_map(:,1), cand_map(:,2),1,N_r,N_c);

        uc_sp = sparse(round(uc_map_1(:,1)), round(uc_map_1(:,2)), uc_map_1(:,3), N_r, N_c);

        cand_map_filtered = cand_map;
        if debug
            cand_map = cand_map(1:100,:,:);
        end
        disp(['Number of initial candidates: ',num2str(length(cand_map_filtered))])

        for idx=1:length(cand_map)
            [~, ~, val] = get_next_best_cand(cand_map(idx,:), uc_map, score, sparse_cand_map,search_radius_for_urban ,sel_mode);
            if val==0
                cand_map_filtered(idx,3) = 0;
            end
        end

        path_list = {};
        path_length = [];
        path_num = 0;
    else
        cand_map_filtered = old_cand_map;
        path_list = old_path_list;
        path_length = old_path_length;
        path_num = length(path_list);
    end

    cand_map = cand_map_filtered;
    sparse_cand_map = sparse(cand_map_filtered(:,1), cand_map_filtered(:,2),1*(cand_map_filtered(:,3)~=0),N_r,N_c);

    if debug==0
        N_iter = new_total_length;
    end
    for idx=1:N_iter
        [v_max, id_max] = max(cand_map(:,3));
        if v_max == 0
            disp('no more candidates available')
            break
        end
        if cand_map(id_max,4)==0
            path_num = path_num+1;
            path_list = [path_list,cand_map(id_max,1:2)];
            path_length = [path_length, 1];
            cand_map(id_max,4) = path_num;
            score(cand_map(id_max,1), cand_map(id_max,2)) = 0;
            dist_mask = sqrt(sum((cand_map(id_max,1:2)-cand_map(:,1:2)).^2,2))<=road_spacing & cand_map(:,4)==0 ;
            cand_map(dist_mask,3) = 0 ;
            [cx, cy, val] = get_next_best_cand(cand_map(id_max,:), uc_map, score, sparse_cand_map,search_radius_for_urban,sel_mode);
            cand_map(id_max,1:3) = [cx,cy,val];
        else
            path_list{cand_map(id_max,4)} = [path_list{cand_map(id_max,4)},cand_map(id_max,1:2)];
            path_length(cand_map(id_max,4)) = path_length(cand_map(id_max,4))+1;
            score(cand_map(id_max,1), cand_map(id_max,2)) = 0;
            dist_mask = sqrt(sum((cand_map(id_max,1:2)-cand_map(:,1:2)).^2,2))<=road_spacing & cand_map(:,4)==0 ;
            cand_map(dist_mask,3) = 0 ;
            [cx, cy, val] = get_next_best_cand(cand_map(id_max,:), uc_map, score, sparse_cand_map,search_radius_for_urban,sel_mode);
            cand_map(id_max,1:3) = [cx,cy,val];
        end

        if sum(path_length) - sum(sum(road_data)) >= new_total_length
            disp('path limit reached')
            break
        end
    end

    disp(['Total increased road length: ', num2str(length([path_list{:}])/2)])

    path_left = sum(path_length(path_length<min_number_road_pixel));

    road_idx = find(path_length<min_number_road_pixel);

    disp(['length of road with less than ', num2str(min_number_road_pixel),' pixels: ', num2str(path_left)]);
    for idx = road_idx

        if path_left > 0
            id_next = find(cand_map(:,4)==idx);
            if cand_map(id_next,3) == 0
                path_list{idx} = [];
                path_length(idx) = 0;

            else

                for idx2 = 1:path_left
                    path_list{idx} = [path_list{idx},cand_map(id_next,1:2)];
                    score(cand_map(id_next,1), cand_map(id_next,2)) = 0;
                    dist_mask = sqrt(sum((cand_map(id_next,1:2)-cand_map(:,1:2)).^2,2))<=road_spacing & cand_map(:,4)==0 ;
                    cand_map(dist_mask,3) = 0 ;
                    [cx, cy, val] = get_next_best_cand(cand_map(id_next,:), uc_map, score, sparse_cand_map,search_radius_for_urban,sel_mode);
                    cand_map(id_next,1:3) = [cx,cy,val];
                    if val == 0
                        break
                    end
                end
                if path_length(idx)+idx2>=min_number_road_pixel
                    path_length(idx) = path_length(idx)+idx2;
                    path_left = path_left - path_length(idx);
                else
                    path_list{idx} = [];
                    path_length(idx) = 0;
                    cand_map(cand_map(:,4) == idx,3) = 0;
                end
            end

        else
            path_list{idx} = [];
            path_length(idx) = 0;
            cand_map(cand_map(:,4) == idx,3) = 0;
        end
    end

    disp([num2str(path_left),'road left to be added'])
    disp(['Total increased road length: ', num2str(length([path_list{:}])/2)])

    new_rd = sparse(zeros(N_r, N_c));
    for idx=1:length(path_list)
        if ~isempty(path_list{idx})
            for idx1 = 1:2:length(path_list{idx})-1
                coord= path_list{idx}(idx1: idx1+1);
                new_rd(coord(1),coord(2)) = 1;
            end
        end
    end

    added_road = new_rd;
    updated_road = new_rd+road_data;
    updated_road = (updated_road~=0)*1.0;
    num_new_road = sum(sum(updated_road)) - sum(sum(road_data));

end
