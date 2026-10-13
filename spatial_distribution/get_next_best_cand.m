function [cx, cy, val] = get_next_best_cand(cand, uc_map, score, sparse_cand_map,search_radius,sel_mode)

    [N_r, N_c] = size(score);
    [ROI_r, ROI_c] = get_adjusted_ROI(cand(1), cand(2),1,N_r,1,N_c);

    if strcmp(sel_mode, 'dist_only')
        if iscell(uc_map)
            uc_map_temp = uc_map{1};
            [nn_uc_dist, nearest_uc] = min(sum((uc_map_temp(:,1:2) - cand(1:2)).^2,2));

            poi_map_temp = uc_map{2};
            [nn_poi_dist, nearest_poi] = min(sum((poi_map_temp(:,1:2) - cand(1:2)).^2,2));

            if nn_uc_dist/uc_map{3} > nn_poi_dist
                if search_radius==0
                    nn_uc_dist = nn_poi_dist;
                    sel_uc = poi_map_temp(nearest_poi,1:2);
                elseif nn_poi_dist <= search_radius^2
                    nn_uc_dist = nn_poi_dist;
                    sel_uc = poi_map_temp(nearest_poi,1:2);
                end
            else
                sel_uc = uc_map_temp(nearest_uc,1:2);
            end

        else
            [nn_uc_dist, nearest_uc] = min(sum((uc_map(:,1:2) - cand(1:2)).^2,2));
        end
    elseif strcmp(sel_mode,'dist_pop_combo')
        dist_from_current = max(sum((uc_map(:,1:2) - cand(1:2)).^2,2),1e-3);
        weights = uc_map(:,3)./dist_from_current;
        if search_radius~=0
            weights(dist_from_current>search_radius^2) = 0;
        end
        [~, sorted_idx] = sort(weights,'descend');
        if weights(sorted_idx(1)) - weights(sorted_idx(2)) <= 0.05*weights(sorted_idx(1))
            [~, sel_idx] = min([dist_from_current(sorted_idx(1:2))]);
            nearest_uc = sorted_idx(sel_idx);
        else
            nearest_uc = sorted_idx(1);
        end
        nn_uc_dist = dist_from_current(nearest_uc);

    elseif strcmp(sel_mode,'dist_height_combo')
        dist_from_current = max(sum((uc_map(:,1:2) - cand(1:2)).^2,2),1e-3);
        weights = uc_map(:,5)./dist_from_current;
        if search_radius~=0
            weights(dist_from_current>search_radius^2) = 0;
        end
        [~, sorted_idx] = sort(weights,'descend');
        if weights(sorted_idx(1)) - weights(sorted_idx(2)) <= 0.05*weights(sorted_idx(1))
            [~, sel_idx] = min([dist_from_current(sorted_idx(1:2))]);
            nearest_uc = sorted_idx(sel_idx);
        else
            nearest_uc = sorted_idx(1);
        end
        nn_uc_dist = dist_from_current(nearest_uc);

    elseif strcmp(sel_mode,'dist_pop_height_combo')
        dist_from_current = max(sum((uc_map(:,1:2) - cand(1:2)).^2,2),1e-3);
        weights = 0.5*(uc_map(:,3)+uc_map(:,5))./dist_from_current;
        if search_radius~=0
            weights(dist_from_current>search_radius^2) = 0;
        end
        [~, sorted_idx] = sort(weights,'descend');
        if weights(sorted_idx(1)) - weights(sorted_idx(2)) <= 0.05*weights(sorted_idx(1))
            [~, sel_idx] = min([dist_from_current(sorted_idx(1:2))]);
            nearest_uc = sorted_idx(sel_idx);
        else
            nearest_uc = sorted_idx(1);
        end
        nn_uc_dist = dist_from_current(nearest_uc);

    elseif strcmp(sel_mode,'dist_pop_combo_inv')
        dist_from_current = max(sum((uc_map(:,1:2) - cand(1:2)).^2,2),1e-3);
        weights = 1./(uc_map(:,3).*dist_from_current);
        if search_radius~=0
            weights(dist_from_current>search_radius^2) = 0;
        end
        [~, sorted_idx] = sort(weights,'descend');
        if weights(sorted_idx(1)) - weights(sorted_idx(2)) <= 0.05*weights(sorted_idx(1))
            [~, sel_idx] = min([dist_from_current(sorted_idx(1:2))]);
            nearest_uc = sorted_idx(sel_idx);
        else
            nearest_uc = sorted_idx(1);
        end
        nn_uc_dist = dist_from_current(nearest_uc);

    elseif strcmp(sel_mode,'dist_pop_urban')
        dist_from_current = max(sum((uc_map(:,1:2) - cand(1:2)).^2,2),1e-3);
        weights = (uc_map(:,3).*uc_map(:,4))./dist_from_current;
        if search_radius~=0
            weights(dist_from_current>search_radius^2) = 0;
        end
        [~, sorted_idx] = sort(weights,'descend');
        if weights(sorted_idx(1)) - weights(sorted_idx(2)) <= 0.05*weights(sorted_idx(1))
            [~, sel_idx] = min([dist_from_current(sorted_idx(1:2))]);
            nearest_uc = sorted_idx(sel_idx);
        else
            nearest_uc = sorted_idx(1);
        end
        nn_uc_dist = dist_from_current(nearest_uc);

    elseif strcmp(sel_mode,'dist_urban_inv')
        dist_from_current = max(sum((uc_map(:,1:2) - cand(1:2)).^2,2),1e-3);
        weights = 1./(uc_map(:,4).*dist_from_current);
        if search_radius~=0
            weights(dist_from_current>search_radius^2) = 0;
        end
        [~, sorted_idx] = sort(weights,'descend');
        if weights(sorted_idx(1)) - weights(sorted_idx(2)) <= 0.05*weights(sorted_idx(1))
            [~, sel_idx] = min([dist_from_current(sorted_idx(1:2))]);
            nearest_uc = sorted_idx(sel_idx);
        else
            nearest_uc = sorted_idx(1);
        end
        nn_uc_dist = dist_from_current(nearest_uc);
    else
        error([sel_mode , ' is not implemented']);
    end
    if iscell(uc_map)

    else
        sel_uc = uc_map(nearest_uc,1:2);
    end
    if search_radius==0 || nn_uc_dist <= search_radius^2
        [~, sorted_idx] = sort(sum(([ROI_r, ROI_c] - sel_uc).^2,2));
        dir_score = zeros(length(ROI_r),1);
        dir_score(sorted_idx(1:3)) = 1;
        [val, idx_nn] = max(dir_score.*score(sub2ind([N_r, N_c],ROI_r, ROI_c)).*(1-sparse_cand_map(sub2ind([N_r, N_c],ROI_r, ROI_c))));
        cx = ROI_r(idx_nn);
        cy = ROI_c(idx_nn);
    else
        val = 0;
        cx = cand(1);
        cy = cand(2);
    end

end
