function [path_list] = get_next_best_cand_radial(start_pt, uc_map, score)
    [N_r, N_c] = size(score);
    path_list = [start_pt];
    cx = start_pt(1);
    cy = start_pt(2);
    flag = 1;
    while flag
        [ROI_r, ROI_c] = get_adjusted_ROI(cx, cy,1,N_r,1,N_c);
        [~, sorted_idx] = sort(sum(([ROI_r, ROI_c] - uc_map).^2,2));
        dir_score = zeros(length(ROI_r),1);
        dir_score(sorted_idx(1:3)) = 1;
        [val, idx_nn] = max(dir_score.*score(sub2ind([N_r, N_c],ROI_r, ROI_c)));
        cx = ROI_r(idx_nn);
        cy = ROI_c(idx_nn);
        path_list = [path_list; cx,cy];
        score(cx,cy) = 0;
        if sum(([ROI_r(idx_nn),ROI_c(idx_nn)] - uc_map).^2) <=2 || val == 0
            if val ~= 0
                path_list = [path_list; uc_map];
            end
            break
        end

    end

end
