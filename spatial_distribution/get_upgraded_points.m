function [to_be_upgraded, A2_upgrade] = get_upgraded_points(A1, A2, road_density_map,...
                                        max_radius,upgrade_length, min_upgrade_per_intersect)
    [N_r, N_c] = size(A1);
    CC = bwconncomp(A1);

    x1 = cellfun(@(x) get_subs(x, N_r, N_c), CC.PixelIdxList, 'UniformOutput',false);

    CC = bwconncomp(A2);

    x2 = cellfun(@(x) get_subs(x, N_r, N_c), CC.PixelIdxList, 'UniformOutput',false);

    intersect_pts = [];
    segment_id = [];
    for idx1 = 1:length(x1)
        for idx2 = 1:length(x2)
            b = pdist2(x1{idx1}, x2{idx2}, 'euclidean','Smallest',1);
            temp = b<=sqrt(2);
            intersect_pts = [intersect_pts,reshape(x2{idx2}(temp,:)',1,[])];
            segment_id = [segment_id, idx2*ones(1,sum(temp))];
        end
    end
    intersect_pts = reshape(intersect_pts,2,[])';

    [intersect_pts,ia,~] = unique(sub2ind([N_r, N_c], intersect_pts(:,1),intersect_pts(:,2)));
    segment_id = segment_id(ia);
    [u,v] = ind2sub([N_r, N_c], intersect_pts);
    intersect_pts = [u,v];

    density_vals = road_density_map(sub2ind([N_r, N_c],intersect_pts(:,1),intersect_pts(:,2)));
    upgrade_radius_per_point = max_radius*density_vals/max(density_vals);

    A2_upgrade = A2;

    upgrade_left = upgrade_length;

    to_be_upgraded = [];
    rand_order = randperm(length(segment_id));
    for idx=rand_order
        if upgrade_left <= 0
            break
        end
        mask = sum((intersect_pts(idx,:)-x2{segment_id(idx)}).^2, 2)<= upgrade_radius_per_point(idx)^2;
        idx_to_upgrade = sub2ind([N_r, N_c], x2{segment_id(idx)}(mask,1), x2{segment_id(idx)}(mask,2));
        temp = A2_upgrade;
        temp(idx_to_upgrade) = -1;

        if sum(sum(temp==-1))-sum(sum(A2_upgrade==-1))>= min_upgrade_per_intersect
            to_be_upgraded = vertcat(to_be_upgraded, x2{segment_id(idx)}(mask,:));
            A2_upgrade(idx_to_upgrade) = -1;
            upgrade_left = upgrade_length - sum(sum(A2_upgrade==-1));
        end

    end

end

function [coord] = get_subs(x, N_r, N_c)
    [a,b] = ind2sub([N_r, N_c], x);
    coord = [a,b];
end
