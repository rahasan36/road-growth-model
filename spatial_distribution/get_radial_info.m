function [current_radial,candidate_node] = get_radial_info(cropped_A,cropped_score, ...
                            radial_r, theta_r,factor_sector_mask,show_graph)

    [N_r, N_c] = size(cropped_A);
    [br_mask, ep_mask] = make_skel(cropped_A);
    disjoint_img = cropped_A - br_mask-ep_mask;
    BW = br_mask+ep_mask;
    L = bwlabeln(BW,8);
    node_list = label2idx(L);
    CC = bwconncomp(disjoint_img,8);

    edge_collection = cellfun(@(x) get_sub(size(BW),x) ,CC.PixelIdxList,'uni',0);
    connected_nodes = cell(1, length(edge_collection));

    for node_idx=1:length(node_list)
        [r,c] = ind2sub(size(BW),[node_list{node_idx}]);
        dist_from_node = cellfun(@(x) min(pdist2(x,[r,c],'squaredeuclidean','Smallest',1)),edge_collection );
        connected_edge_idx = find(dist_from_node<=2);

        for temp_idx=connected_edge_idx
            connected_nodes{temp_idx} = [connected_nodes{temp_idx} node_idx];
        end
    end

    out = cellfun(@(x) get_path(x), connected_nodes, 'uni',0);

    edge_map = out(~cellfun('isempty',out));
    edge_map = vertcat(edge_map{:});

    if isempty(edge_map)
        current_radial = [];
        candidate_node = [];
    else
        G = graph(edge_map(:,1), edge_map(:,2));
        node_subs = cellfun(@(x) get_nn_px(get_sub(size(BW),x)) ,node_list,'uni',0);
        node_subs = vertcat(node_subs{:});

        node_subs = node_subs(1:max(unique(edge_map)),:);

        if show_graph
            figure
            imshow(cropped_A)
            hold on
            plot(G,'-b', 'XData', node_subs(:,2), 'YData',node_subs(:,1), 'LineWidth',1.5, 'NodeColor','red', 'Markersize',2)
            set(gca, 'YDir','reverse')
            hold off
        end
        deg_node = degree(G);
        grid_cand = find((deg_node>=5));

        radial_list = [];
        px_count = [];
        for cand_idx=1:length(grid_cand)
            center_pt = node_subs (grid_cand(cand_idx),:);

            nbr_nodes = setdiff(find(sum((node_subs-center_pt).^2,2)<= radial_r^2), grid_cand(cand_idx));
            r_list = [];
            for idx=1:length(nbr_nodes)
                [~,d] = shortestpath(G,grid_cand(cand_idx),nbr_nodes(idx));
                if ~isinf(d)
                    r_list = [r_list,sqrt(sum((node_subs(nbr_nodes(idx),:)-center_pt).^2,2))];
                end

            end
            r_list = r_list(abs(r_list-mean(r_list))<=5);
            if length(r_list)>=5 && mean(r_list)>=5
                radial_list = [radial_list, grid_cand(cand_idx)];
                px_count = [px_count,sum(r_list)];
            end

        end

        if show_graph
            binary_mask = zeros(size(cropped_A));
            binary_mask(vertcat(node_list{radial_list})) = 1;
            figure
            imshow(imoverlay(cropped_A, binary_mask, 'red'))
        end

        current_radial = cell(3, length(radial_list));

        for idx=1:length(radial_list)

            current_radial{1,idx} = node_subs(radial_list(idx),:);
            current_radial{2,idx} = mean(cropped_score([node_list{radial_list(idx)}]));
            current_radial{3,idx} = px_count(idx);
        end

        grid_cand = find((deg_node>=3)&(deg_node<=4));

        cand_list = [];
        cand_end_pts = cell(0);
        for cand_idx=1:length(grid_cand)
            node_to_grow = grid_cand(cand_idx);
            center_pt = node_subs(node_to_grow,:);

            nbr_nodes = setdiff(find(sum((node_subs-center_pt).^2,2)<= radial_r^2), node_to_grow);
            r_list = [];
            end_pts = [];

            for idx=1:length(nbr_nodes)
                [P,d] = shortestpath(G,node_to_grow,nbr_nodes(idx));
                if ~isinf(d)
                    r_list = [r_list,sqrt(sum((node_subs(nbr_nodes(idx),:)-center_pt).^2,2))];
                    end_pts = [end_pts; node_subs(P(end),:)];
                end

            end
            if ~isempty(end_pts)
                x = end_pts - center_pt;
                angle_list = wrapTo360(atan2d(x(:,1),x(:,2)))';
                [a,sorted_idx] = sort(angle_list);

                temp = diff([a,a(1)+360]);
                filtered_end_pts_idx = sorted_idx(abs(temp)>=15);
                filtered_end_pts = end_pts(filtered_end_pts_idx,:);
                r_list = r_list(filtered_end_pts_idx);

                rad_flag = abs(r_list-mean(r_list))<=5;
                r_list = r_list(rad_flag);

                num_new_edge = (mean(r_list)>=5)*(5- length(r_list));

                if (num_new_edge>=1) && (num_new_edge<=2)

                    x = filtered_end_pts - center_pt;
                    angle_list = wrapTo360(atan2d(x(:,1),x(:,2)));

                    b = angle_list';
                    c = [];
                    for idx=1: num_new_edge
                        a = sort(b);
                        temp = diff([a,a(1)+360]);
                        sel_a = a(abs(temp)>=15);
                        temp = temp(abs(temp)>=15);

                        [~,max_angle_idx] = max(temp);
                        c = [c,sel_a(max_angle_idx)+temp(max_angle_idx)/2];
                        b = [b,c(end)];
                    end

                    new_angle = c;
                    r_new = mean(r_list);

                    new_end_pt = round(center_pt + r_new*[sind(new_angle)', cosd(new_angle)']);
                    valid_end_pt = 1;
                    for idx=1:num_new_edge
                        sector_mask = get_sector_mask(center_pt, new_end_pt(idx,:), theta_r, N_r, N_c);
                        r_squared = sum((center_pt-new_end_pt(idx,:)).^2);
                        sector_mask_ind = find(sector_mask)';
                        [r,c] = ind2sub(size(BW),[node_list{node_to_grow}]);
                        dist_from_node = cellfun(@(x) min(pdist2(x,[r,c],'squaredeuclidean','Smallest',1)),edge_collection );
                        connected_edge_idx = find(dist_from_node<=2);

                        common_ind = [];
                        for temp_idx=connected_edge_idx
                            edge_ind = sub2ind(size(cropped_A), edge_collection{temp_idx}(:,1),edge_collection{temp_idx}(:,2));
                            common_ind = [common_ind, intersect(sector_mask_ind, edge_ind)'];
                        end

                        if numel(unique(common_ind)) >= sqrt(r_squared)*factor_sector_mask
                            valid_end_pt = 0;
                            break
                        end
                    end
                    if valid_end_pt
                        cand_list = [cand_list, grid_cand(cand_idx)];
                        cand_end_pts = [cand_end_pts, new_end_pt];
                    end

                end
            end

        end

        if show_graph
            binary_mask = zeros(size(cropped_A));
            binary_mask(vertcat(node_list{cand_list})) = 1;
            figure
            imshow(imoverlay(cropped_A, binary_mask, 'red'))
        end

        candidate_node = cell(3, length(cand_list));

        for idx=1:length(cand_list)
            candidate_node{1,idx} = node_subs(cand_list(idx),:);
            candidate_node{2,idx} = mean(cropped_score([node_list{cand_list(idx)}]));
            candidate_node{3, idx} = cand_end_pts{idx};
        end

    end

end

function v = get_sub(s,x)
    [r,c] = ind2sub(s,x);
    v = [r,c];
end

function x_nn = get_nn_px(X)

    if size(X,1)==1
        x_nn = X;
    else
        x_mean = mean(X,1);
        dist_X = sum((x_mean-X).^2,2);
        [~,idx]= min(dist_X);
        x_nn = X(idx,:);
    end

end
function P = get_path(x)
    if length(x)<2; P = [];
    else
        P = nchoosek(x,2);
    end
end
function sector_mask = get_sector_mask(x_0, x_f, theta_r, N_r, N_c)

    d_0 = x_f-x_0;
    r_0 = sqrt(sum(d_0.^2));
    theta_0 = wrapTo360(atan2d(d_0(1),d_0(2)));

    new_angle = theta_0 + [theta_r -theta_r]'/2;
    x_vert = round(x_0 + r_0*[sind(new_angle), cosd(new_angle)]);

    poly_vertices = [x_0; x_vert(1,:); x_f;x_vert(2,:); x_0];

    sector_mask = poly2mask(poly_vertices(:,2), poly_vertices(:,1),N_r, N_c);
end
