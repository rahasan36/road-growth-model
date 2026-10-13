function make_urban_clusters(input_dir,output_dir,country_name, base_year)

    if base_year == 0 || str2num(input_dir(end))==1
        file_name = [input_dir, filesep, country_name,'_urban.tif'];
        height_file =[input_dir, filesep, country_name,'_height.tif'];
    else
        temp_dir = input_dir;
        temp_dir(end) = '1';
        file_name = [temp_dir, filesep, country_name,'_urban.tif'];
        height_file =[temp_dir, filesep, country_name,'_height.tif'];
    end

    pop_dens_file = [input_dir, filesep, country_name,'_pop_density_total.tif'];

    min_pixel_count = 5;
    save_filename = [output_dir, filesep, country_name,'_uc_data.mat'];
    epsilon = 3;
    minpts = 5;
    [A,R] = geotiffread(file_name);
    urban_data = sparse(double(A/max(max(A))));
    [u,v,~] = find(urban_data);
    [idx, core_pts] = dbscan([u,v],epsilon,minpts);

    cluster_id = unique(idx);
    cluster_id = cluster_id(2:end);
    num_pts = zeros(1,length(cluster_id));

    for id=1:length(cluster_id)
        num_pts(id) = num_pts(id) + sum(idx==cluster_id(id));
    end

    disp(['total number of clusters:', num2str(length(num_pts))])
    cluster_id = cluster_id(num_pts>=min_pixel_count);
    num_pts = num_pts(num_pts>=min_pixel_count);
    disp(['total number of clusters with at least ',num2str(min_pixel_count),' pixels : ', num2str(length(num_pts))])
    [N_r,N_c] = size(urban_data);
    [A,R] = geotiffread(pop_dens_file);
    pop_data = double(A);
    pop_data(pop_data<0) = 0;
    pop_data = double(pop_data/max(max(pop_data)));

    [A,R] = geotiffread(height_file);
    height_data = double(A/max(max(A)));
    if all(size(pop_data)==size(urban_data))
        urban_center = zeros(length(cluster_id), 5);

    else
        error(['dimension of poputation data: ', num2str(size(pop_data)),'\n dimension of urban data: ', num2str(urban_data)])
    end

    for id = 1:length(urban_center)
        mask = cluster_id(id)==idx;
        urban_center(id,1) = mean(u(mask));
        urban_center(id,2) = mean(v(mask));
        urban_center(id,3) = mean(pop_data(sub2ind([N_r, N_c],u(mask), v(mask))));
        urban_center(id,5) = mean(height_data(sub2ind([N_r, N_c],u(mask), v(mask))));
        urban_center(id,4) = sum(mask);
    end

    urban_center(:,3) = urban_center(:,3)/max(urban_center(:,3));
    urban_center(:,5) = urban_center(:,5)/max(urban_center(:,5));

    save(save_filename, 'urban_center')

    urban_center = round(urban_center);

    mask_uc = sparse(urban_center(:,1),urban_center(:,2),1,N_r, N_c);

    info = geotiffinfo(file_name);
    info.GeoTIFFTags.GeoKeyDirectoryTag.GTModelTypeGeoKey = 1;
    info.GeoTIFFTags.GeoKeyDirectoryTag.ProjectedCSTypeGeoKey = 32767;
    geotiffwrite([output_dir,filesep,country_name,'_urban_cluster_centers.tif'],single(full(mask_uc)),R,'GeoKeyDirectoryTag' ,info.GeoTIFFTags.GeoKeyDirectoryTag)

end
