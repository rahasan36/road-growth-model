function [ROI_r, ROI_c] = get_adjusted_ROI(r0,c0, r1,r2, c1, c2)
    temp_r = [-ones(1,3);zeros(1,3);ones(1,3)];
    temp_c = temp_r';
    if (r0-r1)*(r0-r2)*(c0-c1)*(c0-c2) ~= 0
        ROI_r = temp_r;
        ROI_c = temp_c;
    else
        if r0==r1
           if c0==c1
               ROI_r = temp_r(2:end,2:end);
               ROI_c = temp_c(2:end,2:end);
           elseif c0==c2
               ROI_r = temp_r(2:end,1:2);
               ROI_c = temp_c(2:end,1:2);
           else
               ROI_r = temp_r(2:end,:);
               ROI_c = temp_c(2:end,:);
           end
        elseif r0==r2
            if c0==c1
               ROI_r = temp_r(1:2,2:end);
               ROI_c = temp_c(1:2,2:end);
           elseif c0==c2
               ROI_r = temp_r(1:2,1:2);
               ROI_c = temp_c(1:2,1:2);
           else
               ROI_r = temp_r(1:2,:);
               ROI_c = temp_c(1:2,:);
           end
        else
            if c0==c1
               ROI_r = temp_r(:,2:end);
               ROI_c = temp_c(:,2:end);
            else
               ROI_r = temp_r(:,1:2);
               ROI_c = temp_c(:,1:2);
            end
        end

    end

    ROI_r = ROI_r(:) + r0;
    ROI_c = ROI_c(:) + c0;
end
