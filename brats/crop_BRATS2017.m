function crop_BRATS2017()

hgg_folder= '/media/bioserver3/Data/Database/BRATS/BRATS2017/MICCAI_BraTS17_Data_Training/HGG/';
output_folder = '/media/bioserver3/Data/Database/BRATS/BRATS2017/MICCAI_BraTS17_Data_Training_crop/HGG/'; 
padding_folder = '/media/bioserver3/Data/Database/BRATS/BRATS2017/MICCAI_BraTS17_Data_Training_padding/HGG/'; 
mkdir(output_folder)
mkdir(padding_folder)
hgg_list = dir(hgg_folder);
maxX = 0;
maxY = 0;
maxZ = 0;
for i = 3: length(hgg_list)
    hgg_name = hgg_list(i).name
    mkdir([output_folder, hgg_name])
    mkdir([padding_folder, hgg_name])
    nii_list = dir([hgg_folder, hgg_name, '/*seg.nii']);
    for j = 1 : length(nii_list)
        nii_name  = nii_list(j).name;
        info  = nii_read_header([hgg_folder, hgg_name ,'/', nii_name]);
        V = nii_read_volume(info);
        minz = 10000;
        miny = 10000;
        maxz = 0;
        maxy = 0;
        img = zeros(size(V,2), size(V,3));
        for temp = 1 : size(V,1)
            temp_img = V(temp,:,:);
            img= reshape(temp_img, size(img));
            [y1,z1] = find(img>0);
            if ~isempty(y1)  && ~isempty(z1)
                 minz = min([minz; z1]);miny = min([miny; y1]);
                 maxz = max([maxz; z1]);maxy = max([maxy; y1]);
            end
        end

        minx = 10000;
        maxx = 0;
        img = zeros(size(V,1), size(V,3));
        for temp = 1 : size(V,2)
           
            temp_img = V(:,temp,:);
            img= reshape(temp_img, size(img));
            [x1,z1] = find(img>0);
             if ~isempty(z1) && ~isempty(x1)
                minz = min([minz; z1]);minx = min([minx; x1]);
                maxz = max([maxz; z1]);maxx = max([maxx; x1]);
             end
        end

        img = zeros(size(V,1), size(V,2));
        for temp = 1 : size(V,3)
            temp_img = V(:,:,temp);
            img= reshape(temp_img, size(img));
            [x1,y1] = find(img>0);
             if ~isempty(x1) && ~isempty(z1)
                miny = min([miny; y1]);minx = min([minx; x1]);
                maxy = max([maxy; y1]);maxx = max([maxx; x1]);
             end
        end

        
    end
    nii_list = dir([hgg_folder, hgg_name, '/*.nii']);
    for j = 1 : length(nii_list)
        nii_name  = nii_list(j).name;
        info  = nii_read_header([hgg_folder, hgg_name ,'/', nii_name]);
        V = nii_read_volume(info);
        V_padding = zeros(size(V));
        V_padding(minx:maxx, miny: maxy, minz:maxz) = V(minx:maxx, miny: maxy, minz:maxz);
        V_crop = V(minx:maxx, miny: maxy, minz:maxz);
        maxX = max(maxX, maxx - minx);
        maxY = max(maxY, maxy - miny);
        maxZ = max(maxZ, maxz - minz);
        maxX
        maxY
        maxZ
        filename_crop = [output_folder, hgg_name, '/' nii_name];
        niftiwrite(V_crop,filename_crop)
        gzip(filename_crop);
        filename_padding = [padding_folder, hgg_name, '/' nii_name];
        niftiwrite(V_padding,filename_padding)
        gzip(filename_padding);
    end  

end

save('max_size.mat', 'maxX', 'maxY', 'maxZ');
% aligment
output_adjust_folder = '/media/bioserver3/Data/Database/BRATS/BRATS2017/MICCAI_BraTS17_Data_Training_crop_adjust/HGG/'; 
for i = 3: length(hgg_list)
    hgg_name = hgg_list(i).name;
    nii_list = dir([output_folder, hgg_name, '/*.nii']);
    for j = 1 : length(nii_list)
        nii_name  = nii_list(j).name;
        info  = nii_read_header([output_folder, hgg_name ,'/', nii_name]);
        V = nii_read_volume(info);
        startX = int(0.5*(maxX - size(V,1)));
        startY = int(0.5*(maxY - size(V,2)));
        startZ = int(0.5*(maxZ - size(V,3)));
        newV = zeros(maxX, maxY, maxZ);
        newV(startX: startX + size(V,1)-1, startY: startY + size(V,2)-1, startZ: startZ + size(V,3)-1) = V;
        filename_crop = [output_adjust_folder, hgg_name, '/' nii_name];
        niftiwrite(V_crop,filename_crop)
        gzip(filename_crop);
    end
end




