function convert_BRATS17_VOC(input_dir, output_dir)
% % % % ====================================== Flair 
% % % % Whole Tumor (Flair) : 2
    % input_dir = '/home/kyle/datasets/brats/'        % where brats is located
    % output_dir = '/home/kyle/datasets/brats_VOC/'   % where brats VOC is created 
    seg_file_end = '_seg.nii';
    modes = ["flair", "t1", "t2", "t1ce"]
    labels = [[1,2,3,4], 1, 3, 4]

    for i = 1:len(modes)
        mode = modes[i]
        label = labels[i]

        input_path = [input_dir, mode+"/"];
        seg_path = [output_dir, "seg/"];
        output_path = [output_dir, mode+"/"]; mkdir(output_path);
        output_img = [output_path, 'images/']; mkdir(output_img);
        output_label = [output_path, 'labels/']; mkdir(output_label);
        output_mask = [output_path, 'masks/']; mkdir(output_mask);
        mode_file_end = '_'+mode+'.nii';
        
        create_data(input_path, seg_path, output_img, output_label, output_mask, mode_file_end, seg_file_end, label);
    end
    
    % flair_label = [1, 2, 3, 4]; %whole tumor
    % t1_label= 1;
    % t2_label= 3;
    % t1ce_label= 4;
    
end

function create_data(input_path, seg_path,  output_img, output_label, output_mask, subsec0, seg_file_end, flair_label)
    input_list = dir([input_path, '*.nii']);
    for i = 1 : length(input_list)        
        input_name = input_list(i).name;
        seg_name = strrep(input_name, subsec0, seg_file_end);
        if isfile([input_path, input_name]) && isfile([seg_path, seg_name])
            info  = nii_read_header([input_path, input_name]);
            V = nii_read_volume(info);
            info  = nii_read_header([seg_path, seg_name]);
            V1 = nii_read_volume(info);
            for j = 1 : size(V, 3)
                I = V(:,:,j);
                I = double(I);
                dif = max(I(:)) - min(I(:));
                if dif == 0 
                    continue;
                else
                	I = (I - min(I(:)))/dif;
                    I1 = V1(:,:,j);
                    I1 = double(I1);
                    label = zeros(size(I1)); % non brain part
                    label(I>0)=1; % brain
                    for k = 1 : size(flair_label,2)
                        label(I1==flair_label(k)) = 2; % tumor
                    end
                
                    label = imfill(label, 'holes');
                    label = double(label);
                end
                I = im2uint8(I);
                final_im =cat(3, I, I, I);
                imwrite(final_im, [output_img, strrep(input_name, subsec0, ['_', num2str(j), '.png'])]);
                final_label = uint8(label);
                imwrite(final_label, [output_label, strrep(input_name, subsec0, ['_', num2str(j), '.png'])]);
                
                
            end
        end

     


    end
end
