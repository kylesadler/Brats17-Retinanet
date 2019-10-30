function convert_BRATS17_VOC()
% % % % ====================================== Flair 
% % % % Whole Tumor (Flair) : 2

    input_path = '/home/kyle/datasets/brats_voc/nii/flair/';
    seg_path = '/home/kyle/datasets/brats_voc/nii/seg/';
    output_path = '/home/kyle/datasets/brats_voc/flair/'; mkdir(output_path);
    output_img = [output_path, 'images/']; mkdir(output_img);
    output_label = [output_path, 'labels/']; mkdir(output_label);
    output_mask = [output_path, 'masks/']; mkdir(output_mask);
    subsec0 = '_flair.nii';
    subsec1 = '_seg.nii';
    flair_label= [1, 2, 3, 4]; %whole tumor
    create_data(input_path, seg_path, output_img, output_label, output_mask, subsec0, subsec1, flair_label);
    
    
    input_path = '/home/kyle/datasets/brats_voc/nii/t1/';
    seg_path = '/home/kyle/datasets/brats_voc/nii/seg/';
    output_path = '/home/kyle/datasets/brats_voc/t1/'; mkdir(output_path);
    output_img = [output_path, 'images/']; mkdir(output_img);
    output_label = [output_path, 'labels/']; mkdir(output_label);
    output_mask = [output_path, 'masks/']; mkdir(output_mask);
    subsec0 = '_t1.nii';
    subsec1 = '_seg.nii';
    t1_label= 1;
    create_data(input_path, seg_path, output_img, output_label, output_mask, subsec0, subsec1, t1_label);
    
    input_path = '/home/kyle/datasets/brats_voc/nii/t2/';
    seg_path = '/home/kyle/datasets/brats_voc/nii/seg/';
    output_path = '/home/kyle/datasets/brats_voc/t2/'; mkdir(output_path);
    output_img = [output_path, 'images/']; mkdir(output_img);
    output_label = [output_path, 'labels/']; mkdir(output_label);
    output_mask = [output_path, 'masks/']; mkdir(output_mask);
    subsec0 = '_t2.nii';
    subsec1 = '_seg.nii';
    t2_label= 3;
    create_data(input_path, seg_path, output_img, output_label, output_mask, subsec0, subsec1, t2_label);
    
    input_path = '/home/kyle/datasets/brats_voc/nii/t1ce/';
    seg_path = '/home/kyle/datasets/brats_voc/nii/seg/';
    output_path = '/home/kyle/datasets/brats_voc/t1ce/'; mkdir(output_path);
    output_img = [output_path, 'images/']; mkdir(output_img);
    output_label = [output_path, 'labels/']; mkdir(output_label);
    output_mask = [output_path, 'masks/']; mkdir(output_mask);
    subsec0 = '_t1ce.nii';
    subsec1 = '_seg.nii';
    t1ce_label= 4;
    create_data(input_path, seg_path, output_img, output_label, output_mask, subsec0, subsec1, t1ce_label);
    
end

function create_data(input_path, seg_path,  output_img, output_label, output_mask, subsec0, subsec1, flair_label)
    input_list = dir([input_path, '*.nii']);
    for i = 1 : length(input_list)        
        input_name = input_list(i).name;
        seg_name = strrep(input_name, subsec0, subsec1);
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
