import nibabel
import os
import sys
import numpy as np
from PIL import Image

"""
	purpose: convert brats dataset into coco format
	
	input format:
		BraTS is originally in 240 x 240 x 150 data format
		[height, width, depth] with four modality .nii and one
		segmentation .nii
	
	output format:
		(240*4) x 240 x 3 grayscale images of the four modes
		stacked on top of each other
		
		target is label of the brain stacked 4 times
		
		repeat for each view (axial, sagittal, corneal)
		repeat for each type: whole_tumor, tumor_core, enhancing_tumor
		
		output_dir
			|-> enhancing_tumor
			|-> tumor_core
			|-> whole_tumor
				|-> corneal
				|-> sagittal
				|-> axial
					|-> labels
					|-> masks
					|-> images
						|-> 0000001.png
						|-> 0000002.png
						...
		
	labels: 
	1,2,3,4 = enhancing_tumor,”
“non-enhancing (solid) core,” “necrotic (or fluid-filled) core,”
and “non-enhancing core
	
"""
'''
def create_data(input_path, seg_path,  output_img, output_label, output_mask, subsec0, seg_file_end, flair_label):
    input_list = dir([input_path, '*.nii']);
    for i = 1 : length(input_list)   % for each file in input_dir   
        input_name = input_list(i).name;
        seg_name = strrep(input_name, subsec0, seg_file_end);
        if isfile([input_path, input_name]) && isfile([seg_path, seg_name])
            info  = nii_read_header([input_path, input_name]);
            V = nii_read_volume(info); % mode
            info  = nii_read_header([seg_path, seg_name]);
            V1 = nii_read_volume(info); % seg
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
'''
def mkdir(path):
	if(not os.path.exists(path)):
		os.mkdir(path)


def get_file_id(file):
	return "_".join(file.split("_")[:-1])


'''input_dir = '/home/kyle/datasets/brats/'        # where brats is located
output_dir = '/home/kyle/datasets/brats_VOC/'   # where brats VOC is created 

The WT describes the union of the ET, NET and
ED, whereas the TC describes the union of the ET and NET.
'''
# whole_tumor is 1,2,4
# tumor_core is 1,4
# enhancing_tumor is 4



"""

The sub-regions considered for evaluation are: 1) the "enhancing tumor" (ET), 2) the "tumor core" (TC), and 3) the "whole tumor" (WT) [see figure below]. The ET is described by areas that show hyper-intensity in T1Gd when compared to T1, but also when compared to “healthy” white matter in T1Gd. The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (fluid-filled) and the non-enhancing (solid) parts of the tumor. The appearance of the necrotic (NCR) and the non-enhancing (NET) tumor core is typically hypo-intense in T1-Gd when compared to T1. The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edema (ED), which is typically depicted by hyper-intense signal in FLAIR.
The labels in the provided data are: 1 for NCR & NET, 2 for ED, 4 for ET, and 0 for everything else.

"""

# TODO check direction and labels

input_dir = sys.argv[1] # where brats is located

output_dir = sys.argv[2] # where brats VOC is created
mkdir(output_dir)

directions = ["axial", "coronal", "sagittal"]
modes = ["flair", "t1", "t2", "t1ce"]
folders = ["images", "labels", "masks"]

labels = {"whole_tumor":[1,2,3,4], "tumor_core":[1,4], "enhancing_tumor":4} #, "none":4}

for labeltype in labels:
    label = labels[labeltype]
    labeltype_path = os.path.join(output_dir, labeltype)
    mkdir(labeltype_path)

    for direction in directions:
        output_path = os.path.join(labeltype_path, direction)
        img_folder = os.path.join(output_path, "images")
        label_folder = os.path.join(output_path, "labels")

        mkdir(output_path)
        mkdir(img_folder)
        mkdir(label_folder)

        file_ids = [get_file_id(x) for x in os.listdir(os.path.join(input_dir, "seg"))]
        # print('file_ids')
        # print(file_ids)
	    	
        for file_id in file_ids: # process each set of files

			# seg data should be 1: healthy brain, 2: labeltype, 0:background
            seg_data_temp = nibabel.load(os.path.join(input_dir, "seg", file_id+"_seg.nii")).get_data()
            seg_data = seg_data_temp > 0 # + np.zeros(seg_data_temp.shape)

            for l in label:
                seg_data += seg_data_temp == l 
            # print('seg_data.shape') (240, 240, 155)
            # print(seg_data.shape)

            file_id_data = [] # "flair", "t1", "t2", "t1ce"
            for mode in modes:
                
                # height, width, depth
                data = nibabel.load(os.path.join(input_dir, mode, file_id+"_"+mode+".nii")).get_data()
                # print('data.shape') (240, 240, 155)
                # print(data.shape)
                assert(seg_data.shape == data.shape)
                
                file_id_data.append(data)
                


            """
			
			slicing z is axial
			slicing y is sagittal
			slicing x is coronal
			
            """
            # transpose so axis 0 is sliced
            if(direction == "axial"):
                
                file_id_data = np.concatenate(file_id_data, axis=0)
                assert(file_id_data.shape == (960, 240, 155))
                
                # slice images and save
                for i in range(data.shape[2]):
                    	# print('file_id_data.shape')
                    # print(file_id_data.shape)
                    img = file_id_data[:,:,i:i+1]
                    assert(img.shape == (960, 240, 1))
					
					# convert into three chanels
                    img = np.concatenate((img,img,img), axis=-1)
                    assert(img.shape == (960, 240, 3))
					
                    # print('img.shape')
                    # print(img.shape) 
                    
                    seg = np.concatenate((seg_data,seg_data,seg_data,seg_data), axis=0)
                    # print('img.shape') (4, 1, 240, 155)
                    # print(img.shape)
                    Image.fromarray(img).save(os.path.join(img_folder, file_id+"_"+str(i)+".png"))
                    Image.fromarray(seg).save(os.path.join(label_folder, file_id+"_"+str(i)+".png"))
            
            elif(direction == "coronal"):
                
                file_id_data = np.concatenate(file_id_data, axis=1)
                # np.transpose(file_id_data, [2,1,0])
                # slice images and save
                for i in range(data.shape[0]):
                    print('file_id_data.shape') #(4, 240, 240, 155)
                    print(file_id_data.shape)
                    print('file_id_data[:,i:i+1,:,:].shape') #(4, 1, 240, 155)
                    print(file_id_data[i:i+1,:,:].shape) 
                    img = file_id_data[i:i+1,:,:]
                    seg = np.concatenate((seg_data,seg_data,seg_data,seg_data), axis=0)
                    # print('img.shape') (4, 1, 240, 155)
                    # print(img.shape)
                    Image.fromarray(img).save(os.path.join(img_folder, file_id+"_"+str(i)+".png"))
                    Image.fromarray(seg).save(os.path.join(label_folder, file_id+"_"+str(i)+".png"))
            
            elif(direction == "sagittal"):
                
                file_id_data = np.concatenate(file_id_data, axis=0)
                # slice images and save
                for i in range(data.shape[1]):
                    print('file_id_data.shape') #(4, 240, 240, 155)
                    print(file_id_data.shape)
                    
                    print('file_id_data[:,i:i+1,:,:].shape') #(4, 1, 240, 155)
                    print(file_id_data[:,i:i+1,:].shape) 
                    img = file_id_data[:,i:i+1,:]
                    seg = np.concatenate((seg_data,seg_data,seg_data,seg_data), axis=0)
                    # print('img.shape') (4, 1, 240, 155)
                    # print(img.shape)
                    Image.fromarray(img).save(os.path.join(img_folder, file_id+"_"+str(i)+".png"))
                    Image.fromarray(seg).save(os.path.join(label_folder, file_id+"_"+str(i)+".png"))
            
            else:
                raise


		
"""end

% flair_label = [1, 2, 3, 4]; %whole tumor
% t1_label= 1; % tumor core?
% t2_label= 3;
% t1ce_label= 4;

Le 645

"""
