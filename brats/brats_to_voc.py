import nibabel
import os

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
		
		repeat for each view (axial, sagitarial, corneal)
		repeat for each type: whole tumor, core, edema
		
		output_dir
			|-> edema
			|-> enhancing_core
			|-> whole_tumor
				|-> corneal
				|-> sagitarial
				|-> axial
					|-> labels
					|-> masks
					|-> images
						|-> 0000001.png
						|-> 0000002.png
						...
		
	labels: 
	1,2,3,4 = “edema,”
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
	if(not path.exists(path)):
		os.mkdir(path)

'''input_dir = '/home/kyle/datasets/brats/'        # where brats is located
output_dir = '/home/kyle/datasets/brats_VOC/'   # where brats VOC is created '''


input_dir = sys.argv[1] # where brats is located

output_dir = sys.argv[2] # where brats VOC is created

seg_file_end = '_seg.nii'

directions = ["axial", "corneal", "sagitarial"]
modes = ["flair", "t1", "t2", "t1ce"]
folders = ["images", "labels", "masks"]


labels = {"whole_tumor":[1,2,3,4], "enhancing_core":1, "edema":3, "none":4}

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
        print('file_ids')
        print(file_ids)
	    	
        for file_id in file_ids: # process each set of files

            seg_data_temp = nibabel.load(os.path.join(input_dir, "seg", file_id+"_seg.nii")).get_data()
            seg_data = seg_data_temp > 0 + np.zeros(seg_data_temp.shape)

            for l in label:
                seg_data += seg_data_temp == l 
            print('seg_data.shape')
            print(seg_data.shape)

            file_id_data = [] # "flair", "t1", "t2", "t1ce"
            for mode in modes:
                
                # depth, width, height
                data = nibabel.load(os.path.join(input_dir, mode, file_id+"_"+mode+".nii")).get_data()
                print('data.shape')
                print(data.shape)
                assert(seg_data.shape == data.shape)
                
                file_id_data.append(data)
                

            # transpose so axis 0 is sliced
            if(direction == "axial"):
                pass
                # np.transpose(data, [2,1,0])
                # np.transpose(seg_data, [2,1,0])
            elif(direction == "corneal"):
                pass
                # np.transpose(data, [2,1,0])
                # np.transpose(seg_data, [2,1,0])
            elif(direction == "sagitarial"):
                pass
                # np.transpose(data, [2,1,0])
                # np.transpose(seg_data, [2,1,0])
            else:
                raise

            # slice images and save
            for i in range(data.shape[0]):
                img = np.concatenate(file_id_data[:][i:i+1,:,:], axis=0)
                seg = np.concatenate((seg_data,seg_data,seg_data,seg_data), axis=0)
                print('img.shape')
                print(img.shape)
                Image.write(img)
                Image.write(seg)
                


def get_file_id(file):
	return "_".join(file.split("_")[:-1])

		
"""end

% flair_label = [1, 2, 3, 4]; %whole tumor
% t1_label= 1; % tumor core?
% t2_label= 3;
% t1ce_label= 4;

Le 645

"""
