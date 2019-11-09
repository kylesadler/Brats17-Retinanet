import nibabel
import os
import sys
import numpy as np
from PIL import Image

""" 
info
	purpose: convert brats dataset into coco format
	
	input format:
		BraTS is originally in 240 x 240 x 150 data format
		[height, width, depth] with four modality .nii and one
		segmentation .nii
	
	output format:
		mri_images is (240*4) x 240 x 3 grayscale images of the four modes stacked on top of each other
		
		target is label of the brain stacked 4 times with 4 labels: 
            whole_tumor
            tumor_core
            enhancing_tumor
            everything_else
		
		repeated for each view (axial, sagittal, corneal)
		
		output_dir
            |-> corneal
            |-> sagittal
            |-> axial
                |-> labels
                |-> images
                    |-> 0000001.png
                    |-> 0000002.png
                    ...
"""

def main():
    input_dir = sys.argv[1] # where brats is located
    output_dir = sys.argv[2] # where brats VOC is created
    mkdir(output_dir)

    directions = ["axial", "coronal", "sagittal"]
    slice_axes = {"coronal":0, "sagittal":1, "axial":2}
    modes = ["flair", "t1", "t2", "t1ce"]
    labels = {"whole_tumor":[1,2,4], "tumor_core":[1,4], "enhancing_tumor":[4]}

    file_ids = [get_file_id(x) for x in os.listdir(os.path.join(input_dir, "seg"))]
    # print('file_ids')
    # print(file_ids)               

    for file_id in file_ids: # process each set of files
            
        # load segmentation data
        raw_seg_data = nibabel.load(os.path.join(input_dir, "seg", file_id + "_seg.nii")).get_data()
        assert (raw_seg_data.shape == (240, 240, 155))

        # convert to whole_tumor:1, enhancing_tumor:2, and tumor_core:3, else:0 labels
        segmentation = np.zeros(raw_seg_data.shape).astype(int)
        for labeltype in labels:
            for l in labels[labeltype]:
                segmentation += (raw_seg_data == l).astype(int)
        assert (np.max(segmentation) == 3)
        assert (np.min(segmentation) == 0)
        segmentation = normalize(segmentation)
        


        # load all mri scans from patient
        mri_scans = [] # "flair", "t1", "t2", "t1ce"
        for mode in modes:
            data = nibabel.load(os.path.join(input_dir, mode, file_id+"_"+mode+".nii")).get_data()
            assert (data.shape == (240, 240, 155))
            mri_scans.append(data)
            


        for direction in directions:
            output_path = os.path.join(output_dir, direction)
            if(os.path.exists(output_path)):
                continue
            mkdir(output_path)

            slice_axis = slice_axes[direction]
            target_shape = [240, 240, 155] # for each non-concatenated slice
            target_shape[slice_axis] = 1

            # normalize each image slice in mri_scans
            mri_data = []
            for mri_scan in mri_scans:
                
                normalized_mri = []
                for i in range(mri_scan.shape[slice_axis]):
                    mri = np.reshape(normalize(get_slice(mri_scan, slice_axis, i)), target_shape)
                    check_normalized(mri)
                    normalized_mri.append(mri)
                
                normalized_mri = np.concatenate(normalized_mri, axis=slice_axis)
                assert(normalized_mri.shape == (240, 240, 155))
                
                mri_data.append(normalized_mri)


            file_data = np.concatenate(mri_data, axis=slice_axis-1)
            seg_data = np.concatenate((segmentation,segmentation,segmentation,segmentation), axis=slice_axis-1)

            assert(file_data.shape == seg_data.shape)
            check_normalized(seg_data)
            check_normalized(file_data)

            # slice images, normalize, and save
            img_folder = os.path.join(output_path, "images")
            mkdir(img_folder)
            label_folder = os.path.join(output_path, "labels")
            mkdir(label_folder)
            for i in range(file_data.shape[slice_axis]):

                img = get_slice(file_data, slice_axis, i)
                seg = get_slice(seg_data, slice_axis, i)

                if(np.max(img) - np.min(img) == 0 and np.max(seg) - np.min(seg) == 0):
                    continue

                save(img, slice_axis, i, img_folder, file_id)
                save(seg, slice_axis, i, label_folder, file_id)


def process(path, has_label=True):
    label = np.array(
            nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')

    images = np.stack([
        np.array(nib_load(path + modal + '.nii.gz'), dtype='float32', order='C')
        for modal in modalities], -1)

    mask  = images.sum(-1) > 0

    for k in range(4):
        x = images[..., k]
        y = x[mask]
        lower = np.percentile(y, 0.2)
        upper = np.percentile(y, 99.8)
        x[mask & (x < lower)] = lower
        x[mask & (x > upper)] = upper

        y = x[mask]

        # 0.8885
        #x[mask] -= y.mean()
        #x[mask] /= y.std()

        # 0.909
        x -= y.mean()
        x /= y.std()

        #0.8704
        #x /= y.mean()

        images[..., k] = x

    #return images, label

    #output = path + 'data_f32_divm.pkl'
    output = path + 'data_f32.pkl'
    with open(output, 'wb') as f:
        pickle.dump((images, label), f)

    #mean, std = [], []
    #for k in range(4):
    #    x = images[..., k]
    #    y = x[mask]
    #    lower = np.percentile(y, 0.2)
    #    upper = np.percentile(y, 99.8)
    #    x[mask & (x < lower)] = lower
    #    x[mask & (x > upper)] = upper

    #    y = x[mask]

    #    mean.append(y.mean())
    #    std.append(y.std())

    #    images[..., k] = x
    #path = '/home/thuyen/FastData/'
    #output = path + 'data_i16.pkl'
    #with open(output, 'wb') as f:
    #    pickle.dump((images, mask, mean, std, label), f)

    if not has_label:
        return

    for patch_shape in patch_shapes:
        dist2center = get_dist2center(patch_shape)

        sx, sy, sz = dist2center[:, 0]                # left-most boundary
        ex, ey, ez = mask.shape - dist2center[:, 1]   # right-most boundary
        shape = mask.shape
        maps = np.zeros(shape, dtype="int16")
        maps[sx:ex, sy:ey, sz:ez] = 1

        fg = (label > 0).astype('int16')
        bg = ((mask > 0) * (fg == 0)).astype('int16')

        fg = fg * maps
        bg = bg * maps

        fg = np.stack(fg.nonzero()).T.astype('uint8')
        bg = np.stack(bg.nonzero()).T.astype('uint8')

        suffix = '{}x{}x{}_'.format(*patch_shape)

        output = path + suffix + 'coords.pkl'
        with open(output, 'wb') as f:
            pickle.dump((fg, bg), f)
"""

import glob
import os
import warnings
import shutil
import argparse
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from nipype.interfaces.ants import N4BiasFieldCorrection

def N4BiasFieldCorrect(filename, output_filename):
    normalized = N4BiasFieldCorrection()
    normalized.inputs.input_image = filename
    normalized.inputs.output_image = output_filename
    normalized.run()
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='training data path', default="/data/dataset/BRATS2018/training/")
    parser.add_argument('--out', help="output path", default="./N4_Normalized")
    parser.add_argument('--mode', help="output path", default="training")
    args = parser.parse_args()
    if args.mode == 'test':
        BRATS_data = glob.glob(args.data + "/*")
        patient_ids = [x.split("/")[-1] for x in BRATS_data]
        print("Processing Testing data ...")
        for idx, file_name in tqdm(enumerate(BRATS_data), total=len(BRATS_data)):
            mod = glob.glob(file_name+"/*.nii*")
            output_dir = "{}/test/{}/".format(args.out, patient_ids[idx])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for mod_file in mod:
                if 'flair' not in mod_file and 'seg' not in mod_file:
                    output_path = "{}/{}".format(output_dir, mod_file.split("/")[-1])
                    N4BiasFieldCorrect(mod_file, output_path)
                else:
                    output_path = "{}/{}".format(output_dir, mod_file.split("/")[-1])
                    shutil.copy(mod_file, output_path)
    else:
        HGG_data = glob.glob(args.data + "HGG/*")
        LGG_data = glob.glob(args.data + "LGG/*")
        hgg_patient_ids = [x.split("/")[-1] for x in HGG_data]
        lgg_patient_ids = [x.split("/")[-1] for x in LGG_data]
        print("Processing HGG ...")
        for idx, file_name in tqdm(enumerate(HGG_data), total=len(HGG_data)):
            mod = glob.glob(file_name+"/*.nii*")
            output_dir = "{}/HGG/{}/".format(args.out, hgg_patient_ids[idx])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for mod_file in mod:
                if 'flair' not in mod_file and 'seg' not in mod_file:
                    output_path = "{}/{}".format(output_dir, mod_file.split("/")[-1])
                    N4BiasFieldCorrect(mod_file, output_path)
                else:
                    output_path = "{}/{}".format(output_dir, mod_file.split("/")[-1])
                    shutil.copy(mod_file, output_path)
        print("Processing LGG ...")
        for idx, file_name in tqdm(enumerate(LGG_data), total=len(LGG_data)):
            mod = glob.glob(file_name+"/*.nii*")
            output_dir = "{}/LGG/{}/".format(args.out, lgg_patient_ids[idx])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for mod_file in mod:
                if 'flair' not in mod_file and 'seg' not in mod_file:
                    output_path = "{}/{}".format(output_dir, mod_file.split("/")[-1])
                    N4BiasFieldCorrect(mod_file, output_path)
                else:
                    output_path = "{}/{}".format(output_dir, mod_file.split("/")[-1])
                    shutil.copy(mod_file, output_path)



if __name__ == "__main__":
    main()

"""
        
def check_normalized(data):
    assert (np.max(data) == 255)
    assert (np.min(data) == 0)

def mkdir(path):
	if(not os.path.exists(path)):
		os.mkdir(path)

def get_file_id(file):
	return "_".join(file.split("_")[:-1])

def check_dims(nparray, slice_axis):
	if(slice_axis == 0):
		assert(nparray.shape == (240, 620))
	elif(slice_axis == 1):
		assert(nparray.shape == (960, 155))
	elif(slice_axis == 2):
		assert(nparray.shape == (240, 960))
	else:
		raise

def get_slice(ndarray, slice_axis, i):
	if(slice_axis == 0):
		return ndarray[i,:,:]
	elif(slice_axis == 1):
		return ndarray[:,i,:]
	elif(slice_axis == 2):
		return ndarray[:,:,i]
	else:
		raise

def normalize(data):
    min = np.min(data)
    return ((data - min) / (np.max(data) - min) * 255).astype(np.uint8)

def save(img, slice_axis, i, folder, file_id):
    check_dims(img, slice_axis)
    Image.fromarray(img, "L").save(os.path.join(folder, file_id + "_" + str(i) + ".png"))

if __name__ == '__main__':
    main()


"""
label information
    input_dir = '/home/kyle/datasets/brats/'        # where brats is located
    output_dir = '/home/kyle/datasets/brats_VOC/'   # where brats VOC is created 

    The WT describes the union of the ET, NET and
    ED, whereas the TC describes the union of the ET and NET.

    The sub-regions considered for evaluation are: 1) the "enhancing tumor" (ET), 2) the "tumor core" (TC), and 3) the "whole tumor" (WT) [see figure below]. The ET is described by areas that show hyper-intensity in T1Gd when compared to T1, but also when compared to “healthy” white matter in T1Gd. The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (fluid-filled) and the non-enhancing (solid) parts of the tumor. The appearance of the necrotic (NCR) and the non-enhancing (NET) tumor core is typically hypo-intense in T1-Gd when compared to T1. The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edema (ED), which is typically depicted by hyper-intense signal in FLAIR.
    The labels in the provided data are: 1 for NCR & NET, 2 for ED, 4 for ET, and 0 for everything else.

    # whole_tumor is 1,2,4
    # tumor_core is 1,4
    # enhancing_tumor is 4
    labels: 
    “non-enhancing (solid) core,” “necrotic (or fluid-filled) core,”
    and “non-enhancing core
"""
"""
def main():
    input_dir = sys.argv[1] # where brats is located
    output_dir = sys.argv[2] # where brats VOC is created
    mkdir(output_dir)

    directions = ["axial", "coronal", "sagittal"]
    slice_axes = {"coronal":0, "sagittal":1, "axial":2}
    modes = ["flair", "t1", "t2", "t1ce"]
    labels = {"whole_tumor":[1,2,4], "tumor_core":[1,4], "enhancing_tumor":[4]}

    for direction in directions:
        output_path = os.path.join(output_dir, direction)
        img_folder = os.path.join(output_path, "images")
        label_folder = os.path.join(output_path, "labels")

        if(os.path.exists(output_path)):
            continue

        mkdir(output_path)
        mkdir(img_folder)
        mkdir(label_folder)

        file_ids = [get_file_id(x) for x in os.listdir(os.path.join(input_dir, "seg"))]
        slice_axis = slice_axes[direction]

        # print('file_ids')
        # print(file_ids)

        for file_id in file_ids: # process each set of files

            # load segmentation data
            raw_seg_data = nibabel.load(os.path.join(input_dir, "seg", file_id + "_seg.nii")).get_data()
            assert (raw_seg_data.shape == (240, 240, 155))

            # convert to whole_tumor:1, enhancing_tumor:2, and tumor_core:3, else:0 labels
            segmentation = np.zeros(raw_seg_data.shape).astype(int)
            for labeltype in labels:
                for l in labels[labeltype]:
                    segmentation += (raw_seg_data == l).astype(int)

            
            assert (np.max(segmentation) == 3)
            assert (np.min(segmentation) == 0)
            segmentation = normalize(segmentation)
            seg_data = np.concatenate((segmentation,segmentation,segmentation,segmentation), axis=slice_axis - 1)


            # load all mri scans from patient
            file_data = [] # "flair", "t1", "t2", "t1ce"
            for mode in modes:
                data = nibabel.load(os.path.join(input_dir, mode, file_id+"_"+mode+".nii")).get_data()
                assert (data.shape == (240, 240, 155))
                
                normalized_mri = []
                for i in range(data.shape[slice_axis]):
                    target_shape = [240, 240, 155]
                    target_shape[slice_axis] = 1
                    normalized_mri.append(np.reshape(normalize(get_slice(data, slice_axis, i)), target_shape))
                normalized_mri = np.concatenate(normalized_mri, axis=slice_axis)
                assert(normalized_mri.shape == (240, 240, 155))
                file_data.append(normalized_mri)

            file_data = np.concatenate(file_data, axis=slice_axis-1)


            assert(file_data.shape == seg_data.shape)
            assert (np.max(seg_data) == 255)
            assert (np.min(seg_data) == 0)
            assert (np.max(file_data) == 255)
            assert (np.min(file_data) == 0)


            # slice images, normalize, and save
            for i in range(file_data.shape[slice_axis]):

                img = get_slice(file_data, slice_axis, i)
                seg = get_slice(seg_data, slice_axis, i)

                if(np.max(img) - np.min(img) == 0 and np.max(seg) - np.min(seg) == 0):
                    continue

                save(img, slice_axis, i, img_folder, file_id)
                save(seg, slice_axis, i, label_folder, file_id)
                
                # slice_and_save(file_data, slice_axis, i, img_folder, file_id)
                # slice_and_save(seg_data, slice_axis, i, label_folder, file_id)
"""               
"""
def slice_and_save(data, slice_axis, i, folder, file_id):

    img = get_slice(data, slice_axis, i)

    if(np.max(img) - np.min(img) == 0):
        return

    check_dims(img, slice_axis)

    Image.fromarray(img, "L").save(os.path.join(folder, file_id + "_" + str(i) + ".png"))
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
"""
def collapse_this():
    % flair_label = [1, 2, 3, 4]; %whole tumor
    % t1_label= 1; % tumor core?
    % t2_label= 3;
    % t1ce_label= 4;
"""
