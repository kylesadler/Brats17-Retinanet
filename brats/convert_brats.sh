#!/bin/bash

if [ -z $1 ]
then
echo "usage:"
echo "convert_brats.sh /path/to/brats.zip/ /path/to/created/brats_coco/"
exit 1

fi

move_files(){
	files=("$1"/*)
	size=${#files[@]}

	for ((i=0;i<$size;i++));
	do
	mv ${files[$i]}/*flair.nii.gz flair
	mv ${files[$i]}/*t2.nii.gz t2
	mv ${files[$i]}/*t1ce.nii.gz t1ce
	mv ${files[$i]}/*t1.nii.gz t1
	mv ${files[$i]}/*seg.nii.gz seg
	done
}

cd $1

# unzip MICCAI_BraTS17_Data_Training_for_NLe.zip 
cd MICCAI_BraTS17_Data_Training/
# rm survival_data.csv
# mkdir flair/ t1/ t1ce/ t2/ seg/

# move_files "HGG"
# move_files "LGG"

# rm -r HGG LGG
# gunzip -vr * &
#echo $(pwd)/'MICCAI_BraTS17_Data_Training/'
#echo $2

brats_voc_dir=$(pwd)'/../brats2017_voc/'
mkdir $brats_voc_dir
matlab -nodisplay -nosplash -nodesktop -r "try, convert_BRATS17_VOC($(pwd), $brats_voc_dir), catch me, fprintf('%s / %s\n',me.identifier,me.message) exit(1), end, exit(0)"

python brats_to_coco.py $brats_voc_dir $2
