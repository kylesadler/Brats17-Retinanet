#!/bin/bash

if [ -z $1 ]
then
echo "usage:"
echo "convert_brats.sh /path/to/created/brats_coco/"
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

unzip MICCAI_BraTS17_Data_Training_for_NLe.zip 
cd MICCAI_BraTS17_Data_Training/
rm survival_data.csv
mkdir flair/ t1/ t1ce/ t2/ seg/

move_files "HGG"
move_files "LGG"

rm -r HGG LGG
gunzip -vr *
#echo $(pwd)/'MICCAI_BraTS17_Data_Training/'
#echo $2
matlab -nodisplay -nosplash -nodesktop -r "try, convert_BRATS17_VOC($(pwd)/'MICCAI_BraTS17_Data_Training/', $1), catch me, fprintf('%s / %s\n',me.identifier,me.message) exit(1), end, exit(0)"
#python 
