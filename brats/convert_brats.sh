#!/bin/bash

move_files(){

files=("$1"/*)
size=${#files[@]}
# echo $size
for ((i=0;i<$size;i++));
do
mv ${files[$i]}/*flair.nii flair
mv ${files[$i]}/*t2.nii t2
mv ${files[$i]}/*t1ce.nii t1ce
mv ${files[$i]}/*t1.nii t1
mv ${files[$i]}/*seg.nii seg

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

#matlab -nodisplay -nosplash -nodesktop -r "try, convert_BRATS17_VOC($1, $2), catch me, fprintf('%s / %s\n',me.identifier,me.message) exit(1), end, exit(0)"
#python 
