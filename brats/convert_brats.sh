#!/bin/bash
unzip MICCAI_BraTS17_Data_Training_for_NLe.zip 
cd MICCAI_BraTS17_Data_Training/
mkdir

files=(HGG/*)
size=${#files[@]}

for i in {0..$size}
do
    index=$(($RANDOM % $size))
    echo ${array[$index]}
done


matlab -nodisplay -nosplash -nodesktop -r "try, convert_BRATS17_VOC($1, $2), catch me, fprintf('%s / %s\n',me.identifier,me.message) exit(1), end, exit(0)"
python 
