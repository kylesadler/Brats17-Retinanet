#!/bin/bash
unzip MICCAI_BraTS17_Data_Training_for_NLe.zip 
cd MICCAI_BraTS17_Data_Training/
mkdir flair t1 t1ce t2

files="HGG"/*
size=${#files[@]}
for ((i=0;i<$size;i++));
do
    index=$(($RANDOM % $size))
    echo ${files[$index]}
done


matlab -nodisplay -nosplash -nodesktop -r "try, convert_BRATS17_VOC($1, $2), catch me, fprintf('%s / %s\n',me.identifier,me.message) exit(1), end, exit(0)"
python 
