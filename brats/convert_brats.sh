#!/bin/bash


matlab -nodisplay -nosplash -nodesktop -r "try, convert_BRATS17_VOC($1, $2), catch me, fprintf('%s / %s\n',me.identifier,me.message) exit(1), end, exit(0)"
python 
