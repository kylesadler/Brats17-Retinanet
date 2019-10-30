#!/bin/bash

matlab -nodisplay -nosplash -nodesktop -r /r "try, run('/home/kyle/research/retinamask_brats17/convert_BRATS17_VOC.m'), catch me, fprintf('%s / %s\n',me.identifier,me.message), end, exit"


