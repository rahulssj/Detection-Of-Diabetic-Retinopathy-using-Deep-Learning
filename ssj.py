# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 11:47:15 2019

@author: rahulSSJ
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
from ssjmod import segmentation
path = "D:\\DRIVE\\test\\images\\*.*"
destinationFolder = "drive_output"
if not os.path.exists(destinationFolder):
	os.mkdir(destinationFolder)
for bb,file in enumerate (glob.glob(path)):
    print(bb,file)
    a=cv2.imread(file)
    print(a)
    c = segmentation(a)
    cv2.imshow('Color image', c)
    #writing the images in a folder output_images
    cv2.imwrite('D:\\modify\\drive_output\\messigray{}.png'.format(bb), c)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
#from ssjmod import segmentation
#from resized import resizeImage
path = "C:\\Users\\rahul\\Desktop\\React-Native\\reviewapp\\assets\\images\\*.*"

destinationFolder = "comp"
if not os.path.exists(destinationFolder):
	os.mkdir(destinationFolder)
for bb,file in enumerate (glob.glob(path)):
    print(bb,file)
    a=cv2.imread(file)
    
    c = cv2.resize(a, (100,150), interpolation = cv2.INTER_AREA)
    
    #writing the images in a folder output_images
    cv2.imwrite('C:\\Users\\rahul\\Desktop\\React-Native\\reviewapp\\assets\\images\\comp\\comp{}.png'.format(bb), c)

