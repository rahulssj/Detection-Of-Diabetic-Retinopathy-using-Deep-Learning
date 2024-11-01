# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 05:07:31 2019

@author: rahulSSJ
"""
import cv2
import imutils

import numpy as np
def resizeImage(img):

    width=500
    resized = imutils.resize(img, width=width)
    return resized

    
    
    
    
        		
		