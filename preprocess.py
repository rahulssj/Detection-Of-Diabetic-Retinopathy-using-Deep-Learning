# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 02:56:44 2019

@author: rahulSSJ
"""
import cv2
import numpy as np



def processing(image):
    
# perform the actual resizing of the image and show it
    
    b,green_fundus,r = cv2.split(image)
    cv2.imshow("grenn channel",green_fundus)
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

    filtered_img = cv2.GaussianBlur(g_kernel, (5, 5), 0)
    cv2.imshow('image', green_fundus)
    cv2.imshow('filtered image', filtered_img)
    h, w = g_kernel.shape[:2]
    #green_fundus = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('gabor kernel (resized)', green_fundus)
    
    
    
    #green_fundus = cv2.bilateralFilter(green_fundus, 15, 75, 75)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)
    cv2.imshow("grenn channel",contrast_enhanced_green_fundus)

	# applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    cv2.imshow("morphology",R3)	
    f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
    
    f5 = clahe.apply(f4)
    
    return f5		
    '''width = 420
        
    resized = imutils.resize(img, width=width)
    b,g,r=cv2.split(resized)
    cv2.imshow("Width=%dpx" % (width), resized)
    # Checcking the size
    blur = cv2.GaussianBlur(g, (5, 5), 0)
    
    cv2.imshow("blur=%dpx" % (width), blur)
    # Visualizing one of the images in the array
    
    
    # Checcking the size
    #dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    #cv2.imshow("fundus=%dpx" % (width), dst)'''
    
    
    

 