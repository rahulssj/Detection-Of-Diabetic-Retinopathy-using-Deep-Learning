# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 06:27:03 2019

@author: rahulSSJ
"""

  
import cv2
import numpy as np
import os
import imutils
from resized import resizeImage
from preprocess import processing


def segmentation(image):
    image=resizeImage(image)
    f5=processing(image)
    cv2.imshow("preprocessed",f5)
    
	# removing very small contours through area parameter noise removal
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)	
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
    contours,herarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)			
        im = cv2.bitwise_and(f5, f5, mask=mask)
        ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)			
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	
    newfin=  cv2.dilate(newfin,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
    

	# removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
	#vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin)	
    
    xmask = np.ones(image.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)   				
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"	
        else:
            shape = "veins"
        if(shape=="circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)	
	
    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
    cv2.imshow("Threshold And erosion",finimage)
    blood_vessels = cv2.bitwise_not(finimage)
    return blood_vessels	

if __name__ == "__main__":	
	pathFolder = "DRIVE//test//images"
	filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
	destinationFolder = "DRIVE//test//output"
	if not os.path.exists(destinationFolder):
		os.mkdir(destinationFolder)
	for file_name in filesArray:
		file_name_no_extension = os.path.splitext(file_name)[0]
		fundus = cv2.imread(pathFolder+'/'+file_name)
        		
		bloodvessel = segmentation(fundus)
		cv2.imwrite(destinationFolder+file_name_no_extension+"_bloodvessels.png",bloodvessel)
        
        
        
        