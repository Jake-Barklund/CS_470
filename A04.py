import numpy as np
import math as m
import tensorflow as tf 
import cv2
import os
import pandas
import sklearn
import sys
import math
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

from enum import Enum

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D

def getLBPImage(image):
    radius = 1  
    sampleCnt = 8

    image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
    lbpImage = np.zeros_like(image)

    for h in range(0,image.shape[0] - sampleCnt):
        for w in range(0,image.shape[1] - sampleCnt):            
            temp = image[h:h+sampleCnt,w:w+sampleCnt]
            center = temp[1,1]
            img = (temp>=center)*1.0
            imgVector = img.flatten()
            
            imgVector = np.delete(imgVector,4)
            
            tempImgVect = np.where(imgVector)[0]
            
            if len(tempImgVect) >= 1:
                num = np.sum(2**tempImgVect)
            else:
                num = 0

            lbpImage[h+1,w+1] = num

    return(lbpImage)

def getOneRegionLBPFeatures(subImages):
    hist = cv2.calcHist([subImages], [0], None, [10], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist

def getLBPFeatures(featureImage, regionSideCnt):
    width = int(featureImage.shape[1])
    height = int(featureImage.shape[0])

    allHists = []

    for i in range(0, height-regionSideCnt):
        for j in range(0, width-regionSideCnt, regionSideCnt):
            subImage = featureImage[i:i+regionSideCnt, j+regionSideCnt]
            subImageHist = getOneRegionLBPFeatures(subImage)
            allHists.append(subImageHist)

    allHists = np.array(allHists)
    allHists = np.reshape(allHists, (allHists.shape[0]*allHists.shape[1],))

    return allHists

def main():    
    imageDirectory = sys.argv[1]
    outputDirectory = sys.argv[2]
    regionSideCnt = int(sys.argv[3])
    
    outputFile = open(outputDirectory, "a+")
    imgs = os.listdir(imageDirectory)

    for imgnm in imgs:
        image = cv2.imread(os.path.join(imageDirectory, imgnm))

        imgLBP = getLBPImage(image)
        imgHist = getLBPFeatures(imgLBP, regionSideCnt)

        content = str(imgHist)

        outputFile.write(content)
        outputFile.write("\n")

    outputFile.close()

    cv2.waitKey(-1)
    cv2.destroyAllWindows()


if __name__ == "__main__": 
    main()