import numpy as np
import math as m
import tensorflow as tf 
import matplotlib.pyplot as plt
import cv2
import pandas
import sklearn
import sys

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

from enum import Enum

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D

def find_WBC(image):
    segments = slic(image, n_segments=100, sigma=5)
    boundaryImage = mark_boundaries(image, segments)

    mask = np.where(boundaryImage == None, 255, 0).astype("uint8")
    maskMean = cv2.mean(mask)
    
    ret, bestLabels, centers = cv2.kmeans(maskMean, 4, None, cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, cv2.kmeansFlags, cv2.KMEANS_RANDOM_CENTERS)
    
    
    

def main():
    image = cv2.imread('image.jpg', flags=cv2.Color)
    cv2.imshow("image", image)

if __name__ == "__main__": 
    main()