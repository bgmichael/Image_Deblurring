#Image deblurring using the Richardson-Lucy Algorithm

import numpy as np
import math
import cv2
import random as r
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
import scipy
from scipy.ndimage import gaussian_filter


def Resize_Image(image, scalingFactor):
    img = image
    scale_percent = scalingFactor # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dimensions = (width, height)
  
    # resize image
    resized = cv2.resize(img, dimensions, interpolation = cv2.INTER_AREA)
 
    print('Resized Dimensions : ',resized.shape)
 
    cv2.imshow("Resized image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return resized

def General_Gaussian_FilterBlur(image, gauss=None):
    #A Gaussian blur to an image, with a hard coded filter. Creates a zeroes array, computers the convolution;
    #and then fills in the zeroes array as the return. The quadruple nested for loop is the traversal of the image
    #pixels

    #this hard coded gaus filter can be changed later
    if gauss is None:
        # gauss = [[0.109634, 0.111842, 0.109634],
        #          [0.111842, 0.114094, 0.111842],
        #          [0.109634, 0.111842, 0.109634],
        #          ]
        gauss = [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1],
                 ]
        # sobel = [[1/9 for i in range(3)] for i in range(3)]

    h = (len(image) - (len(gauss) - 1))
    w = (len(image[0]) - (len(gauss[0]) - 1))

    #out = np.zeros((h+1, w+1, 3), np.float32)
    out = np.zeros((h+2, w+2), np.int8)

    for i in range(h):
        for j in range(w):
            sum = 0
            for ki in range(len(gauss)):
                for kj in range(len(gauss[0])):
                    sum += int(image[i + ki][j + kj] * gauss[ki][kj])

            out[i][j] = sum

    return out

def GreyScale_Image(imageMatrix):

    gray_image = cv2.cvtColor(imageMatrix, cv2.COLOR_BGR2GRAY) #In built opencv2 function for greyscaling
    
    return gray_image
def Create_PointSpreadFunction(windowSize, valueList):
    #Parameters: windowSize - How many values are in the PSF window; 
    # valueList - The values in the PSF, filling in row by row, sequentially.
    #Return: The Point Spread Function, with each element in its own list


    windowLength = math.sqrt(windowSize) #The window must be a sqare, therefore the row/column size is the sqrt
    blankBaseMatrix = [] #For creating the window
    row_iterator = 0
    #Below I create a List of Lists for the PSF and then fill in each value of the list
    #Currently each element gets its own list, though that could be changed easily
    while row_iterator < windowLength:
        blankBaseMatrix.append([])
        row_iterator = row_iterator + 1
    column_iterator = 0
    row_iterator = 0
    while column_iterator < windowLength:
        while row_iterator < windowLength:
            blankBaseMatrix[column_iterator].append([])
            row_iterator = row_iterator + 1
        row_iterator = 0
        column_iterator = column_iterator + 1
    row_iterator = 0
    column_iterator = 0
    column_placeholder = 0
    value_index = 0
    while column_iterator < windowLength:
        row_iterator = 0
        while row_iterator < windowLength:
            value = valueList[value_index]
            blankBaseMatrix[column_iterator][row_iterator].append(value)
            value_index = value_index + 1
            row_iterator = row_iterator + 1
        column_iterator = column_iterator + 1
    


    return blankBaseMatrix

def PreBuilt_Gaussian_Blur(imageArray):
    blurredGausImage = gaussian_filter(imageArray, sigma=2)
    return blurredGausImage








def main():

    UnblurredImage = "NikonSharp.jpg"
    BlurredImage = "NikonBlurred.jpg"
 
    #LOAD IMAGES
    #Here we load the two images into the program as matrices. We create copies and blank images of 
    #identical sizes for use in other parts of the Algorithm. 

    origUnblurredImage = cv2.imread(UnblurredImage, 1)# the '1' flag means load a color image, 0 is grayscale, -1=unchanged alpha channel
    origBlurredImage = cv2.imread(BlurredImage, 1) # Loads the blurred image matrix
    copy_origBlurredImage = origBlurredImage.copy() # for greyscale
    imageHeight = origBlurredImage.shape[0] #Find how many pixels high the image is
    imageWidth = origBlurredImage.shape[0] #Width
    blackImage = np.zeros((imageHeight, imageWidth)) # Create a black image (zeroes array) of same size, for later use. 

    PSF = Create_PointSpreadFunction(9, [1,2,1,2,4,2,1,2,1])
    print(PSF)

# Below we display several images, testing the pre built greyScale and guassian blur functions

    # cv2.imshow('Before Gaussian Blur', origUnblurredImage) #Display Image with title
    # cv2.waitKey(0)# Infinite Delay until keystroke
    # cv2.destroyAllWindows() #get rid of the window

    #Run the gaussian blur on the sharp colored image
    coloredBlurAttempt = PreBuilt_Gaussian_Blur(origUnblurredImage)
    # cv2.imshow('After Colored Gaussian Blur', coloredBlurAttempt)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #GreyScale the sharp image and display
    greyUnblurredImage = GreyScale_Image(origUnblurredImage)
    # cv2.imshow('GreyScale Before Gaussian Blur', greyUnblurredImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #Blur the GreyScale image and display
    greyScaleBlurAttempt = PreBuilt_Gaussian_Blur(greyUnblurredImage)
    # cv2.imshow('After GreyScale Gaussian Blur', greyScaleBlurAttempt)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    scaledGreyUnblurredImage = Resize_Image(greyUnblurredImage, 30)
    Self_Filter_Image = General_Gaussian_FilterBlur(scaledGreyUnblurredImage)
    cv2.imshow('General Gaus blur on greyScale Image', Self_Filter_Image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    main()