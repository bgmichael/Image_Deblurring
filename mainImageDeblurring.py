#Image deblurring using the Richardson-Lucy Algorithm

import numpy as np
import math
import cv2
import random as r
from matplotlib import pyplot as plt

def GreyScale_Image(imageMatrix):

    gray_image = cv2.cvtColor(imageMatrix, cv2.COLOR_BGR2GRAY) #In built opencv2 function for greyscaling
    
    return gray_image
def Create_PointSpreadFunction(windowSize, valueList):

    windowLength = math.sqrt(windowSize) #The window must be a sqare, therefore the row/column size is the sqrt
    blankBaseMatrix = [] #For creating the window
    row_iterator = 0
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

    return blankBaseMatrix




def main():

    UnblurredImage = "NikonSharp.jpg"
    BlurredImage = "NikonBlurred.jpg"
 
    #LOAD IMAGES
    #Here we load the two images into the program as matrices. We create copies and blank images of 
    #identical sizes for use in other parts of the Algorithm. 

    origBlurredImage = cv2.imread(BlurredImage, 0) # Loads the image matrix
    copy_origBlurredImage = origBlurredImage.copy() # for greyscale
    imageHeight = origBlurredImage.shape[0] #Find how many pixels high the image is
    imageWidth = origBlurredImage.shape[0] #Width
    blackImage = np.zeros((imageHeight, imageWidth)) # Create a black image (zeroes array) of same size, for later use. 

    Create_PointSpreadFunction(9, [1,2,3,4,5,6,7,8,9])


    






if __name__ == "__main__":
    main()