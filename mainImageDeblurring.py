#Image deblurring using the Richardson-Lucy Algorithm

from typing import List
import numpy as np
import math
import cv2
import random as r
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
import scipy
from scipy.ndimage import gaussian_filter
import os


def Resize_Image(image, scalingFactor):
    #Parameters: image = the opencv array of the image; scalingFactor=the percentage of the original for new image size
    #This function simply resizes the image for ease of processing 

    img = image
    scale_percent = scalingFactor # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dimensions = (width, height)
  
    # resize image
    resized = cv2.resize(img, dimensions, interpolation = cv2.INTER_AREA)
 
    print('Resized Dimensions : ',resized.shape)
    print(len(resized.shape))
 
    # cv2.imshow("Resized image", resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
        #gaus array was changed to a very basic gaussian blur
        gauss = [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1],
                 ]
    else:
        gauss = gauss

    if len(image.shape) == 2:


        h = (len(image) - (len(gauss) - 1))
        w = (len(image[0]) - (len(gauss[0]) - 1))

        #out = np.zeros((h+1, w+1, 3), np.float32)
        out = np.zeros((h+2, w+2), np.uint8)

        for i in range(h): #navigate the rows
            for j in range(w): #navigate the columns
                sum = 0 #value for new pixel
                for ki in range(len(gauss)): #navigate the kernel rows
                    for kj in range(len(gauss[0])): #navigate the kernel columns
                        sum += int(image[i + ki][j + kj] * gauss[ki][kj]) #multiple the kernel value by the image array pixel

                out[i][j] = sum/16 #normalizing the blur by dividing by the kernel weight

    elif len(image.shape) == 3:

        h = (len(image) - (len(gauss) - 1))
        w = (len(image[0]) - (len(gauss[0]) - 1))
        d = 3

        out = np.zeros((h+2, w+2, 3), np.uint8)

        for i in range(h): #navigate the rows
            for j in range(w): #navigate the columns
                for k in range(d):
                    sum = 0
                    for ki in range(len(gauss)):
                        for kj in range(len(gauss[0])):
                            image_pixel_value = image[i + ki][j + kj][k]
                            filter_value = gauss[ki][kj]
                            sum = sum + int(image[i + ki][j + kj][k] * gauss[ki][kj])

                    out[i][j][k] = sum/16
                    #print(out[i][j][k])
        
        #print(out)

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
    #Its basically just a bunch of iterators creating the lists and filling each spot in
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
    blurredGausImage = gaussian_filter(imageArray, sigma=1)
    return blurredGausImage

def Blur_Estimate(estimateArray, blurFilter=None):
    #Parameters: estimateArray=the current 'I' value to be blurred; blurFilter=The kernel to blur with
    #Takes the estimate array (which is 'I' in the formula) and blurs it with a gaussian kernel
    #This function can be built up for different types of blurs once those filters are integrated

    currentEstimate = estimateArray
    if blurFilter is None:
        newEstimate = General_Gaussian_FilterBlur(currentEstimate)
    else:
        newEstimate = General_Gaussian_FilterBlur(currentEstimate, blurFilter)

    return newEstimate

def Divide_OriginalBlurredImage(originalBlurredImage, currentEstimate, PSF):

    # This is following the RLA multiplicative forumla:
    #   Ok+1 = Ok X ( I/(Ok ** PSF)**PSF(Transposed) )
    #       originalBlurredImage = I;   currentEstimate=Ok; 
    #       divisorArray = (Ok**PSF);   newEstimate= Ok+1

    I = originalBlurredImage #From the formula
    denominator_array = Blur_Estimate(currentEstimate, PSF)
    dividend = np.floor_divide(I, denominator_array)
    #transposedPSF = np.transpose(PSF)
    #correctionFactor = General_Gaussian_FilterBlur(dividend, transposedPSF)

    # cv2.imshow("Correction FActor Within Divide Function" , correctionFactor)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    newEstimate = np.multiply(currentEstimate, dividend)

    return newEstimate


def Main_Iteration(I, Ok, PSF, numberOfIterations):

    newEstimate = Ok
    iterator = 0
    ListOfImages = []
    while iterator < numberOfIterations:
        ListOfImages.append(newEstimate)
        newEstimate = Divide_OriginalBlurredImage(I, newEstimate, PSF)
        iterator = iterator + 1

        text_string = 'Main iteration %s' %iterator
        print(text_string)
        # cv2.imshow(text_string , newEstimate)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    combinedImage = Combine_Images(ListOfImages)

    cv2.imshow("Final combined Image (Summation Ok)" , combinedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return combinedImage, ListOfImages

def Combine_Images(ListOfImages):
    numberOfImages = len(ListOfImages)
    finalImage = np.zeros((ListOfImages[0].shape[0], ListOfImages[0].shape[1], ListOfImages[0].shape[2]), np.uint8)
    for i in range(numberOfImages):
        np.add(finalImage, ListOfImages[i], finalImage)

    return finalImage



def PrebuiltVersion_Main_Iteration(I, Ok, PSF, numberOfIterations):

    newEstimate = Ok
    iterator = 0
    ListOfImages = []

    while iterator < numberOfIterations:
        ListOfImages.append(newEstimate)
        #I = originalBlurredImage #From the formula
        denominator_array = PreBuilt_Gaussian_Blur(newEstimate)
        dividend = np.floor_divide(I, denominator_array)
        #transposedPSF = np.transpose(PSF)
        #correctionFactor = General_Gaussian_FilterBlur(dividend, transposedPSF)
        oldEstimate = newEstimate
        newEstimate = np.multiply(newEstimate, dividend)
        #Estimate_Convergence(newEstimate, oldEstimate)
        iterator = iterator + 1

        text_string = 'Prebuilt Main iteration %s' %iterator
        print(text_string)
        

    combinedImage = Combine_Images(ListOfImages)

    cv2.imshow(text_string , combinedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return combinedImage, ListOfImages

def Estimate_Convergence(newEstimate, oldEstimate):
    Ok = oldEstimate
    Ok_PlusOne = newEstimate
    current_convergence = np.floor_divide(Ok, Ok_PlusOne)

    print("Current Convergence: %d", current_convergence)

    return current_convergence

def Matrix_Testing(matrix, x_val, y_val):
    pixel = matrix[x_val][y_val]
    pixel_R = matrix[x_val][y_val][0]
    pixel_G = matrix[x_val][y_val][1]
    pixel_B = matrix[x_val][y_val][2]
    #print("The Pixel value at %d and %d is: %s", %x_val, %y_val, %pixel)
    print("Red: %d" % pixel_R)
    print("Green: %d" % pixel_G)
    print("Blue: %d" % pixel_B)
    print(pixel)

    

    return None


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
    imageWidth = origBlurredImage.shape[1] #Width
    blackImage = np.zeros((imageHeight, imageWidth)) # Create a black image (zeroes array) of same size, for later use. 

    # PSF = Create_PointSpreadFunction(9, [1,2,1,2,4,2,1,2,1]) #change this function to every row gets its own list
    # print(PSF)
    PSF =       [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1],
                 ] 

# Below we display several images, testing the pre built greyScale and guassian blur functions
# Many of the image displays are commented out for ease of running the program, but they can be 
# uncommented for troubleshooting purposes, or removed later for cleaner code. 


    # #GreyScale the sharp image and display
    greyUnblurredImage = GreyScale_Image(origUnblurredImage)
    

    scaledOriginalBlurred = Resize_Image(origBlurredImage, 10)
    scaledGreyUnblurredImage = Resize_Image(greyUnblurredImage, 10)
    scaledGreyBlurredImage = GreyScale_Image(scaledOriginalBlurred)
   
    InitialFilterImage = General_Gaussian_FilterBlur(scaledOriginalBlurred)
    finalEstimate, ListOfEstimates = Main_Iteration(InitialFilterImage, InitialFilterImage, PSF, 400)
    #finalEstimate, ListOfEstimates = PrebuiltVersion_Main_Iteration(InitialFilterImage, InitialFilterImage, PSF, 1000)

    directory = r"C:\Users\Benjamin\Desktop\Temp_Image_Deblurring"
    #os.chdir(directory)
    Internal_Combination_List = []

    for i in range(len(ListOfEstimates)//10):
        if i == 0:
            number = i
            Internal_Combination_List.append(ListOfEstimates[number])
            Inner_Combination_PlaceHolder = Combine_Images(Internal_Combination_List)
            fileName = 'Image %d.png' %number
            #cv2.imread(ListOfEstimates[i])
            cv2.imwrite(fileName, Inner_Combination_PlaceHolder)
        else:
            number = i * 20
            Internal_Combination_List.append(ListOfEstimates[number - 1])
            Inner_Combination_PlaceHolder = Combine_Images(Internal_Combination_List)
            fileName = 'Image %d.png' %number
            #cv2.imread(ListOfEstimates[i])
            cv2.imwrite(fileName, Inner_Combination_PlaceHolder)
    
    Matrix_Testing(scaledOriginalBlurred, 10, 10)
    


if __name__ == "__main__":
    main()