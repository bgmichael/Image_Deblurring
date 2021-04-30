#Image deblurring using the Richardson-Lucy Algorithm

from typing import List
import numpy as np
import math
import cv2
import random as r
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
from numpy.core.fromnumeric import transpose
from numpy.lib.function_base import diff
from numpy.lib.type_check import _imag_dispatcher
import scipy
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.ndimage import convolve
import os
import time

from skimage import restoration

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
   
    return resized

def Combine_Images(ListOfImages):
    #Takes each estimant from the iterations and iteratively combines them into a single image
    #finalImage is the combined image
    numberOfImages = len(ListOfImages)
    finalImage = np.zeros((ListOfImages[0].shape[0], ListOfImages[0].shape[1], ListOfImages[0].shape[2]), np.uint8)
    for i in range(numberOfImages):
        np.add(finalImage, ListOfImages[i], finalImage)

    return finalImage



def GreyScale_Image(imageMatrix):

    gray_image = cv2.cvtColor(imageMatrix, cv2.COLOR_BGR2GRAY) #In built opencv2 function for greyscaling
    
    return gray_image

def Prebuilt_Convolultion(image, filter): #NONFUNCTIONAL CURRENTLY

    if filter is None:
        gauss = [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1],
                 ]
    else:
        gauss = filter

    baseMatrix = image

    imageLength = len(baseMatrix)
    imageWidth = len(baseMatrix[0])
    imageDepth = len(baseMatrix[0][0])

    smallerMatrix = filter
    row = baseMatrix[0]
    #print(row)
    pixel = baseMatrix[0][0]
    print()
    print()
    #print(pixel)

    blank_matrix = np.zeros([imageLength, imageWidth, imageDepth], 'uint8')

    for i in range(imageLength):
        inner_row = baseMatrix[i]
        inner_convolution = convolve(inner_row, filter)
        blank_matrix[i] = inner_convolution

    return blank_matrix

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
    #Does not function correctly, for the purposes of our algorithm
    blurredGausImage = gaussian_filter(imageArray, sigma=1)

    return blurredGausImage

def PrebuiltVersion_Main_Iteration(I, Ok, PSF, numberOfIterations):
    # An attempt to use prebuilt convolution and blurring function as a test base for our version. Did not deblur image, so it 
    # was put on hold. Could be useful to debug for future improvement. 
    newEstimate = Ok
    iterator = 0
    ListOfImages = []

    while iterator < numberOfIterations:
        ListOfImages.append(newEstimate)
        #I = originalBlurredImage #From the formula

        denominator_array = Prebuilt_Convolultion(newEstimate, PSF)
        dividend = np.floor_divide(I, denominator_array)
        transposedPSF = np.transpose(PSF)
        correctionFactor = Prebuilt_Convolultion(dividend, transposedPSF)
        oldEstimate = newEstimate
        newEstimate = np.multiply(newEstimate, correctionFactor)
        #Estimate_Convergence(newEstimate, oldEstimate)
        iterator = iterator + 1

        text_string = 'Prebuilt Conv. iter.  %s' %iterator
        print(text_string)
        
    #combinedImage = Combine_Images(ListOfImages)

    cv2.imshow(text_string , newEstimate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return newEstimate, ListOfImages

def Estimate_Convergence(newEstimate, oldEstimate):
    #Not utalized yet, it will be the method by which it is determined if the iterations should continue 
    Ok = oldEstimate
    Ok_PlusOne = newEstimate
    current_convergence = np.floor_divide(Ok, Ok_PlusOne)

    print("Current Convergence: %d", current_convergence)

    return current_convergence

def Matrix_Testing(matrix, x_val, y_val):
    #Just a function to test the outputs of a particular pixel in a image matrix, used for debugging purposes
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

def General_Gaussian_FilterBlur(image, gauss=None):
    #A Gaussian blur to an image, with a hard coded filter. Creates a zeroes array, computers the convolution;
    #and then fills in the zeroes array as the return. The quadruple nested for loop is the traversal of the image
    #pixels
    if gauss is None:
        gauss = [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1],
                 ]
    else:
        gauss = gauss
    
    if len(gauss) == 3:
        offset = 1
    elif len(gauss) == 5:
        offset = 2

    filterWeight = 0
    for i in range(len(gauss)):
        for j in range(len(gauss[i])):
            filterWeight = filterWeight + gauss[i][j]
    
    #below loop is for grayscale images, not used in final implementation
    if len(image.shape) == 2:


        h = (len(image))
        w = (len(image[0]))

        #out = np.zeros((h+1, w+1, 3), np.float32)
        out = np.zeros((h, w), np.uint8)

        for i in range(h-1): #navigate the rows
            for j in range(w-1): #navigate the columns
                sum = 0 #value for new pixel
                for ki in range(len(gauss)): #navigate the kernel rows
                    for kj in range(len(gauss[0])): #navigate the kernel columns
                        sum += int(image[i + ki][j + kj] * gauss[ki][kj]) #multiple the kernel value by the image array pixel

                out[i + offset][j + offset] = sum/filterWeight #normalizing the blur by dividing by the kernel weight
    #For color images
    elif len(image.shape) == 3:

        h = (len(image))
        w = (len(image[0]))
        d = 3

        #out = np.zeros((h+2, w+2, 3), np.uint8)
        out = np.zeros((h, w, d))

        for i in range(1,len(gauss),1):
            numberCol = len(out[0])
            numberRows = len(out)
            row1 = image[0]
            lastRow = image[numberRows-i]
            col1 = image[:,0]
            lastCol = image[:,(numberCol-i)]
            out[0] = row1
            out[numberRows-i] = lastRow
            out[:,(numberCol-i)] = lastCol
            out[:,0] = col1


        for i in range(h-len(gauss)): #navigate the rows
            for j in range(w-len(gauss[0])): #navigate the columns
                for k in range(d):
                    sum = 0
                    for ki in range(len(gauss)):
                        for kj in range(len(gauss[0])):
                            image_pixel_value = image[i + ki][j + kj][k]
                            filter_value = gauss[ki][kj]

                            if np.isnan(image_pixel_value) == True:
                                image_pixel_value = np.nan_to_num(image_pixel_value)
                                print(image_pixel_value)
                                print(i+ki, j+kj, k)
                                print()
                            if np.isposinf(image_pixel_value) == True:
                                image_pixel_value = np.nan_to_num(image_pixel_value)
                                print(image_pixel_value)
                                image_pixel_value = 255
                                print(image_pixel_value)
                                print(i+ki, j+kj, k)
                                print()
                            if np.isneginf(image_pixel_value) == True:
                                image_pixel_value = np.nan_to_num(image_pixel_value)
                                print(image_pixel_value)
                                image_pixel_value = 1
                                print(image_pixel_value)
                                print(i+ki, j+kj, k)
                                print()

                            
                            sum = sum + int(image_pixel_value * filter_value)
                            #sum = sum + (image_pixel_value * filter_value)
                            # not casting the sum to an integer value yielded poor results
                    new_pixel_component = sum/filterWeight
                    if new_pixel_component == 0:
                        new_pixel_component = 1
                    out[i + offset][j + offset][k] = new_pixel_component

    return out

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
    #For my convolution function
    denominator_array = Blur_Estimate(currentEstimate, PSF)
    # denominator_array = PreBuilt_Gaussian_Blur(currentEstimate)
    # dividend = np.floor_divide(I, denominator_array)
    dividend = np.divide(I, denominator_array)
    out = dividend
    
    return out


def Main_Iteration_V2(I, Ok, PSF, numberOfIterations, UnblurredImage=None):
    #This is the main loop for the final implementation of the RLA. It calls most other functions during its execution. 
    currentEstimate = Ok
    newEstimate = currentEstimate
    newEstimate_temp = Ok
    iterator = 0
    ListOfImages = []
    ListOfDifferences = []
    stop_Flag = 0

    cv2.imshow('Original Value I ' , newEstimate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    while iterator < numberOfIterations and stop_Flag == 0:
        startLoop_Time = time.perf_counter()
        if iterator > 0:
            if (iterator > 50) == 0:
                stopMatrix, stop_List, stop_Flag = Estimate_Image_Convergence(newEstimate, oldEstimate, 1.01, .99, .95)
                #The above two loops are placeholders for future implementation of convergence functions.

        text_string = 'New Est. B4 Divide %s' %iterator
        ListOfImages.append(newEstimate)
        oldEstimate = newEstimate

        Dividend = Divide_OriginalBlurredImage(I, newEstimate, PSF) # I/(Estimate*PSF)
        text_string = 'Dividend %s' %iterator

        transpose_PSF = transpose(PSF)
        Dividend_by_transpose = General_Gaussian_FilterBlur(Dividend, transpose_PSF)
        #Dividend_by_transpose = PreBuilt_Gaussian_Blur(Dividend)
        text_string = 'Dividend By Transpose %s' %iterator
        newEstimate = np.multiply(Dividend_by_transpose, newEstimate)
        # newEstimate = np.multiply(Dividend, newEstimate)# without the transpose yields green sky

        out = TurnMatrix_To_uint(newEstimate)
        ChangesMatrix = np.subtract(oldEstimate, out)
        ListOfDifferences.append(ChangesMatrix)

        endLoop_Time = time.perf_counter()

        # cv2.imshow("Difference" , ChangesMatrix)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        sumPixels = np.sum(ChangesMatrix)
        numberOfpixels = ((len(ChangesMatrix)*(len(ChangesMatrix[0])*3)))
        averagePixels = sumPixels // numberOfpixels
        if averagePixels < 2:
            print(averagePixels)
            print(sumPixels)
            print(numberOfpixels)
            stop_Flag = 1
            print("Stop Flag Triggered %s" %iterator)
        text_string = 'New Estimate iter. %s' %iterator
        newEstimate = out

        print(text_string)
        text_string = 'New Estimate iter. %s' %iterator


        # cv2.imshow(text_string , newEstimate)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        LoopTime = endLoop_Time - startLoop_Time
        loop_text = ("Loop", iterator, "time: ", LoopTime)
        print(loop_text)
        iterator = iterator + 1
           
    # cv2.imshow(text_string , newEstimate)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return newEstimate, ListOfImages, ListOfDifferences

def Subtraction_Check_Function(OriginalUnblurredImage, ImageToCheck):
    difference = np.subtract(OriginalUnblurredImage, ImageToCheck)

    return difference

def Save_ImageList(ListOfImages, ImageName, directory, saveFactor):
    directory = directory
    #directory = r"C:\Users\Benjamin\Desktop\Temp_Image_Deblurring"
    #os.chdir(directory)
    Internal_Combination_List = []
    for i in range(len(ListOfImages)//saveFactor):
        if i == 0:
            number = i
            Internal_Combination_List.append(ListOfImages[number])
            #Inner_Combination_PlaceHolder = Combine_Images(Internal_Combination_List)
            name = ImageName + str(number)
            fileName = '%s.png' %name
            text_string = "Iteration: %d" %i

            cv2.imshow(text_string , ListOfImages[number])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imwrite(fileName, ListOfImages[number])
        else:
            number = i * saveFactor
            Internal_Combination_List.append(ListOfImages[number])
            #Inner_Combination_PlaceHolder = Combine_Images(Internal_Combination_List)
            name = ImageName + str(number)
            fileName = '%s.png' %name

            text_string = "Iteration number: %d" %number

            cv2.imshow(text_string , ListOfImages[number])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imwrite(fileName, ListOfImages[number])

    return None

def TurnMatrix_To_uint(array):
    newEstimate = array

    h = (len(newEstimate))
    w = (len(newEstimate[0]))
    d = 3
    out = np.zeros((h, w, 3), np.uint8)
    for i in range(h): #navigate the rows
        for j in range(w): #navigate the columns
            for k in range(d):
                if np.isnan(newEstimate[i][j][k]) == True:
                    newEstimate[i][j][k] = np.nan_to_num(newEstimate[i][j][k])
                    print(newEstimate[i][j][k])
                if np.isposinf(newEstimate[i][j][k]) == True:
                    newEstimate[i][j][k] = np.nan_to_num(newEstimate[i][k][k])
                    print(newEstimate[i][j][k])
                if np.isneginf(newEstimate[i][j][k]) == True:
                    newEstimate[i][j][k] = np.nan_to_num(newEstimate[i][j][k])
                    print(newEstimate[i][j][k])
                    # print(i, j, k)
                    # print()
                pixel_int = round(newEstimate[i][j][k])
                out[i][j][k] = pixel_int

    return out

def Estimate_Pixel_Convergence(pixel_k,pixel_k2,upper_threshold, lower_threshold):
    sum_k = sum(pixel_k)
    sum_k2 = sum(pixel_k2)
    average_k = sum_k/3
    average_k2 = sum_k2/average_k
    sum_average = average_k2/sum_k
    if sum_average > upper_threshold:
        out = 0
    elif sum_average < lower_threshold:
        out = 0
    else:
        out = 1
    
    return out

def Estimate_Image_Convergence(newEstimate, oldEstimate, upper_threshold, lower_threshold, convergenceTheshold):
    h = (len(newEstimate))
    w = (len(newEstimate[0]))
    d = 3

    stopMatrix = np.zeros((h, w, 3), np.uint8)
    stopMatrix = np.zeros((h, w), np.uint8)
    number_pixels = len(newEstimate) * len(newEstimate[0])
    convergence_sum = 0
    stop_List = []

    for i in range(h): #navigate the rows
        for j in range(w): #navigate the columns
            old_pixel = oldEstimate[i][j]
            new_pixel = newEstimate[i][j]
            stopMatrix_pixel_condition = Estimate_Pixel_Convergence(old_pixel, new_pixel, upper_threshold, lower_threshold)
            if stopMatrix_pixel_condition == 1:
                convergence_sum = convergence_sum + 1
                stopMatrix[i][j] = stopMatrix_pixel_condition
                stop_List.append([h, w])
            else:
                stopMatrix[i][j] = stopMatrix_pixel_condition
    
    convergence_ratio = convergence_sum/number_pixels
    if convergence_ratio > convergenceTheshold:
        stop_Flag = 1
    else:
        stop_Flag = 0

    return stopMatrix, stop_List, stop_Flag



def main():
    # UnblurredImage = "NikonSharp.jpg"
    # BlurredImage = "NikonFocusBlurred.jpg"
    # UnblurredImage = "BuildingSharp.jpg"
    # BlurredImage = "BuildingDefocusedBlurred.jpg"
    # UnblurredImage = "HonorSharp.jpg"
    # BlurredImage = "HonorBlurred.jpg"
    # UnblurredImage = "BedroomSharp.jpg"
    # BlurredImage = "BedroomBlurred.jpg"
    # UnblurredImage = "LivingRoomSharp.jpg"
    # BlurredImage = "LivingRoomBlurred.jpg"
    # UnblurredImage = "CokeSharp.jpg"
    # BlurredImage = "CokeBlurred.jpg"
    # UnblurredImage = "RailroadSharp.jpg"
    # BlurredImage = "RailroadBlurred.jpg"
    UnblurredImage = "HeartsSharp.jpg"
    BlurredImage = "HeartsBlurred.jpg"

#LOAD IMAGES
#Here we load the two images into the program as matrices. We create copies and blank images of 
#identical sizes for use in other parts of the Algorithm. 
    origUnblurredImage = cv2.imread(UnblurredImage, 1)# the '1' flag means load a color image, 0 is grayscale, -1=unchanged alpha channel
    origBlurredImage = cv2.imread(BlurredImage, 1) # Loads the blurred image matrix
    copy_origBlurredImage = origBlurredImage.copy() # for greyscale
    imageHeight = origBlurredImage.shape[0] #Find how many pixels high the image is
    imageWidth = origBlurredImage.shape[1] #Width
    blackImage = np.zeros((imageHeight, imageWidth)) # Create a black image (zeroes array) of same size, for later use. 
    # PSF =       [[0, 2, 0],
    #              [2, 4, 2],
    #              [0, 2, 0],
    #              ] 

    PSF =       [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1],
                 ] 
    #For testing of the prebuilt SCIPY RLA function. Did not work correctly. 
    PSF2 = [[[1/16,1/16,1/16],[2/16,2/16,2/16],[1/16,1/16,1/16]],
    [[2/16,2/16,2/16],[4/16,4/16,4/16],[2/16,2/16,2/16]],
    [[1/16,1/16,1/16],[2/16,2/16,2/16],[1/16,1/16,1/16]]]

    numpy_PSF = np.array(PSF2)
    # testImage = restoration.richardson_lucy(scaledOriginalBlurred, numpy_PSF, 5, 'same')
    # cv2.imshow("Prebuilt Restoration" , testImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    #A test to see if the program could incooperate a larger blur kernel. Was able to run, but degraded blurring performance. 
    PSF_5 = [
    [1,4,6,4,1],
    [4,16,24,26,4],
    [6,24,-476,24,6],
    [4,16,24,16,4],
    [1,4,6,4,1]]

    

# Below we display several images, testing the pre built greyScale and guassian blur functions
# Many of the image displays are commented out for ease of running the program, but they can be 
# uncommented for troubleshooting purposes, or removed later for cleaner code. 

    greyUnblurredImage = GreyScale_Image(origUnblurredImage)
    scaleFactor = 15
    scaledOriginalBlurred = Resize_Image(origBlurredImage, scaleFactor)
    scaledOriginalUnblurred = Resize_Image(origUnblurredImage, scaleFactor)
    scaledGreyBlurredImage = GreyScale_Image(scaledOriginalBlurred)
    baseSubtract = np.subtract(scaledOriginalUnblurred,scaledOriginalBlurred)


    # InitialFilterImage = General_Gaussian_FilterBlur(scaledOriginalUnblurred)
    # InitialFilterImage2 = General_Gaussian_FilterBlur(InitialFilterImage)
    # RectifiedInitialFilterImage = TurnMatrix_To_uint(InitialFilterImage2)

    # cv2.imshow("Initial Gaussian Blur" , RectifiedInitialFilterImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
   
   

    finalEstimate, ListOfEstimates, ListOfDifferences = Main_Iteration_V2(scaledOriginalBlurred, 
    scaledOriginalBlurred, PSF, 10, scaledOriginalUnblurred)

    #os.chdir(directory)
    directory = r"C:\Users\Benjamin\Desktop\Temp_Image_Deblurring\OutputPictures"
    saveFactor = 1
    Save_ImageList(ListOfEstimates, BlurredImage, directory, saveFactor)

    for image in range(len(ListOfDifferences)):
        text_string = "Difference %s" %image
        cv2.imshow(text_string , ListOfDifferences[image])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()