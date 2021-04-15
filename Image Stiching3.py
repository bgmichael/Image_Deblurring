import random
import math
import numpy as np
import cv2
import cv
from matplotlib import pyplot as plt

# img1 = cv2.imread('waterfall1.jpg',0) # queryImage
# img2 = cv2.imread('waterfall2.jpg',0) # trainImage
#
# # Initiate SIFT detector
# #sift = cv2.xfeatures2d.SIFT_create(4)
# sift = cv2.features2d.SIFT_create(4)
#
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
#
# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)
#
# # Apply ratio test #Is this the Inlier calculation with distance being the ||HA - B||?
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append([m])
#
# # cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2) #NOTE: 'None' parameter has to be added (not in documentation)
#
# plt.imshow(img3),plt.show()



# Extra Syntax
###### Get coordinates for the ith match
##        qIdx = good[i][0].queryIdx
##        tIdx = good[i][0].trainIdx
##        x1 = kp1[qIdx].pt[0]
##        y1 = kp1[qIdx].pt[1]
##        x2 = kp2[tIdx].pt[0]
##        y2 = kp2[tIdx].pt[1]


###### Run SVD
##    U, s, V = np.linalg.svd(a, full_matrices=True)
##    hCol = np.zeros((9,1), np.float64)
##    hCol = V[8,:]


###### Invert matrix
##    Hinv = np.linalg.inv(H)

def RANSAC(keypoints):
    match_list = keypoints # each keypoint from good has a (x1,y1,x2,y2)
    Total_Error = 0
    for i in range(1000):
        #Grab 4 keypoints from the good list.
        C1 = match_list[random.randint(1, len(match_list) - 1)]# Grab a random Keypoint Pair in Keypoints
        C2 = match_list[random.randint(1, len(match_list) - 1)]# Each of these should hav four values
        C3 = match_list[random.randint(1, len(match_list) - 1)]
        C4 = match_list[random.randint(1, len(match_list) - 1)]
        if C1 == C2 or C4 == C2 or C1 == C3 or C1 == C4 or C2 == C3: #Account for duplicate points selected
            C1 = match_list[random.randint(1, len(match_list) - 1)]
            C2 = match_list[random.randint(1, len(match_list) - 1)]
            C3 = match_list[random.randint(1, len(match_list) - 1)]
            C4 = match_list[random.randint(1, len(match_list) - 1)]
        Homography1 = Build_Homography_A(C1)# takes a single keypoint pair and creates the 2X9
        Homography2 = Build_Homography_A(C2)
        Homography3 = Build_Homography_A(C3)
        Homography4 = Build_Homography_A(C4)
        A = Build_Full_A(Homography1, Homography2, Homography3, Homography4) #Builds the 8X9
        HCOL2 = Run_SVD(A) #Get the relevent values from SVD from the constructed 8X9

        #Test the homography on the creation of new pixel
        newpixel1 = Multiply_Homography_By_PixelX(HCOL2, C1)# C1 will be the whole keypoint
        newpixel2 = Multiply_Homography_By_PixelX(HCOL2, C2)#this keypoint will need to be narrowed to x1,y1
        newpixel3 = Multiply_Homography_By_PixelX(HCOL2, C3)#Newpixel value is what results from multiplying an x,y
        newpixel4 = Multiply_Homography_By_PixelX(HCOL2, C4)#By the constructed "H"
        E1 = Compute_Error(C1, newpixel1)#Compute the difference between the SIFT value and the calculated value
        E2 = Compute_Error(C2, newpixel2)
        E3 = Compute_Error(C3, newpixel3)
        E4 = Compute_Error(C4, newpixel4)
        Temp_Total_Error = E1 + E2 + E3 + E4
        if Total_Error == 0:
            Total_Error = Temp_Total_Error
            RANSAC_H = HCOL2
        if Temp_Total_Error < Total_Error:
            Total_Error = Temp_Total_Error
            RANSAC_H = HCOL2
    bestH = RANSAC_H
    return bestH

def Build_Homography_A(keypoint1):
    #Build a Homography takes a "keypoint" which contains an x,y and x', y'. It then puts it into the final
    #Homography 2X9
    A = []
    x1 = keypoint1[0]
    y1 = keypoint1[1]
    x2 = keypoint1[2]
    y2 = keypoint1[3]
    A_1 = [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2]
    A.append(A_1)
    A_2 = [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2]
    A.append(A_2)

    return A
def Build_Full_A(A1, A2, A3, A4):
    Full_A = A = np.zeros([8,9], np.float32)
    A[0] = A1[0]
    A[1] = A1[1]
    A[2] = A2[0]
    A[3] = A2[1]
    A[4] = A3[0]
    A[5] = A3[1]
    A[6] = A4[0]
    A[7] = A4[1]

    return Full_A

def Run_SVD(matrix):

    U, s, V = np.linalg.svd(matrix, full_matrices=True)
    hCol = np.zeros((9, 1), np.float64)
    hCol = V[8, :]
    hCol = hCol.reshape((3,3))
    # H here will be the proposed Homography
    Normalized_H = hCol*(1/(hCol[2][2]))
    #H = [[hCol[0], hCol[1], hCol[2]], [hCol[3], hCol[4], hCol[5]], [hCol[6], hCol[7], hCol[8]]]

    return Normalized_H



def Multiply_Homography_By_PixelX(H, x):
    new_pixel = []
    x1 = x[0]
    y1 = x[1]
    w = 1

    HPrime = [(H[0][0]*x1 + H[0][1]*y1 + H[0][2]),
              (H[1][0]*x1 + H[1][1]*y1 + H[1][2]),
              (H[2][0]*x1 + H[2][1]*y1 + H[2][2])]

    H_normalized = [(HPrime[0]/HPrime[2]),
                    (HPrime[1]/HPrime[2]),
                    (HPrime[2]/HPrime[2])]
    HPrime = H_normalized
    return HPrime

def Multiply_Other(H, x):
    new_pixel = []
    x1 = x[0]
    y1 = x[1]
    w = 1

    HPrime = [(H[0][0]*x1 + H[0][1]*y1 + H[0][2]),
              (H[1][0]*x1 + H[1][1]*y1 + H[1][2]),
              (H[2][0]*x1 + H[2][1]*y1 + H[2][2])]

    H_normalized = [(HPrime[0] / HPrime[2]),
                    (HPrime[1] / HPrime[2]),
                    (HPrime[2] / HPrime[2])]
    HPrime = H_normalized
    return HPrime

def Compute_Error(SIFTpixel, CalculatedPixel):
    normalize = CalculatedPixel[2]
    normalized_x = CalculatedPixel[0]
    normalized_y = CalculatedPixel[1]
    x1 = SIFTpixel[2]
    y1 = SIFTpixel[3]
    x2 = normalized_x
    y2 = normalized_y

    Distance = (((x2 - x1)**2 + (y2 - y1)**2))**(1/2)
    Error = Distance

    return Error

def create_blank(H, img1, img2):

    im1H = img1.shape[0]#Height - y
    im1W = img1.shape[1]#Width - x
    im2H = img2.shape[0]
    im2W = img2.shape[1]

    c1 = [0, 0, 1] #UpperLeft corner
    c2 = [im1W, 0, 1]   #Upper Right
    c3 = [0, im1H, 1]   #LowerLeft
    c4 = [im1W, im1H, 1]    #LowerRight

    mc1 = Multiply_Other(H, c1) #Mulitply an pixel
    mc2 = Multiply_Other(H, c2) #by a 3X3 "h"
    mc3 = Multiply_Other(H, c3)
    mc4 = Multiply_Other(H, c4)


    min_x = min(mc1[0], mc3[0], 0)
    max_x = max(mc2[0], mc4[0], im2W)
    min_y = min(mc1[1], mc2[1], 0)
    max_y = max(mc3[1], mc4[1], im2H)

    stitched_length = max_x - min_x
    stitched_height = max_y - min_y

    Black_Matrix = np.zeros((int(stitched_height), int(stitched_length), 3))

    return Black_Matrix

def Copy_Image2(img2, blank_matrix):
#Copies over the second of the images to the blank matrix#
    im2H = img2.shape[0]
    im2W = img2.shape[1]

    stitched_length = blank_matrix.shape[0]
    stitched_width = blank_matrix.shape[1]


    for i in range(im2H):
        for j in range(im2W):
            # blank_matrix[i-(stitched_length - 1)][j-(stitched_width - 1)] = img2[i][j]
            blank_matrix[i][j] = img2[i][j]

    copied_image = blank_matrix.copy()



    return copied_image

def Map_Image_1(stitched_background, img1, img2, H):
    #H = np.linalg.inv(H)
    im1H = img1.shape[0]
    im1W = img1.shape[1]
    im2H = img2.shape[0]
    im2W = img2.shape[1]
    stitched_length = stitched_background.shape[0]
    stitched_width = stitched_background.shape[1]

    c1 = [0, 0, 1]  # UpperLeft corner
    c2 = [im1W, 0, 1]  # Upper Right
    c3 = [0, im1H, 1]  # LowerLeft
    c4 = [im1W, im1H, 1]  # LowerRight

    mc1 = Multiply_Other(H, c1)  # Mulitply an pixel
    mc2 = Multiply_Other(H, c2)  # by a 3X3 "h"
    mc3 = Multiply_Other(H, c3)
    mc4 = Multiply_Other(H, c4)

    min_x = min(mc1[0], mc3[0], 0)
    max_x = max(mc2[0], mc4[0], im2W)
    min_y = min(mc1[1], mc2[1], 0)
    max_y = max(mc3[1], mc4[1], im2H)

    min_x = math.floor(min_x)
    max_x = math.ceil(max_x)
    min_y = math.floor(min_y)
    max_y = math.ceil(max_y)

    stitched_background2 = stitched_background.copy()

    H = np.linalg.inv(H)

    for i in range(min_y, stitched_width):
        for j in range(min_x, stitched_length):
            new_pixel_x = (H[0][0]*j+H[0][1]*i+H[0][2])/(H[2][0]*j+H[2][1]*i+H[2][2])
            new_pixel_y = (H[1][0]*j+H[1][1]*i+H[1][2])/(H[2][0]*j+H[2][1]*i+H[2][2])
            # new_pixel_w = (H[2][0]*j+H[2][1]*i+H[2][2])
            normalized_x = new_pixel_x#/new_pixel_w
            normalized_y = new_pixel_y#/new_pixel_w

            try:
                if normalized_x > 0 and normalized_y > 0:
                    stitched_background2[int(round(i - min_y))][int(round(j - min_x))] =\
                        img1[int(normalized_y)][int(normalized_x)]
            except IndexError:
                pass


    return stitched_background, stitched_background2


def Linear_Blend(copied_image, stitched_image):
    blendedIM = copied_image
    imH = copied_image.shape[0]
    imW = copied_image.shape[1]
    stimH = stitched_image.shape[0]
    stimW = stitched_image.shape[1]
    pixelcount = 0
    sum_red = 0
    sum_blue = 0
    sum_green = 0

    for i in range(imH):
        for j in range(imW):
            cpixel = blendedIM[i][j]
            spixel = stitched_image[i][j]
            avg_red = cpixel[0] - spixel[0]
            avg_green = cpixel[1] - spixel[1]
            avg_blue = cpixel[2] - spixel[2]
            if (avg_red + avg_green + avg_blue) != 0:
                sum_red = sum_red + avg_red
                sum_blue = sum_blue + avg_blue
                sum_green = sum_green + avg_green
                pixelcount = pixelcount + 1
    red_change = sum_red / pixelcount
    green_change = sum_green / pixelcount
    blue_change = sum_blue / pixelcount

    for i in range(imH):
        for j in range(imW):
            cpixel = blendedIM[i][j]
            spixel = stitched_image[i][j]
            avg_red = cpixel[0] - spixel[0]
            avg_green = cpixel[1] - spixel[1]
            avg_blue = cpixel[2] - spixel[2]
            if (avg_red + avg_green + avg_blue) != 0:
                spixel[0] = spixel[0] + red_change
                spixel[1] = spixel[1] + green_change
                spixel[2] = spixel[2] + blue_change

    return stitched_image, copied_image
def Split_Into_Elements(Panoramic_Image, img1, img2, Homography):
    H = Homography
    PiMH = Panoramic_Image.shape[0]
    PiMW = Panoramic_Image.shape[1]
    blank_background = np.zeros((PiMH, PiMW, 3))
    blank_background2 = np.zeros((PiMH, PiMW, 3))

    imH1 = img1.shape[0]
    imW1 = img1.shape[1]
    imH2 = img2.shape[0]
    imW2 = img2.shape[1]

    c1 = [0, 0, 1]  # UpperLeft corner
    c2 = [imW1, 0, 1]  # Upper Right
    c3 = [0, imH1, 1]  # LowerLeft
    c4 = [imW1, imH1, 1]  # LowerRight

    mc1 = Multiply_Other(H, c1)  # Mulitply an pixel
    mc2 = Multiply_Other(H, c2)  # by a 3X3 "h"
    mc3 = Multiply_Other(H, c3)
    mc4 = Multiply_Other(H, c4)

    min_x = min(mc1[0], mc3[0], 0)
    max_x = max(mc2[0], mc4[0], imW2)
    min_y = min(mc1[1], mc2[1], 0)
    max_y = max(mc3[1], mc4[1], imH2)

    min_x = math.floor(min_x)
    max_x = math.ceil(max_x)
    min_y = math.floor(min_y)
    max_y = math.ceil(max_y)


    for i in range(imH1 - 2):
        for j in range(imW1 - 2):
            new_pixel = Multiply_Homography_By_PixelX(H, [j, i])
            # x = int(round(new_pixel[0]))
            # y = int(round(new_pixel[1]))
            x = int(new_pixel[0])
            y = int(new_pixel[1])
            # print(i)
            # print(j)
            if sum(blank_background[y][x]) == 0:
                blank_background[y][x] = img1[i][j]
            if sum(blank_background[y][x]) != 0:
                blank_background[y + 1][x + 1] = img1[i][j]
                if sum(blank_background[y - 1][x - 1]) == 0:
                    blank_background[y - 1][x - 1] = img1[i][j]
    #Smoothing out the black lines with a little borrowing from other pixels
    for i in range(int(mc2[1]), int(mc4[1])):
        for j in range(int(mc3[0]), int(mc2[0])):
            if sum(blank_background[i][j]) == 0:
                blank_background[i][j] = blank_background[i - 1][j - 1]


    # bimH = blank_background.shape[0]
    # bimW = blank_background.shape[1]

    # for i in range(bimH - 1):
    #     for j in range(bimW - 1):
    #         if i != 0 and j != 0:
    #             if sum(blank_background[i][j]) == 0:
    #                 if sum(blank_background[i -1][j - 1]) != 0 and\
    #                      sum(blank_background[i + 1][j + 1] == 0):
    #                     blank_background[i][j] = sum(blank_background[i - 1][j - 1])

    blank_background_2 = Copy_Image2(img2, blank_background2)

    return blank_background, blank_background_2

def Compute_Spherical_Weight(image):
    imH = image.shape[0]
    imW = image.shape[1]

    center_pixel = image[(imH // 2)][(imW // 2)]

    cy = (imH // 2)
    cx = (imW // 2)

    weight_matrix = np.zeros((imH, imW, 3))

    diagonal = ((cy ** 2) + (cx ** 2)) ** (1/2)

    for i in range(imH):
        for j in range(imW):
            xdistance = cx - j
            ydistance = cy - i
            d = ((xdistance ** 2) + (ydistance ** 2)) ** (1/2)
            weight = 1 - d/diagonal
            weight_matrix[i][j] = [i, j, weight]

    return weight_matrix

def Compute_Blending_Weight(image1, image2):
    im1H = image1.shape[0]
    im1W = image1.shape[1]

    im2H = image2.shape[0]
    im2W = image2.shape[1]

    blended_weight_one = np.zeros((im1H, im1W, 3))
    blended_weight_two = np.zeros((im2H, im2W, 3))



    for i in range(im1H):
        for j in range(im1W):
            w1 = image1[i][j][2]
            w2 = image2[i][j][2]
            if w1 > w2:
                blended_weight_one[i][j] = 1
                blended_weight_two[i][j] = 0
            elif w2 > w1:
                blended_weight_one[i][j] = 0
                blended_weight_two[i][j] = 1

    return blended_weight_one, blended_weight_two



def main():


    img1 = cv2.imread('yosemite2.jpg')  # queryImage
    img2 = cv2.imread('yosemite1.jpg')  # trainImage

    im1H = img1.shape[0]
    im1W = img1.shape[1]
    im2H = img2.shape[0]
    im2W = img2.shape[1]


    # Initiate SIFT detector
    ###### SIFT IDENTIFIES THE KEYPOINTS NEEDED TO STITCH THE IMAGES TOGETHER#################
    sift = cv2.xfeatures2d.SIFT_create()
    print(sift)
    #sift = cv2.features2d.SIFT_create(4)

    # find the keypoints and descriptors with SIFT
    ############ THE ACTUAL METHOD OF SIFT FOR GETTING THE KEYPOINTS######################3333
    kp1, des1 = sift.detectAndCompute(img1, None) #Keypoints of image one
    kp2, des2 = sift.detectAndCompute(img2, None) #Keypoints of image2

    # BFMatcher with default params
    #########OUT OF THE MATCHES FOUND IN SIFT, BFMatcher FINDS WHICH ARE MOST SIMILIAR############33333
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    ############# CREATES A LIST OF WHICH OF THOSE MATCHES ARE THE BEST#####################
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])

    #### Get coordinates for the ith match
    keypoints = []
    point = []
    ##################################

    keypoints = []
    point = []


    for i in range(len(good)):
        point = []
        qIdx = good[i][0].queryIdx
        tIdx = good[i][0].trainIdx
        x1 = kp1[qIdx].pt[0]
        point.append(x1)
        y1 = kp1[qIdx].pt[1]
        point.append(y1)
        x2 = kp2[tIdx].pt[0]
        point.append(x2)
        y2 = kp2[tIdx].pt[1]
        point.append(y2)
        keypoints.append(point)


    # RANSAC(keypoints)

    H = RANSAC(keypoints)
    background = create_blank(H, img1, img2)
    partial = Copy_Image2(img2, background)


    Stiched = Map_Image_1(partial, img1, img2, H)
    copied_image = Stiched[0]
    stitched_image = Stiched[1]
    # cv2.imshow("Before Blending", stitched_image / 255)
    blendedIM = Linear_Blend(copied_image, stitched_image)
    # cv2.imshow("Stitched_Image", blendedIM[0]/255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow("Before Blending", stitched_image / 255)

    background_images = Split_Into_Elements(stitched_image, img1, img2, H)

    cv2.imshow("Stitched_Image1", background_images[0] / 255)
    cv2.imshow("Stitched_Image2", background_images[1]/255)

    cv2.imshow("Stitched_Image1 * 255", background_images[0])
    cv2.imshow("Stitched_Image2 * 255", background_images[1])

    Black_n_White1 = background_images[0]
    Black_n_White2 = background_images[1]

    mapped_color_image1 = background_images[0]
    mapped_color_image2 = background_images[1]


    Weight_One = Compute_Spherical_Weight(mapped_color_image1)
    Weight_Two = Compute_Spherical_Weight(mapped_color_image2)

    cv2.imshow("Mapped_image_one", Weight_One / 255)
    cv2.imshow("Mapped_image_two", Weight_Two / 255)

    cv2.imshow("Mapped_image_one * 255", Weight_One)
    cv2.imshow("Mapped_image_two * 255", Weight_Two)

    Blended_Image_one = Compute_Blending_Weight(Weight_One, Weight_Two)[0]
    Blended_Image_two = Compute_Blending_Weight(Weight_One, Weight_Two)[1]

    cv2.imshow("Blended One", Blended_Image_one / 255)
    cv2.imshow("Blended Two", Blended_Image_two / 255)

    cv2.imshow("Blended One * 255", Blended_Image_one)
    cv2.imshow("Blended Two * 255", Blended_Image_two)


    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()



    # # cv2.drawMatchesKnn expects list of lists as matches.
    # ########TAKES THE LIST "GOOD" WHICH WAS PREVIOUSLY GENERATED OFF OF THE BEST KEYPOINTS AND########
    # ########DRAWS THE NEW IMAGE, USING THE GOOD POINTS AS THE POINTS CONNECTED BY THE LINES###########
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None,
    #                           flags=2)  # NOTE: 'None' parameter has to be added (not in documentation)
    #
    # plt.imshow(img3), plt.show()

