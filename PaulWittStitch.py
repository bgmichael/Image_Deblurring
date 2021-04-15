"""
    Paul Witt
    CSC 340
"""

import numpy as np
import math
import cv2
import random as r


def dist(x1,x2,y1,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

def CalcM(H,j,i):
    return [(H[0][0]*j+H[0][1]*i+H[0][2])/(H[2][0]*j+H[2][1]*i+H[2][2]),
            (H[1][0]*j+H[1][1]*i+H[1][2])/(H[2][0]*j+H[2][1]*i+H[2][2]),
            1]

def main():
    #file2 = "CROPwaterfall1.jpg"
    #file1 = "CROPwaterfall2.jpg"
    file2 = "waterfall1.jpg"
    file1 = "waterfall2.jpg"
    #file1 = "waterfall2.jpg"
    #file2 = "waterfall1.jpg"
    
    #file1="reportSet1a.jpg" 
    #file2="reportSet1b.jpg"
    
    #file1="reportSet2a.jpg" 
    #file2="reportSet2b.jpg"
    
    #file1="reportSet3 (1).jpg" 
    #file2="reportSet3 (2).jpg"

    # For copying color picture
    colorImage1 = cv2.imread(file1)
    colorImage2 = cv2.imread(file2)

    img1 = cv2.imread(file1,0)
    img2 = cv2.imread(file2,0)

    # Size 
    imH = len(img1)
    imW = len(img1[0])

    coords = []
    A = np.zeros([8,9], np.float32)
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])

    counter = 0
    
    #### Extra Syntax
    #### Get coordinates for the ith match
    minError = -1
    bestH = 0
    for k in range(1000):
        t = []
        for asd in range(4):
            t.append(r.randrange(len(good)))

        counter = 0 
        err = 0
        for i in t:
            qIdx = good[i][0].queryIdx
            tIdx = good[i][0].trainIdx
            x1 = kp1[qIdx].pt[0]
            y1 = kp1[qIdx].pt[1]
            x2 = kp2[tIdx].pt[0]
            y2 = kp2[tIdx].pt[1]
            coords.append(((x1,y1,1),(x2,y2,1)))   

            A[counter] = (0,0,0,-x1,-y1,-1,x1*y2,y2*y1,y2)
            counter +=1

            A[counter] = (x1,y1,1,0,0,0,-x2*x1,-x2*y1,-x2)
            counter+=1

        ## Run SVD
        U, s, V = np.linalg.svd(A, full_matrices=True)
        hCol    = np.zeros((9,1), np.float64)
        hCol    = V[8,:]

        H = hCol.reshape((3,3))
        H = (1/H[2][2])*H
        
        # RANSAC
        for j in range(len(good)):
            qIdx = good[j][0].queryIdx
            tIdx = good[j][0].trainIdx
            x1 = kp1[qIdx].pt[0]
            y1 = kp1[qIdx].pt[1]
            x2 = kp2[tIdx].pt[0]
            y2 = kp2[tIdx].pt[1]

            c = CalcM(H,x1,y1)
            
            err += dist(c[0],x2,c[1],y2)

        if err < minError or minError == -1:
            bestH = H
            minError = err

    print("Smallest Error: ", minError)

    H = bestH
    print("Homography matrix:\n",H)

    ## Resize Image ##
    xneg = 0
    xpos = len(img2[0])
    yneg = 0
    ypos = len(img2)
    
    topleft  = CalcM(H,0,0) # CalcM uses the Homography and the coordinates
    topright = CalcM(H,len(img1[0]),0) # x=xlength of 1st image y=0
    botleft  = CalcM(H,0,len(img1)) #x=0 y=ylength of 1st image
    botright = CalcM(H,len(img1[0]),len(img1)) #y=full length of image1 x=xlength of image1

    # Find new top y (neg)
    if topleft[1] < 0 or topright[1] < 0: #topleft[1] is the y' from Homography calculation
        if topleft[1]<topright[1]: #if the calculated homography values of the left corner is less
            yneg = topleft[1] #than the calculated top right corner, set yneg to calculated top left
        else:
            yneg = topright[1]

    # Find new left x (neg)
    if topleft[0] < 0 or botleft[0] < 0:
        if topleft[0]<botleft[0]:
            xneg = topleft[0]
        else:
            xneg = botleft[0]

    ypos -= yneg #The above math is to calculate values for pixel placement
    xpos -= xneg
    # Find new bot y (large pos)
    if botleft[1]-yneg > ypos or botright[1]-yneg >ypos:  
        if botleft[1]-yneg < botright[1]-yneg:
            ypos = botright[1]-yneg
        else:
            ypos = botleft[1]-yneg

    # Find new right x (large pos)
    if topright[0]-(xneg)>xpos or botright[0]-(xneg)>xpos:
        if topright[0]-xneg<botright[0]-(xneg):
            xpos = botright[0]-xneg
        else:
            xpos = topright[0]-xneg
    
    # Round the right direction
    xneg = math.floor(xneg)
    xpos = math.ceil(xpos)
    yneg = math.floor(yneg)
    ypos = math.ceil(ypos)
    
    #print(xneg,yneg,xpos,ypos)

    # Make new images that will fit everything
    newIm  = np.zeros((ypos,xpos,3), np.float32)
    newIm2 = np.zeros((ypos,xpos,3), np.float32)
    diffSum   = [0,0,0]
    diffCount = 0

    for i in range(len(img2)):
        for j in range(len(img2[0])):
                newIm[i-yneg][j-xneg]=colorImage2[i][j]



    ##### the previous lines equate to our "Copy Image" function

    print("[**] Making transformed picture")
    H = np.linalg.inv(H)

    ylen = len(newIm)
    xlen = len(newIm[0])

    for i in range(yneg,len(newIm)):
        for j in range(xneg,len(newIm[0])):
            newx = (H[0][0]*j+H[0][1]*i+H[0][2])/(H[2][0]*j+H[2][1]*i+H[2][2])
            newy = (H[1][0]*j+H[1][1]*i+H[1][2])/(H[2][0]*j+H[2][1]*i+H[2][2])
            try:
                if 0 <newy and 0 < newx:
                    newIm2[int(round(i-yneg))][int(round(j-xneg))] =\
                    colorImage1[int(newy)][int(newx)]
            except IndexError:
                pass


    test = newIm
    test2 = newIm2


    #Adjust brightness
    print("[**] Finding lighting correction")
    for i in range(len(newIm)):
        for j in range(len(newIm[0])):
            if sum(newIm[i][j]) != 0 and sum(newIm2[i][j]) != 0:
                for k in range(3):
                    diffSum[k] += newIm[i][j][k] - newIm2[i][j][k]
                    diffCount += 1

    diffSum = [i/diffCount for i in diffSum]

    print("Correction:",diffSum)

    for i in range(yneg,len(newIm)):
        for j in range(xneg,len(newIm[0])):
            newx = (H[0][0]*j+H[0][1]*i+H[0][2])/(H[2][0]*j+H[2][1]*i+H[2][2])
            newy = (H[1][0]*j+H[1][1]*i+H[1][2])/(H[2][0]*j+H[2][1]*i+H[2][2])
            try:
                if 0 < newy and 0 < newx:
                    newIm[int(round(i-yneg))][int(round(j-xneg))] =\
                    colorImage1[int(newy)][int(newx)]+diffSum
            except IndexError:
                pass

    cv2.imshow("OUT_PICs",newIm2/255)
    cv2.imshow("OUT_PIC" ,newIm /255)

    cv2.imshow("OUT_PICs", newIm2)
    cv2.imshow("OUT_PIC", newIm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



