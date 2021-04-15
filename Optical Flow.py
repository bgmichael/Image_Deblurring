import numpy as np
import cv2
import math


#################for reference############################
#resize (OpenCV) – resize an image (shrink to speed up testing, enlarge to see details)
#• cvtColor (OpenCV) – convert color image to grayscale (immediately after loading in image)
# • Sobel (OpenCV) – compute gradient
# • arrowedLine (OpenCV) – display arrows on top of image
# • imshow (OpenCV) – display an image
# • imwrite (OpenCV) – save an image
# • zeros (numpy) – make empty image
# • range (Python built-in)
# • float (Python built-in)
# • sum (Python built-in)
# • atan2 (Python – math)
# • pi (Python – math)
################# Given functions ###########################
# image = cv2.imread("20130712_143003.JPG")
# # convert to grayscale
# grayIm = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#
# # gradient in X direction
# gradImX = cv2.Sobel(grayIm,cv2.CV_64F,1,0,ksize=3)
#
# # gradient in Y direction
# gradImY = cv2.Sobel(grayIm,cv2.CV_64F,0,1,ksize=3)

# draw circles on corners
#cv2.circle(image,(j,i), 1, (0,0,255), -1)
# image variable, (coordinate), size, (color), -1 for border)
##############################################################

def Image_load_Gray(filename):
    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def Compute_Ix_gradients(image):
    gradImX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    return gradImX

def Compute_Iy_gradients(image):
    gradImY = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
    return gradImY

def Gradiant_Point(gradient1, gradient2):
    approx = ((gradient1)**2 + (gradient2)**2)**(1/2)
    return approx

def Create_Grad_Matrix(image, image2):
    I1_Y_Gradients = Compute_Iy_gradients(image)
    I1_X_Gradients = Compute_Ix_gradients(image)
    I2_X_Gradients = Compute_Ix_gradients(image2)
    I2_Y_Gradients = Compute_Iy_gradients(image2)

    matrix_H = image.shape[0]
    matrix_W = image.shape[1]
    zero_matrix = np.zeros((matrix_H, matrix_W, 3))
    Sum_Ix_squared = 0
    Sum_Iy_squared = 0
    Sum_IxIy = 0
    Sum_It = 0
    Sum_IxIt = 0
    Sum_IyIt = 0

    pixel_grad_test = I1_X_Gradients[0][94]

    matrix_H2 = image2.shape[0]
    matrix_W2 = image2.shape[1]
    zero_matrix2 = np.zeros((matrix_H2, matrix_W2, 3))
    zero_Color = np.zeros((matrix_H, matrix_W, 3))

    color_image = image.copy()



    for i in range(matrix_H - 1):
        for j in range(matrix_W - 1):
            # pixelgrad_x = cv2.Sobel(image[i][j], cv2.CV_64F, 1, 0, ksize=3)
            # pixelgrad_y = cv2.Sobel(image[i][j], cv2.CV_64F, 0, 1, ksize=3)
            pixelgrad_x = I1_X_Gradients[i][j]
            pixelgrad_y = I1_Y_Gradients[i][j]
            #Gradients already calculated above, grab the values for that pixel

            AVG_X_Sobel = (pixelgrad_x[0] + pixelgrad_x[1] + pixelgrad_x[2])//3
            AVG_Y_Sobel = (pixelgrad_y[0] + pixelgrad_y[1] + pixelgrad_y[2])//3

            Ix = AVG_X_Sobel
            Iy = AVG_Y_Sobel
            IxIy = AVG_Y_Sobel * AVG_X_Sobel
            Ix_squared = AVG_X_Sobel ** 2
            Iy_squared = AVG_Y_Sobel ** 2

            #Average the values together to get the overall x and y gradient for that pixel

            zero_matrix[i][j][0] = AVG_X_Sobel
            zero_matrix[i][j][1] = AVG_Y_Sobel

            #add the average value to the blank zeroes matrix
            #also, compute a sum value for all the avg x gradients in the image this is (Sigma:Ix) and (Sigma:Iy)

            Sum_IxIy = Sum_IxIy + IxIy
            Sum_Iy_squared = Sum_Iy_squared + Iy_squared
            Sum_Ix_squared = Sum_Ix_squared + Ix_squared

            # I2_pixelgrad_x = cv2.Sobel(image2[i][j], cv2.CV_64F, 1, 0, ksize=3)
            # I2_pixelgrad_y = cv2.Sobel(image2[i][j], cv2.CV_64F, 0, 1, ksize=3)

            I2_pixelgrad_x = I2_X_Gradients[i][j]
            I2_pixelgrad_y = I2_Y_Gradients[i][j]

            #Grab values for the second image

            AVG_X_Sobel2 = (I2_pixelgrad_x[0] + I2_pixelgrad_x[1] + I2_pixelgrad_x[2]) // 3
            AVG_Y_Sobel2 = (I2_pixelgrad_x[0] + I2_pixelgrad_x[1] + I2_pixelgrad_x[2]) // 3
            #Grab avg values
            AVG_T_Sobel = int(((AVG_Y_Sobel2 - AVG_Y_Sobel) + (AVG_X_Sobel2 - AVG_X_Sobel))//2)
            It = AVG_T_Sobel

            #Find values for b matrix
            IxIt = AVG_T_Sobel * AVG_X_Sobel
            IyIt = AVG_Y_Sobel * AVG_Y_Sobel

            # (Sigma:It) is found by subtracting I2 from I1? maybe like this.
            Sum_It = Sum_It + It
            Sum_IxIt =  Sum_IxIt + IxIt
            Sum_IyIt = Sum_IyIt + IyIt

            #get the summed values of It
            zero_matrix[i][j][2] = AVG_T_Sobel
            #add to the zeroes matrix

            ############# For Color Coding #################


            pixel_angle = Angle_Calculation(Ix, Iy)
            vis_color = Color_Calc(pixel_angle)
            zero_Color[i][j][0] = vis_color[2]
            zero_Color[i][j][1] = vis_color[1]
            zero_Color[i][j][2] = vis_color[0]
            color_image[i][j] = zero_Color[i][j]
            #print(pixel_angle)

    Arrow_image = Arrow_Draw(image)

    #put it in a different matrix just for naming purposes
    gradient_matrix = zero_matrix

    return gradient_matrix, Sum_IxIy, Sum_Ix_squared, Sum_Iy_squared, Sum_It, Sum_IxIt, Sum_IyIt, color_image, Arrow_image

def Color_Calc(pixel_angle):
    if pixel_angle > 0:
        percentage = pixel_angle/math.pi
        blue_val = 255 * percentage
        green_val = 0
        red_val = 0
    else:
        percentage = pixel_angle/math.pi
        green_val = 255 * percentage
        blue_val = 0
        red_val = 0

    return(red_val, green_val, blue_val)



def Velocity_Vector_Calc(Sum_IxIy, Sum_Ix_squared, Sum_Iy_squared, Sum_It, Sum_IxIt, Sum_IyIt):
    Vx = 0
    Vy = 0
    IxIy = Sum_IxIy
    Ixsquare = Sum_Ix_squared
    Iysquare = Sum_Iy_squared
    It = Sum_It
    IxIt = Sum_IxIt
    IyIt = Sum_IyIt

    detG = (Ixsquare * Iysquare) - (IxIy * IxIy)
    if detG != 0:
        Vx = (((-Iysquare) * (IxIt)) + ((IxIy) * (IyIt)))/detG
        Vy = (((Ixsquare) * (Iysquare)) - ((IxIy) * (IxIy)))/detG
    else:
        Velocity_vector = [0, 0]

    Velocity_vector = [Vx, Vy]

    return Velocity_vector

def V_Vector_Point(Sum_IxIy, Sum_Ix_squared, Sum_Iy_squared, Sum_It, Sum_IxIt, Sum_IyIt):
    Vx = 0
    Vy = 0
    IxIy = Sum_IxIy
    Ixsquare = Sum_Ix_squared
    Iysquare = Sum_Iy_squared
    It = Sum_It
    IxIt = Sum_IxIt
    IyIt = Sum_IyIt

    detG = (Ixsquare * Iysquare) - (IxIy * IxIy)
    Vx = (((-Iysquare) * (IxIt)) + ((IxIy) * (IyIt)))
    Vy = (((Ixsquare) * (Iysquare)) - ((IxIy) * (IxIy)))


    Velocity_vector = [Vx, Vy]

    return Velocity_vector


def Angle_Calculation(Vx, Vy):
    Angle = math.atan2(Vy, Vx)
    return Angle

def Arrow_Draw(image):
    matrix_H = image.shape[0]
    matrix_W = image.shape[1]
    X_gradients = Compute_Ix_gradients(image)
    Y_gradients = Compute_Iy_gradients(image)

    for i in range(0, matrix_H - 1, 20):
        for j in range(0, matrix_W - 1, 20):
            pixelgrad_x = X_gradients[i][j]
            pixelgrad_y = Y_gradients[i][j]
            AVG_X_Sobel = (pixelgrad_x[0] + pixelgrad_x[1] + pixelgrad_x[2]) // 3
            AVG_Y_Sobel = (pixelgrad_y[0] + pixelgrad_y[1] + pixelgrad_y[2]) // 3
            Ix = AVG_X_Sobel
            Iy = AVG_Y_Sobel
            Angle_Calculation(Ix, Iy)
            p2x = int(i + Ix)
            p2y = int(j + Iy)

            cv2.arrowedLine(image, (i, j), (p2x, p2y), 200)

    return image

def Display_images(image1, image2):
    image = image1
    image2 = image2


    testmatrix = Create_Grad_Matrix(image, image2)

    Sum_IxIy = testmatrix[1]
    Sum_Ix_squared = testmatrix[2]
    Sum_Iy_squared = testmatrix[3]
    Sum_It = testmatrix[4]
    Sum_IxIt = testmatrix[5]
    Sum_IyIt = testmatrix[6]

    Vel_Vector = Velocity_Vector_Calc(Sum_IxIy, Sum_Ix_squared, Sum_Iy_squared, Sum_It, Sum_IxIt, Sum_IyIt)

    Color_matrix = testmatrix[7]

    cv2.imwrite("Color_Coded Image.png", Color_matrix)
    image = cv2.imread("Color_Coded Image.png")
    cv2.imshow("Here's is the image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Arrow_image = testmatrix[8]

    cv2.imwrite("Arrow Image.png", Arrow_image)
    image = cv2.imread("Arrow Image.png")
    cv2.imshow("Here's is the image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def main():

    image = cv2.imread("mayaShapes1.png")
    image2 = cv2.imread("mayaShapes2.png")
    image3 = cv2.imread("mayaSolidShapes1.png")
    image4 = cv2.imread("mayaSolidShapes2.png")
    image5 = cv2.imread("house1.bmp")
    image6 = cv2.imread("house2.bmp")
    image7 = cv2.imread("sphere1.jpg")
    image8 = cv2.imread("sphere2.jpg")
    image9 = cv2.imread("tree1.jpg")
    image10 = cv2.imread("tree2.jpg")

    Display_images(image, image2)
    Display_images(image3, image4)
    Display_images(image5, image6)
    Display_images(image7, image8)
    Display_images(image9, image10)




if __name__ == "__main__":
    main()