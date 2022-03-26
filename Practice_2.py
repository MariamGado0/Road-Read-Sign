import cv2
import numpy as np
import sys


def output(img, kernel_sharpen):
    #applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_sharpen)
    return output

    #displaying the difference in the input vs output
    #quits window if q is pressed
    #switches between the two images when any other key is pressed
    quit = False
    while(not quit):
        cv2.imshow('image', img)
        key = cv2.waitKey(0)
        if(key == ord('q')):
            quit = True
            break;
        cv2.imshow('image', output)
        key = cv2.waitKey(0)
        if(key == ord('q')): #quit the window if q is pressed.
            quit = True
    #Destroys the open window
    cv2.destroyAllWindows()
    return output

def sharpen(img):
    #reading the image passed thorugh the command line
    #img = cv2.imread(path)

    #generating the kernels
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

    #process and output the image
    return output(img, kernel)

def excessive(img):
    #reading the image
    #img = cv2.imread(path)

    #generating the kernels
    kernel = np.array([[1,1,1], [1,-7,1], [1,1,1]])

    #process and output the image
    return output(img, kernel)

def edge_enhance(img):
    #reading the image
    #img = cv2.imread(path)

    #generating the kernels
    kernel = np.array([[-1,-1,-1,-1,-1],
                               [-1,2,2,2,-1],
                               [-1,2,8,2,-1],
                               [-2,2,2,2,-1],
                               [-1,-1,-1,-1,-1]])/8.0

    #process and output the image
    return output(img, kernel)




# img = cv2.imread("p3.jpg",1)
# output=sharpen(img)
# # kernel = np.array([[-1,-1,-1,-1,-1],
# #                                [-1,2,2,2,-1],
# #                                [-1,2,8,2,-1],
# #                                [-2,2,2,2,-1],
# #                                [-1,-1,-1,-1,-1]])/8.0


# # output = cv2.filter2D(img, -1, kernel)
# cv2.imshow('image', output)
# cv2.waitKey(0)