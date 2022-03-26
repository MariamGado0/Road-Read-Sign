import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from pract2 import *
import csv
import queue
#import pyttsx3

count_times={}
out_imgs ={}



def hsv_colorname(color):
    # red = np.uint8([[[255,0,0 ]]])
    # hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)

    hsv_color={'blue':[[100,150,0],[140,255,255]],'red':[[170,50,50],[180,255,255],[0,50,50],[10,255,255]],\
               'white':[[0,0,255],[0,0,255],[0,0,255]],'black':[0, 0, 0]}
    hsv = hsv_color[color]
    return (np.array(hsv[0]),np.array(hsv[1]))
    
def BGR2HSV(img,color_names):
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    i=0
    for color in color_names:
            
        lower,upper =hsv_colorname(color)
        
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower, upper)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img,img, mask= mask)
        
        if i != 0:
            final_img += res
        else: 
            final_img = res
        i +=1    
    
    median = cv2.medianBlur(final_img,5)
    # cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    # cv2.imshow("output",np.hstack([median,img]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return median

def img_preprocessing(img):
    # img = cv2.medianBlur(frame, 5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))  # lower red
    mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([230, 255, 255]))  # upper red 179
    mask3 = cv2.inRange(hsv, np.array([100, 150, 0]), np.array([140, 255, 255]))  # upper blue
    mask = mask1 + mask2 + mask3
    # mask = cv2.max(mask1, mask2)
    img = cv2.bitwise_and(img, img, mask=mask)
    # img = cv2.medianBlur(img, 5)
    kernel = np.ones((5, 5), np.float32) / 15
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.medianBlur(img, 5)
    return img

#using hough cicrle
def cicles_detection(image,original_img):
  
    output = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cir_list=[]
    # detect circles in the image
    #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 3,300)   #3,300
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 3, 300)
    if circles  is not None:
        
        circles = np.round(circles[0,:]).astype("int")
        
        for (x,y,r) in circles:
            
            if   r<=50:
                cir_list.append((x,y,r))
                cv2.circle(output,(x,y),r,(0,255,0),4)
                cv2.rectangle(output,(x-5,y-5),(x+5,y+5),(0,128,255),-1) 
                cv2.circle(original_img,(x,y),r,(0,255,0),4)
                cv2.rectangle(original_img,(x-5,y-5),(x+5,y+5),(0,128,255),-1)  
        
    return output,original_img,cir_list 

# using contoure detection 
def rectangle_detection(img,original_img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret,thresh = cv2.threshold(gray,127,255,1)
   
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 10, 250)
   
    rec_list=[]
    _,contours,_ = cv2.findContours(edged,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        area = cv2.contourArea(cnt)
        if area <=200 :
            continue 
        hull = cv2.convexHull(cnt,returnPoints = True)
        
        if len(approx)==3:
            print ("triangle area",area)
            cv2.drawContours(img,[cnt],-1,(0,128,255),4)
            cv2.drawContours(original_img,[cnt],-1,(0,128,255),4)
        elif len(approx)==4:
            leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
            rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
            topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
            bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
            print(len(hull),"shimaa",leftmost,rightmost,topmost,bottommost)
            print ("square area",area )
            M = cv2.moments(cnt)
            rec_list.append([M['m10']/M['m00'],M['m01']/M['m00']])
            cv2.drawContours(img,[cnt],-1,(0,128,255),4)
            cv2.drawContours(original_img,[cnt],-1,(0,128,255),4)
    
    return img,original_img,rec_list

#for triangle
def template_matching(img,frame):

    rectangle_center=[]
    img2 = img.copy()
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('triangle_S2.png',0) # 1.jpg
    #template =cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    #template = cv2.resize(template,img.shape[0:2])
    template=cv2.resize(template,(50, 50))  #triangle
    # cv2.imshow("tmp",template)
    # cv2.waitKey(0)
    #template=cv2.resize(template,(50, 70))  #rectangle
    w, h = template.shape[::-1]
    #All the 6 methods for comparison in a list
    methods = [#'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR_NORMED']#, 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    for meth in methods:

        method = eval(meth)
        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        if max_val > 0.7 :
            print(max_val)
            rectangle_center.append((top_left[0] + 0.5*w, top_left[1] + 0.5*h))
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img2,top_left, bottom_right, 255, 2)
            cv2.rectangle(frame, top_left, bottom_right, 255, 2)
        # cv2.imshow('d',img)
        # cv2.waitKey(0)
    return img2,frame,rectangle_center

def invlidate_circles(rec_l,cir_l,frame):
    for x1,y1 in rec_l:
        for x2,y2,_ in cir_l:
            dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
            print("dis",dist)
            if dist > 90 :
                cv2.circle(frame,(x,y),r,(0,255,0),4)
    return frame        
            
def invalid_drawing(rec_l,cir_l,triangle,frame):
    checl_circl = []
    if len(rec_l) == 0 and len(triangle) == 0 and len(cir_l) != 0:
        for x,y,r in cir_l:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        return -1,frame
    elif len(rec_l)==0 and len(cir_l)!=0 and len(triangle) ==1 :
        print("yes")
        (x2,y2) = triangle[0]
        for i in range(0, len(cir_l)):
            x1,y1,_ = cir_l[i]
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            print("dis", dist)
            if dist > 90 :
                checl_circl.append(cir_l[i])
        return 0,checl_circl

def cnn_label(sign):
    return 1,80

def label_convert():
    with open('signnames.csv') as f:
        out_labels = f.readlines()
    f.close()
    return  out_labels

def prepare_out_img():
    nxt_frame = cv2.imread("black_6.jpg", 1)
    nxt_frame = cv2.resize(nxt_frame, (900, 800))
    o = cv2.imread("black_10.png",1)
    o = cv2.resize(o, (580, 580))
    nxt_frame[85:665, 25:605] = o[:, :]

    cv2.rectangle(nxt_frame, (613, 85), (860, 715), (255, 255, 255), -1)
    textt = "Traffic Sign "
    cv2.putText(nxt_frame, textt, (650, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 255, 255), 2,
                lineType=cv2.LINE_AA)
    g11 = cv2.imread("g11.png", 1)
    g11 = cv2.resize(g11, (200, 50))  # w,h
    nxt_frame[665:715, 25:225] = g11[:, :]
    g22 = cv2.imread("g22.png", 1)
    g22 = cv2.resize(g22, (200, 50))
    nxt_frame[665:715, 225:425] = g22[:, :]
    g33 = cv2.imread("g33.png", 1)
    g33 = cv2.resize(g33, (180, 50))
    nxt_frame[665:715, 425:605] = g33[:, :]
    return  nxt_frame

def voice_word():

    engine = pyttsx3.init()
    mm = 'Welcome'
    volume = engine.getProperty('volume')
    voices = engine.getProperty('voices')
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 50)
    engine.setProperty('voice', voices[1].id)  # 0 is man , 1 is woman
    engine.setProperty('volume', volume + 2000)
    #return  engine
    for i in range(0, 3):
        engine.say(mm)
    engine.runAndWait()

def recognition_2(cir_l,out_frame,nxt_frame):
    global count_times,out_imgs
    size = 120
    sp_x, sp_y = 615, 87
    l=[]
    for key, value in count_times.items():
        if value == 10:
            l.append(key)
        else:
            count_times[key] +=1
            nxt_frame[sp_y:sp_y + size, sp_x:sp_x + size] = out_imgs[key][0][:,:]
            textt = out_imgs[key][1]
            cv2.putText(nxt_frame, textt, (sp_x, sp_y + size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        lineType=cv2.LINE_AA)
            sp_y += size + 20
    for key in l:
        del count_times[key]
        del out_imgs[key]

    for x, y, r in cir_l:
        sign = out_frame[y - r - 2:y + r + 2, x - r - 2:x + r + 2].copy()
        try:
            sign2 = cv2.resize(sign, (50, 50))
        except:
            print("false")
            continue

        sign_label, prob = cnn_label(sign)
        if prob > 70:
            cv2.rectangle(out_frame, (x - r, y - r), (x + r, y + r), (0, 128, 255), 3)
            if sign_label in count_times:
                count_times[sign_label]=0
                continue

            sign = cv2.resize(sign, (size, size))

            nxt_frame[sp_y:sp_y+size, sp_x:sp_x+size] = sign[:, :]

            textt = out_labels[sign_label + 1].split(',')[1]
            cv2.putText(nxt_frame, textt, (sp_x, sp_y+size+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        lineType=cv2.LINE_AA)
            sp_y += size + 20
            count_times[sign_label] = 0
            out_imgs[sign_label] = [sign,textt]

    o = cv2.resize(out_frame, (580, 580))
    nxt_frame[85:665, 25:605] = o[:, :]
    return out_frame, nxt_frame

def recognition(cir_l,rec_l,out_frame):
    nxt_frame = cv2.imread("black.jpg",1)
    nxt_frame = cv2.resize(nxt_frame,(out_frame.shape[1],out_frame.shape[0]))
    count =0
    size = 120
    for x,y,r in cir_l:
        sign = out_frame[y-r-2:y+r+2,x-r-2:x+r+2].copy()
        try:
            sign2 = cv2.resize(sign,(50,50))
            # cv2.imshow("hagur",sign)
            # cv2.waitKey(0)
        except :
            print("false")
            continue

        count +=1
        sign_label,prob=cnn_label(sign)
        if prob > 70:
            print("yes")
            cv2.circle(out_frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(out_frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            sign =cv2.resize(sign,(size,size))
            str= size*(count-1)
            end = size*count
            nxt_frame[str:end,str:end] =  sign[:,:]
            textt = out_labels[sign_label +1].split(',')[1]
            cv2.putText(nxt_frame, textt, (end, end), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),2 ,lineType=cv2.LINE_AA)
            # cv2.imshow("shimaa", nxt_frame)
            # cv2.waitKey(0)


    return  out_frame,nxt_frame

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def correct_prightness(original):
    # loop over various values of gamma
    for gamma in [1.5]:
        # ignore when gamma is 1 (there will be no change to the image)
        if gamma == 1:
            continue

        # apply gamma correction and show the images
        gamma = gamma if gamma > 0 else 0.1
        adjusted = adjust_gamma(original, gamma=gamma)
        cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Images", np.hstack([original, adjusted]))
        cv2.waitKey(0)

cap = cv2.VideoCapture("v2.mp4") #v2.mp4 ,1280,720
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('tested.avi',fourcc, 20.0,(1280,720) ,1)
out_labels = label_convert()
tmp_img =prepare_out_img()

# out.write(tmp_img)
# out.write(tmp_img)
# out.write(tmp_img)
# cv2.namedWindow('hagur', cv2.WINDOW_NORMAL)
# cv2.imshow("hagur",tmp_img)
# cv2.waitKey(0)
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret==True:


        out_frame =  frame.copy()
        img = img_preprocessing(frame)

        img,frame,cir_l = cicles_detection(img,frame)
        img,frame,rec_l = rectangle_detection(img,frame)

        #out_frame,nxt = recognition(cir_l,rec_l,out_frame)
        out_frame, nxt = recognition_2(cir_l,out_frame,tmp_img.copy()) #.copy()

        #out.write(np.hstack([out_frame,nxt]))
        print(nxt.shape)
        out.write(nxt)
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.imshow("output",nxt)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
