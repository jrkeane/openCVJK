
''' proof of concept for the gunnery challenge aka RAN AMWC fall of shot accuracy
by jrk. on my own time
seeing how far computer vision will go before prepping it up and over to a ML
approach
latest addition 25/9: seeing where I'm up to then searching for a reliable horizon finder
27/9 working on horizon finder
'''
#TODO:
'''incorporate a watchdog also
find red rope
find the splash
'''

import cv2
import numpy as np

pathName = '/media/kookaburra/JimsDisk/AMWC Gunnery Footage/'
vidName = 'F150_20150121083643.mp4'
fileName = pathName + vidName
# print(fileName)
# playback = cv2.VideoCapture(fileName) # load video
cap = cv2.VideoCapture(fileName)

vidfactor = .8
vidwidth = int(cap.get(3)*vidfactor)
vidheight = int(cap.get(4)*vidfactor)

'''set up the environment'''
# Creating a window for later use
# cv2.namedWindow('original', cv2.WINDOW_NORMAL)
# cv2.namedWindow('greyscale', cv2.WINDOW_NORMAL)
# cv2.namedWindow('binary', cv2.WINDOW_NORMAL)
# cv2.namedWindow('result', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('original', vidwidth, vidheight)
# cv2.resizeWindow('greyscale', vidwidth, vidheight)
# cv2.resizeWindow('binary', vidwidth, vidheight)
# cv2.resizeWindow('result', vidwidth, vidheight)
# cv2.moveWindow('original', 150, 10)
# cv2.moveWindow('greyscale', 1550, 20)
# cv2.moveWindow('binary', 150, 900)
# cv2.moveWindow('result', 1550, 900)

# Starting with 100's to prevent error while masking
h,s,v = 0,0,120
thresh_lim = 100
# Creating track bar
# cv2.createTrackbar('th_min', 'result',20,120, set_canny_min)
# cv2.createTrackbar('th_max', 'result',80,280, set_canny_max)
# cv2.createTrackbar('v', 'result',0,255,nothing)
canny_min = 80
canny_max = 100

def find_horizon(frame,):

    vidheight, vidwidth, channels = frame.shape

    h_min = int(vidheight*.05)
    h_max = int(vidheight*.95)
    # w_min = int(vidwidth*.1)
    l_w_min = 0
    l_w_max = int(vidwidth*.04)
    r_w_min = int(vidwidth*.96)
    r_w_max = int(vidwidth)

    #converting to HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # Normal masking algorithm
    lower_blue = np.array([0,0,110])
    upper_blue = np.array([180,255,255])

    mask = cv2.inRange(hsv,lower_blue, upper_blue)

    # mask = hsv[:,:,:]
    hsv_thresh = cv2.bitwise_and(frame,frame,mask = mask)
    thresh_lim = 100
    kernel = np.ones((5,5),np.uint8)
    imgray = cv2.cvtColor(hsv_thresh, cv2.COLOR_BGR2GRAY)
    # imgray=cv2.equalizeHist(imgray)
    blur_gray = cv2.GaussianBlur(imgray, (5,5), 0) # apply gaussian filter


    ret,thresh = cv2.threshold(imgray,thresh_lim, 255,0)
    thresh = cv2.bitwise_not(thresh)
    closed4sky = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    # cv2.imshow('morphologyEx', closed4sky)
    inverted4sea = cv2.bitwise_not(closed4sky) # Invert floodfilled image

    thresh_sea, contours_sea, hierarchy_sea = cv2.findContours(inverted4sea[0:vidheight, l_w_min:l_w_max],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours_sea, -1, (0,255,0), 2)

    thresh_sky, contours_sky, hierarchy_sky = cv2.findContours(closed4sky[0:vidheight, l_w_min:l_w_max],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours_sky, -1, (0,0,255), 2)

    if contours_sky:
        contours_left = []
        for contour in contours_sky:
            [x,y,w,h]=cv2.boundingRect(contour)
            contours_left.append(y)
    hor_left = min(contours_left)

    # ### right side
    thresh_sea, contours_sea, hierarchy_sea = cv2.findContours(inverted4sea[0:vidheight, r_w_min:r_w_max],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame[0:vidheight, r_w_min:r_w_max], contours_sea, -1, (0,255,0), 2)

    thresh_sky, contours_sky, hierarchy_sky = cv2.findContours(closed4sky[0:vidheight, r_w_min:r_w_max],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame[0:vidheight, r_w_min:r_w_max], contours_sky, -1, (0,0,255), 2)

    if contours_sky:
        contours_right = []
        for contour in contours_sky:
            [x,y,w,h]=cv2.boundingRect(contour)
            contours_right.append(y)
    hor_right = min(contours_right)
    # l_w_max = int(vidwidth*.04)
    # r_w_min = int(vidwidth*.96)
    hlx = l_w_max
    hrx = r_w_min
    hly = hor_left
    hry = hor_right

    # horizon = cv2.line(frame, (hlx, hly), (hrx, hry), (255,0,0),2)
    if hly > hry:
        hly = hor_right
        hry = hor_left

    # horizon_frame = [hlx, hly, hrx, hry]
    return hlx, hly, hrx, hry

def adj_horizon(hly, hry):

    h_diff = abs(hry - hly)
    h_mid = int((hry+hly)/2)

    if h_diff<100:
        fly=h_mid-50
        fry=h_mid+50
    else:
        fly=hly-25
        fry=hry+25

    return fly, fry

framenum = 0.0
while(1):
    print(framenum)
    _, frame = cap.read()

    hlx, hly, hrx, hry = find_horizon(frame)

    fly, fry = adj_horizon(hly, hry)

    hor_sq = cv2.rectangle(frame, (hlx, fly), (hrx, fry), (255,255,255),3)
    # cv2.imshow('frame',frame)

    f = frame[fly:fry, hlx:hrx]
    fheight, fwidth, __ = f.shape

    hor_h_dif = hry - hly
    horizon = cv2.line(f, (0, hor_h_dif), (fwidth, 0), (100,100,0),1)
    cv2.imshow('f',f)

    # left_bar = frame[h_min:h_max, l_w_min:l_w_max]
    # right_bar = frame[h_min:h_max, r_w_min:r_w_max]

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(f, kernel, iterations = 3)
    erosion_gray = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    erosion_gray = cv2.equalizeHist(erosion_gray)
    # ret, erosion_thresh = cv2.threshold(erosion_gray,100,255,0)
    cv2.imshow('eg', erosion_gray)
    # cv2.imshow('erosion', erosion_thresh)
    # # dilation = cv2.dilate(frame,kernel,iterations = 1)
    # # cv2.imshow('erosion', erosion_gray)
    edges = cv2.Canny(f, canny_min, canny_max)
    # cv2.imshow('edges', edges)erosion

    # cv2.imshow('original', left_bar)
    # cv2.imshow('greyscale', imgray)
    # cv2.imshow('binary',thresh)
    # cv2.imshow('result', right_bar)

    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break
    framenum+=1

cap.release()
cv2.destroyAllWindows()
