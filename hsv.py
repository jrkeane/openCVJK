
''' proof of concept for the gunnery challenge aka RAN AMWC fall of shot accuracy
by jrk. on my own time
seeing how far computer vision will go before prepping it up and over to a ML
approach
latest addition 25/9: seeing where I'm up to then searching for a reliable horizon finder
'''

import cv2
import numpy as np
pathName = '/media/kookaburra/JimsDisk/AMWC Gunnery Footage/'
vidName = 'F150_20150121083643.mp4'
fileName = pathName + vidName
# print(fileName)
# playback = cv2.VideoCapture(fileName) # load video
cap = cv2.VideoCapture(fileName)

def nothing(x):
    pass

def set_canny_min(th_min):
    canny_min = cv2.getTrackbarPos('th_min', 'erosion')
    return canny_min

def set_canny_max(th_max):
    canny_max = cv2.getTrackbarPos('th_max', 'erosion')
    return canny_max

vidfactor = .8
vidwidth = int(cap.get(3)*vidfactor)
vidheight = int(cap.get(4)*vidfactor)

'''set up the environment'''
# Creating a window for later use
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.namedWindow('greyscale', cv2.WINDOW_NORMAL)
cv2.namedWindow('binary', cv2.WINDOW_NORMAL)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('original', vidwidth, vidheight)
cv2.resizeWindow('greyscale', vidwidth, vidheight)
cv2.resizeWindow('binary', vidwidth, vidheight)
cv2.resizeWindow('result', vidwidth, vidheight)
cv2.moveWindow('original', 150, 10)
cv2.moveWindow('greyscale', 1550, 20)
cv2.moveWindow('binary', 150, 900)
cv2.moveWindow('result', 1550, 900)

# Starting with 100's to prevent error while masking
h,s,v = 0,0,120
thresh_lim = 100
# Creating track bar
cv2.createTrackbar('th_min', 'result',20,120, set_canny_min)
cv2.createTrackbar('th_max', 'result',80,280, set_canny_max)
# cv2.createTrackbar('v', 'result',0,255,nothing)
canny_min = 80
canny_max = 100

framenum = 0.0
while(1):
    print(framenum)
    _, frame = cap.read()

    vidheight, vidwidth, channels = frame.shape

    # frame = cv2.GaussianBlur(frame, (25,25), 0) # apply gaussian filter

    h_min = int(vidheight*.05)
    h_max = int(vidheight*.95)
    # w_min = int(vidwidth*.1)
    l_w_min = 0
    l_w_max = int(vidwidth*.04)
    r_w_min = int(vidwidth*.96)
    r_w_max = int(vidwidth)

    left_bar = frame[h_min:h_max, l_w_min:l_w_max]
    right_bar = frame[h_min:h_max, r_w_min:r_w_max]

    hsv_left = cv2.cvtColor(left_bar, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0,0,100])
    upper_hsv = np.array([180,255,255])

    left_mask = cv2.inRange(hsv_left,lower_hsv, upper_hsv)

    cv2.imshow('left thresh', left_mask)
    # left_thresh = cv2.inRange(hsv_left, lower_hsv, upper_hsv)

    hsv_right = cv2.cvtColor(right_bar, cv2.COLOR_BGR2HSV)
    right_mask = cv2.inRange(hsv_right, lower_hsv, upper_hsv)
    # right_thresh = cv2.bitwise_and(right_bar, right_bar, mask = right_mask)

    cv2.imshow('right_thresh', right_mask)
    #TODO see why right bar is processing different to left.

    # hor_thresh_lim = 110
    # right_gray = cv2.cvtColor(right_bar, cv2.COLOR_BGR2GRAY)
    # right_gray=cv2.equalizeHist(right_gray)
    # cv2.imshow('right gray', right_gray)
    # ret, right_thresh = cv2.threshold(right_gray, hor_thresh_lim, 255, 0)
    # kernel = np.ones((15,15),np.uint8)
    # right_closed = cv2.morphologyEx(right_thresh, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow('right thresh', right_closed)

    #converting to HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # print(hsv)
    # get info from track bar and appy to result
    # h = cv2.getTrackbarPos('h','result')
    # s = cv2.getTrackbarPos('s','result')
    # threshlim = cv2.getTrackbarPos('v','result')

    # Normal masking algorithm
    lower_blue = np.array([0,0,110])
    upper_blue = np.array([180,255,255])

    mask = cv2.inRange(hsv,lower_blue, upper_blue)

    # mask = hsv[:,:,:]
    hsv_thresh = cv2.bitwise_and(frame,frame,mask = mask)

    thresh_lim = 100
    imgray = cv2.cvtColor(hsv_thresh, cv2.COLOR_BGR2GRAY)
    # imgray=cv2.equalizeHist(imgray)
    two_gray = cv2.GaussianBlur(imgray, (5,5), 0) # apply gaussian filter

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
    # imgray = clahe.apply(imgray)

    ret,thresh = cv2.threshold(imgray,thresh_lim, 255,0)
    thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((10,10),np.uint8)
    erosion = cv2.erode(frame, kernel, iterations = 1)
    # cv2.imshow('erosion', erosion)
    dilation = cv2.dilate(frame,kernel,iterations = 1)
    cv2.imshow('erosion', erosion)
    edges = cv2.Canny(two_gray, canny_min,canny_max)
    cv2.imshow('edges', edges)

    closed4sky = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('morphologyEx', closed4sky)
    inverted4sea = cv2.bitwise_not(closed4sky) # Invert floodfilled image

    ### left side
    thresh_sea, contours_sea, hierarchy_sea = cv2.findContours(inverted4sea[0:vidheight, l_w_min:l_w_max],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours_sea, -1, (0,255,0), 2)

    thresh_sky, contours_sky, hierarchy_sky = cv2.findContours(closed4sky[0:vidheight, l_w_min:l_w_max],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours_sky, -1, (0,0,255), 2)

    ### right side
    thresh_sea, contours_sea, hierarchy_sea = cv2.findContours(inverted4sea[0:vidheight, r_w_min:r_w_max],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame[0:vidheight, r_w_min:r_w_max], contours_sea, -1, (0,255,0), 2)

    thresh_sky, contours_sky, hierarchy_sky = cv2.findContours(closed4sky[0:vidheight, r_w_min:r_w_max],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame[0:vidheight, r_w_min:r_w_max], contours_sky, -1, (0,0,255), 2)

    if contours_sea:
        yofinterest = []
        for contour in contours_sea:
            #TODO get min y at right bar and left barself.
            # draw a line through those points
            # see how accurate we have the horizon tracked
            [x,y,w,h] = cv2.boundingRect(contour)
            yofinterest.append(y)

        # print(yofinterest)
        # highest_water = np.amin(yofinterest)
        # print(highest_water)




    hly = 0
    hry = vidwidth
    hlx = int(vidheight/2)
    hrx = int(vidheight/2)

    horizon = cv2.line(frame, (hly, hlx), (hry, hrx), (255,0,0),2)


    #
    cv2.imshow('frame',frame)
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
