# gunnery
# function file to gunnery.py
# project by JR Keane

import cv2
import numpy as np
import math
import os

def display_author_info(frame, txtcount):

    vidwidth, vidheight, channels = frame.shape
    textx = int(vidwidth/20)
    texty = int(vidheight/20)
    text2y = texty+40
    text2x = textx
    text3y = text2y+40
    text4y = text3y+40
    notex=int(vidwidth/20)
    notey=int(text4y+40)
    text = "TARGET PRACTICE"
    text2 = "JR KEANE"
    text3 = "VERSION 1.1"
    text4 = "31 Aug 17"
    note_text = "NOTE TO DO: REQUIRE CAMERA SPECIFICATIONS AND DISTANCE-TO-TARGET "
    note2_text="TO CALIBRATE PIXEL2METRE CONVERSION CALCULATIONS"

    if txtcount<30:
        title = cv2.putText(frame, text, (textx, texty), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        author = cv2.putText(frame, text2, (text2x, text2y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        ran = cv2.putText(frame, text3, (text2x, text3y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        date = cv2.putText(frame, text4, (text2x, text4y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        note = cv2.putText(frame, note_text, (notex, notey), cv2.FONT_HERSHEY_SIMPLEX, 1, (250,0,0), 2)
        note2 = cv2.putText(frame, note2_text, (notex, notey+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (250,0,0), 2)

    else:
        title = cv2.putText(frame, text, (textx, texty), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        author = cv2.putText(frame, text2, (text2x, text2y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        ran = cv2.putText(frame, text3, (text2x, text3y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        date = cv2.putText(frame, text4, (text2x, text4y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    return


def findingHorizon(frame, demo):

    vidwidth, vidheight, channels = frame.shape
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray, (25,25), 0) # apply gaussian filter
    thresh_lim = 135 #125
    ret,thresh = cv2.threshold(imgray,thresh_lim,255,0)
    kernel = np.ones((25,25),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    inverted = cv2.bitwise_not(closing) # Invert floodfilled image

    thresh, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # find contoured objects
    bbox_sky, rightmost = findingSky(vidwidth, contours,imgray)
    thresh, contours, hierarchy = cv2.findContours(inverted,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    bbox_sea, bbox2y = findingOcean(vidwidth, vidheight, contours, imgray)

    if demo == 'y':
        cv2.imshow("sky and sea", thresh)

    return thresh, imgray, bbox_sky, bbox_sea, rightmost, bbox2y

def findingOcean(vidwidth, vidheight, contours,imgray):

    bbox_sea=[]
    for contour in contours:
        [xo,yo,wo,ho] = cv2.boundingRect(contour)
        horizonxmin = vidwidth/100
        horizonxmax = vidwidth - vidwidth/100
        horizonymin = vidheight*3/4
        wmin = vidwidth*.75

        if xo<horizonxmin and xo+wo > horizonxmax and yo<horizonymin and wo>wmin:
            bbox_sea = [xo,yo,wo,ho]
            hull = cv2.convexHull(contour,returnPoints = False)
            rect = cv2.minAreaRect(contour)
            rectbox = cv2.boxPoints(rect)
            box = np.int0(rectbox)
            ys=[]

            for i in range(len(box)):
                yval = box[i][1]
                ys.append(yval)

            bbox2y = min(ys)
            text = "horizon"

        else:
            continue

    return bbox_sea, bbox2y

def findingSky(vidwidth, contours,imgray):

    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        horizonxmin = vidwidth/100
        horizonxmax = vidwidth - vidwidth/100
        horizonymin = 100

        if x<horizonxmin and x+w > horizonxmax and y<10:
            bbox_sky = [x,y,w,h]
            hull = cv2.convexHull(contour,returnPoints = False)
            rect = cv2.minAreaRect(contour)
            rectbox = cv2.boxPoints(rect)
            box = np.int0(rectbox)
            leftmost = tuple(contour[contour[:,:,0].argmin()][0])
            rightmost = tuple(contour[contour[:,:,0].argmax()][0])

        else:
            continue

    return bbox_sky,rightmost


def horizonStabilizer(vidwidth, vidheight, bbox_sky, bbox_sea, rightmost, bbox2y):

    x = bbox_sky[0]
    y = bbox_sky[1]
    w = bbox_sky[2]
    h = bbox_sky[3]
    ya = int(y + h)
    yarange = list(range(ya-2, ya+2))
    yb = int(bbox2y)
    right = int(vidwidth)
    rightmosty = rightmost[1]

    if rightmosty in yarange:
        print("rotate anti-clockwise")
        height = (rightmosty-yb)
        rad_angle=math.atan(height/vidwidth)
        rotatey=yb
        middle_earth=int((rightmosty-yb)/2+yb)
        hor_corr = "anticlockwise"

    else:
        # cv2.line(frame, (left, ya), (right, rightmosty), (0,0,255),2)
        print("rotate clockwise")
        height = (rightmosty-ya)
        rad_angle=math.atan(height/vidwidth)
        rotatey=ya
        middle_earth=int((ya-rightmosty)/2+rightmosty)
        hor_corr = "clockwise"

    rot_angle=rad_angle*180/math.pi
    center=(0,rotatey)

    return rot_angle,rotatey,middle_earth,hor_corr

def splashRotateDetect(flatspin, horizon_boundaries):

    x_adj = horizon_boundaries[0]
    y_adj = horizon_boundaries[1]
    x2adj = horizon_boundaries[2]
    y2adj = horizon_boundaries[3]
    thresh_lim = 190 # needs to become adaptive
    splashbox=[]

    grey_c = cv2.cvtColor(flatspin, cv2.COLOR_BGR2GRAY)
    ret,thresh_c = cv2.threshold(grey_c,thresh_lim,255,0)
    kernel = np.ones((20,20),np.uint8)
    closing_c = cv2.morphologyEx(thresh_c, cv2.MORPH_CLOSE, kernel)
    thresh, contours_c, hierarchy = cv2.findContours(closing_c,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours_c, -1, (0,255,0), 1)

    for contour in contours_c:
        [xspl,yspl,wspl,hspl] = cv2.boundingRect(contour)
        splashbox = [xspl,yspl,wspl,hspl]
        boundimg = cv2.rectangle(flatspin,(xspl,yspl),(xspl+wspl,yspl+hspl),(0,0,244),2)

    if splashbox:
        xspl = xspl + x_adj - wspl
        yspl = yspl + y_adj
        wspl=10
        hspl=30
        print("splash")

    return closing_c, splashbox

def splashDetect(horizon_boundaries, frame, zoom, hor_corr, demo):

    x_adj = horizon_boundaries[0]
    y_adj = horizon_boundaries[1]
    x2adj = horizon_boundaries[2]
    y2adj = horizon_boundaries[3]

    thresh_lim = 190 #180
    grey_c = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
    ret,thresh_c = cv2.threshold(grey_c,thresh_lim,255,0)
    kernel = np.ones((20,20),np.uint8)
    closing_c = cv2.morphologyEx(thresh_c, cv2.MORPH_CLOSE, kernel)
    thresh, contours_c, hierarchy = cv2.findContours(closing_c,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    zoomw = x2adj-x_adj
    zoomh = y2adj-y_adj
    splashbox=[]

    for contour in contours_c:
        [xspl,yspl,wspl,hspl] = cv2.boundingRect(contour)
        simtri = int(xspl*zoomh/zoomw) #use similar triangles to check object on horizon.
        h_simtri = simtri

        if hor_corr=="clockwise":
            h_simtri = zoomh-simtri

        h_low=h_simtri-15
        h_high=h_simtri+15
        h_range = list(range(h_low,h_high))
        splashbase = yspl+hspl

        if splashbase in h_range:
            splashbox = [xspl,yspl,wspl,hspl]
            boundimg = cv2.rectangle(zoom,(xspl,yspl),(xspl+wspl,yspl+hspl),(0,0,244),2)
            xspl = xspl + x_adj - wspl
            yspl = yspl + y_adj
            wspl=10
            hspl=30
            print("IMPACT")
            text = "IMPACT"
            textx = xspl+wspl
            texty = yspl+hspl
            status = cv2.putText(frame, text, (textx, texty), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    if demo == 'y':
        cv2.imshow("closing_c",closing_c)

    return closing_c, splashbox

def red_rope(frame, middle_earth):

    vidwidth, vidheight, channels = frame.shape

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) # define the lower (a) and upper(b) filter limits: HSV or BGR 0 - 255
    a1 = 130
    a2 = 0
    a3 = 0
    b1 = 190 # H 0 - 180 likely conversions to HSV
    b2 = 150 # S 0 - 255
    b3 = 150 # V 0 - 255
    lower_color = np.array([a1,a2,a3]) # hsv hue sat value
    upper_color = np.array([b1,b2,b3])
    mask = cv2.inRange(hsv, lower_color, upper_color) # find out color and display in black
    # res = cv2.bitwise_and(im, im, mask = mask)
    kernel = np.ones((15,15),np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # inverted = cv2.bitwise_not(closing) # Invert floodfilled image
    thresh, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # find contoured objects

    hmin = 150
    wmax = 200

    for contour in contours: #for all contours detected
        [x,y,w,h] = cv2.boundingRect(contour)

        if h>hmin and w<wmax: # conditions for rope contour
            rows,cols = mask.shape[:2]
            [vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
            # Now find two extreme points on the line (out of frame)
            rope_line_windowleft_y = int((-x*vy/vx) + y) # pixels above y = 0 where line intersects frame edge
            rope_line_windowright_y = abs(int(((frame.shape[1]-x)*vy/vx)+y)) # pixels below y = 0 where line intersects frame edge
            total_line_height = rope_line_windowleft_y+rope_line_windowright_y
            red_line_base = frame.shape[1]-1
            target_y = middle_earth + 8 # arbirtray 8
            simTri_base = int(target_y+rope_line_windowright_y)
            target_x = int(red_line_base/total_line_height*simTri_base)
            target_x = red_line_base - target_x
            cv2.line(frame,(x, y),(target_x, target_y),(0,0,100),3)
            ship_x = x #output ship posn for later calcs
            ship_y = y

        else:
            continue

    return target_x, target_y, ship_x, ship_y

def missingMissiles(frame, target_x, target_y, splashbox, y_adj):

    vidwidth, vidheight, channels = frame.shape
    miss_x = abs(splashbox[0]-target_x)
    print("miss by " +str(miss_x))
    splashy=splashbox[1]+y_adj+splashbox[3]
    cv2.line(frame,(target_x, target_y),(splashbox[0], splashy),(0,30,0),1)
    if splashbox[0]>target_x:
        r_or_l = "RIGHT"

    else:
        r_or_l = "LEFT"

    hor_m = 2000 #assumed camera field of 2000m
    pix2m = hor_m/vidwidth
    miss_x = int(miss_x*pix2m)
    splash_text = "SPLASH AND MISS BY "+str(miss_x)+ " M ("+str(r_or_l)+")."
    #miss = cv2.putText(frame, splash_text, (textx, texty), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,0,0), 2)


    return splash_text

def openLogFile(vidName, pathName): # write frame number to text so you can pick up from here next time

    fname = vidName+"_log.txt"
    text_file_name = pathName+fname

    FileExists = os.path.exists(text_file_name)
    # if FileExists == True: #if a file exists you'll pick up from where you left off last time
    #     log_f = open(text_file_name, 'r')
    #     string = f.read()
    #     img_num = int(string)
    #     log_f.close()

    log_f = open(text_file_name, 'w')
    log_f.write("new log file for \n" )
    log_f.write(text_file_name +'\n')

    return log_f

def writeLogFile(log_f, framenum, splash_text):

    nl = "\n" #new line
    log_f.write(nl)
    log_f.write(str(framenum)+nl)
    log_f.write(splash_text+nl)

    return
