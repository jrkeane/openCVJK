# playback_ini

# functions called in greenrooms playback.py
# grouped as functions for the video, then
# functions for the img processing
# preprocessing is the only function called in both

import os
import cv2
import numpy as np

# def output_frame(img_num, frame, framenum, savePathName): # save frames from video playback as jpeg
#
#
#     cropframe=frame[0:576, 0:748] #y:y+h, x:x+w
#     savename="Ninox"
#     framenum = int(framenum)
#     cropname = savePathName+str(savename)+str(framenum)+".jpg"
#     cv2.imwrite(cropname, cropframe)
#     k = cv2.waitKey(50) & 0xff
#
#     return


##########################################
###### Functions for img processing ######
##########################################


def setup_defaults(): # sets limits of onscreen text, & intensity and size thresholds

    horizon = 20 #pixel position of horizon (taken from top)
    top_left_text_limit = [0,0,100,340]
    bottom_text_limit = 20
    crosshair_limit = [350, 260, 40, 40]
    mid_left_text_limit = [0, 250, 65, 100]
    out_of_bounds = [horizon, top_left_text_limit, bottom_text_limit, crosshair_limit, mid_left_text_limit]
    threshlim_default = 155 #default thresh value to recall if the adaptive one changes
    box_limit_default = 10
    threshlim = threshlim_default
    box_limit_max = box_limit_default

    return out_of_bounds, threshlim, box_limit_max

def whereAmIup2(img_num_str, flightFolderName, directoryName): # write frame number to text so you can pick up from here next time

    fname = flightFolderName+"_whichFrameAmIUpTo_Num.txt"
    text_file_name = directoryName+fname
    f = open(text_file_name, 'w')
    f.write(img_num_str)
    f.close()

    return

def starting_point(imgPathName, flightFolderName): # pick up where you left off last time

    fName = "_whichFrameAmIUpTo_Num.txt"
    text_file_name = imgPathName+fName
    FileExists = os.path.exists(text_file_name)

    if flightFolderName == "flight1images":
        img_num_start = 4704.0 #first img num
        img_num_last = 207208.0

    elif flightFolderName == "flight2images":
        img_num_start = 78072.0
        img_num_last = 257220.0

    elif flightFolderName == "flight3images":
        img_num_start = 36568.0
        img_num_last = 161888.0

    img_num = img_num_start

    if FileExists == True: #if a file exists you'll pick up from where you left off last time
        f = open(text_file_name, 'r')
        string = f.read()
        img_num = int(string)
        f.close()

    print("starting at "+str(img_num_start))

    return img_num, img_num_start, img_num_last

def exportNum2xml(img_num, indexStr, savePathName, cropframe, numStr, pos): #exports all contacts for frame to xml

    print("savePathName:")
    print(savePathName)

    vidheight, vidwidth, channels = cropframe.shape
    img_num_str = str(img_num)
    print(img_num_str)
    vidwidth = str(vidwidth)
    vidheight = str(vidheight)

    folder = savePathName
    print("folder:  "+folder)
    filename = img_num_str+"_"+numStr+"_"+indexStr+".jpg"
    path = folder+filename

    text_file_name = folder+img_num_str+"_"+numStr+"_"+indexStr+".xml"

    print("exporting:" + text_file_name)
    nl = "\n" #new line
    tb = "\t" #tab
    dt = "\t\t" #double tab
    tt = "\t\t\t" #triple tab
    f =open(text_file_name,'w')
    f.write("<annotation>"+nl+tb+"<folder>"+folder+"</folder>"+nl) #where does harry want it
    f.write(tb+"<filename>"+filename+"</filename>"+nl)
    f.write(tb+"<path>"+path+"</path>"+nl)
    f.write(tb+"<source>"+nl+tb+tb+"<database>Unknown</database>"+nl+tb+"</source>"+nl)
    f.write(tb+"<size>"+nl+dt+"<width>"+vidwidth+"</width>"+nl)
    f.write(dt+"<height>"+vidheight+"</height>"+nl)
    f.write(dt+"<depth>3</depth>"+nl+tb+"</size>"+nl)
    f.write(tb+"<segmented>0</segmented>"+nl)

    xmin = str(pos[0])
    ymin = str(pos[1])
    xmax = str(pos[4])
    ymax = str(pos[5])

    f.write(tb+"<object>"+nl)
    f.write(dt+"<name>"+numStr+"</name>"+nl+dt+"<pose>Unspecified</pose>"+nl)
    f.write(dt+"<truncated>0</truncated>"+nl+dt+"<difficult>0</difficult>"+nl)
    f.write(dt+"<bndbox>"+nl)
    f.write(tt+"<xmin>"+xmin+"</xmin>"+nl)
    f.write(tt+"<ymin>"+ymin+"</ymin>"+nl)
    f.write(tt+"<xmax>"+xmax+"</xmax>"+nl)
    f.write(tt+"<ymax>"+ymax+"</ymax>"+nl)
    f.write(dt+"</bndbox>"+nl+tb+"</object>")
    f.write("</annotation>")
    f.close()

    return

def on_keystrokes_calibrate(k, threshlim, box_limit_max, out_of_bounds, full_fname, screens): # for calibrating thresholds

    print("Calibrating")
    print("left/right - adjust threshold")
    print("up/down    - adjust animal size")
    print("enter when done")

    while k is not 13:

        k = cv2.waitKey(0) & 0xff
        if k == 81: # left
            threshlim = threshlim - 2
            print("threshlim: "+str(threshlim))

        elif k == 83: # right
            threshlim = threshlim + 2

            if threshlim > 255:
                threshlim = 255
                print("threshlim at max")

            print("threshlim: "+str(threshlim))

        elif k == 82: # up
            box_limit_max = box_limit_max + 5
            print("Box limit: "+ str(box_limit_max))

        elif k == 84: # down
            box_limit_max = box_limit_max - 5
            if box_limit_max < 5:
                box_limit_max = 5
                print("box limit at minimum")

            print("Box limit: " + str(box_limit_max))

        elif k == 27:
            break

        cv2.destroyWindow("Detection")
        cv2.destroyWindow("Threshold")
        frame = cv2.imread(full_fname)
        set_windows(box_limit_max, frame, screens) # set up the screen
        closed_frame = preprocessing(threshlim, frame) # prep frame
        num_contours, contours_list = process_contours(closed_frame, box_limit_max, out_of_bounds, frame)

    return threshlim, box_limit_max, num_contours, contours_list, frame

def manual_select(frame, k, num_contours, contours_list): # for the user to manually select targets

    xl = contours_list[0]
    yl = contours_list[1]
    wl = contours_list[2]
    hl = contours_list[3]

    while k is not 13:

        print("Manually select targets")
        cali_box = cv2.selectROI("Detection", frame, True, False)
        x = int(cali_box[0])
        y = int(cali_box[1])
        w = int(cali_box[2])
        h = int(cali_box[3])
        boundimg = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0,0), 2)
        num_contours=num_contours+1
        xl.append(x)
        yl.append(y)
        wl.append(w)
        hl.append(h)
        contours_list = [xl, yl, wl, hl]
        print("contacts identified: " + str(num_contours))
        print("Hit Enter when done. Space to do another. Esc to cancel")
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break

    return contours_list, num_contours

def click_targets(event, x, y, flags, param): # part of the manual_select function

    global clickx, clicky
    if event == cv2.EVENT_LBUTTONDOWN:
        clickx = x
        clicky = y
        print(str(x)+", "+str(y))

    return

def delete_targets(contours_list, num_contours, frame): # if user wants to delete targets

    print(" ")
    print("CLICK target to delete then hit enter")
    xl = contours_list[0]
    yl = contours_list[1]
    wl = contours_list[2]
    hl = contours_list[3]

    while True:

        while True:

            kk = cv2.waitKey(1)
            cv2.setMouseCallback("Detection", click_targets)
            if kk == 13:
                break

        delX = clickx
        delY = clicky
        deletePt = (delX, delY)
        contacts_list = range(num_contours)
        deleted = 0
        for i in reversed (contacts_list):
            x = int(xl[i]-5)
            y = int(yl[i]-5)
            w = int(wl[i]+10)
            h = int(hl[i]+10)
            if delX > x and delX < (x+w) and delY > y and delY < (y+h):
                if deleted > 0:
                    print("and this one ("+str(x)+", "+str(y)+")")

                else:
                    print("deleted this one ("+str(x)+", "+str(y)+")")

                boundimg = cv2.rectangle(frame, (xl[i], yl[i]), (xl[i]+wl[i], yl[i]+hl[i]), (255, 255,255), 2)
                cv2.imshow("Detection", frame)
                del xl[i]
                del yl[i]
                del wl[i]
                del hl[i]
                num_contours = num_contours - 1
                deleted = deleted+1

        contours_list = [xl, yl, wl, hl]
        print("OPTIONS: ")
        print("space - do another")
        print("enter - finished deleting")
        k = cv2.waitKey(0) & 0xff

        if k == 13:
            break

        if k == 27:
            break

    return contours_list, num_contours, frame

def mlROI(frame, k): # focuses on region of interst for ML to read to generate a log file
    # each box is a region of inerest
    #e.g set up to read heading, but add points for additional regions
    boxx = [22] # box start point (pixels from left of screen)
    boxy = [40] # box start point (pixels from Top! of screen)
    boxw = [40] # box width
    boxh = [20] # box height

    for i in range(len(boxx)):
        while k is not 13:
            trackbox = (boxx[i],boxy[i],boxw[i], boxh[i])
            p1 = (int(trackbox[0]), int(trackbox[1]))
            p2 = (int(trackbox[0] + trackbox[2]), int(trackbox[1] + trackbox[3]))
            MLframe=frame[p1[1]:p2[1], p1[0]:p2[0]]

            if k == 81:
                bbox[i] = bbox[i] - 10

            popup = cv2.imshow("readMeMl", MLframe)
            k = cv2.waitKey(0)&0xff

    return
