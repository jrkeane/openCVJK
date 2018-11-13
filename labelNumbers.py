#!/usr/local/bin/python3

# Greenroom playback by Jim Keane
# james@greenroomrobotics.com
# last update Fri, 20 Jul 18
# script for labeling numbers using CV to train machine learning models heaps quicker


import cv2
import time
import numpy as np
import os
from os import path
import labelNumbersInit as lni

flightFolderName = 'flight1images'
imgPathName = '/media/kookaburra/JimsDisk/Ninox/flight1images/'
savePathName = '/media/kookaburra/JimsDisk/Ninox/numbers/'

progressName = "progressNum.txt"
starting_point_file_name = '/media/kookaburra/JimsDisk/Ninox/'+progressName
img_num_start = 4704.0 #first img num
img_num_last = 207208.0
framerate = 8
# cut down on the number of frames you're looking at in the vid (ninox frame rate 24fps)
VI = "img"
#vid or img:
# use vid for saving frames for cv processing, then img (for saving outputs for ML)

if VI=="img": # process frames with animals to and export an xml file for ml training

    rewind = 'false' # just accept it
    f = open(starting_point_file_name, 'r')
    string = f.read()
    img_num = int(string)
    f.close()
    # img_num, img_num_start, img_num_last = lni.starting_point(imgPathName, flightFolderName) # pick up where you left off
    # print(img_num)
    print("starting from: " +str(img_num_start))
    print("end at: " + str(img_num_last))
    (out_of_bounds, threshlim, box_limit_max) = lni.setup_defaults() # set your defaults

    headingx = [33, 45, 53] # box start point (pixels from left of screen)
    headingy = [50, 50, 50] # box start point (pixels from Top! of screen)
    headingw = [11, 9, 11] # box width
    headingh = [15, 15, 15] # box height

    gpsSx = [163, 173, 180, 185, 195, 203, 212, 221, 230]
    gpsSy = [553, 553, 553, 553, 553, 553, 553, 553, 553]
    gpsSw = [10, 10, 5, 10, 10, 10, 10, 10, 10]
    gpsSh = [15, 15, 15, 15, 15, 15, 15, 15, 15]

    gpsEx = [258, 266, 276, 285, 289, 298, 307, 316, 324, 334]
    gpsEy = [553, 553, 553, 553, 553, 553, 553, 553, 553, 553]
    gpsEw = [10, 10, 10, 5, 10, 10, 10, 10, 10, 10]
    gpsEh = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15]

    boxx = headingx+gpsSx+gpsEx
    boxy = headingy+gpsSy+gpsEy
    boxw = headingw+gpsSw+gpsEw
    boxh = headingh+gpsSh+gpsEh

    while(img_num <= img_num_last):

        img_num_str = str(img_num)
        imgname = "Ninox"+img_num_str+".jpg"

        full_fname = imgPathName+imgname
        frame = cv2.imread(full_fname) # read the frame from the folder
        if frame is None: # if no frame to load
            if rewind == 'true': # rewind
                if img_num == img_num_start: # if you're at the start don't rewind further
                    rewind = 'false'

                elif img_num < img_num_start:
                    img_num = img_num_start
                    rewind = 'false'

                else:
                    img_num = int(img_num-framerate)

            else: # go forward, find next
                img_num = int(img_num+framerate)
                # rewind = 'false'

        else: #if there's a frame to load
            print(img_num)
            rewind = 'false' # reset this
            vidheight, vidwidth, channels = frame.shape # get frame props
            # lni.set_windows(box_limit_max, frame, screens) # set up the screen
            newvidwidth = int(vidwidth*3)
            newvidheight = int(vidheight*3)


            # closed_frame = lni.preprocessing(threshlim, frame) # prep frame
            # lni.printTextFrameLoad(full_fname, threshlim)
            # k = cv2.waitKey(0) & 0xff # wait for input: ent, space, del, or esc

            numberLabel =[]

            for i in range(len(boxx)):
                k = 0
                while k is not 13:
                    if k == 81:                 # left key slides frame left
                        boxx[i] = boxx[i] - 1
                    elif k == 83:               # right key slides box right
                        boxx[i] = boxx[i] + 1
                    elif k == 82:               # up slides box up
                        boxy[i] = boxy[i] - 1
                    elif k == 84:               # down slides box down
                        boxy[i] = boxy[i] + 1
                    elif k == 97:               # a decreases width
                        boxw[i] = boxw[i] - 2
                        boxx[i] = boxx[i] + 1
                    elif k == 115:              # w increases height
                        boxh[i] = boxh[i] + 2
                        boxy[i] = boxy[i] - 1
                    elif k == 119:              # s decreases height
                        boxh[i] = boxh[i] - 2
                        boxy[i] = boxy[i] + 1
                    elif k == 100:              # d increases width
                        boxw[i] = boxw[i] + 2
                        boxx[i] = boxx[i] - 1

                    trackbox = (boxx[i],boxy[i],boxw[i], boxh[i])
                    p1 = (int(trackbox[0]), int(trackbox[1]))
                    p2 = (int(trackbox[0] + trackbox[2]),
                            int(trackbox[1] + trackbox[3]))
                    MLframe=frame[p1[1]:p2[1], p1[0]:p2[0]]

                    popup = cv2.imshow("readMeMl", MLframe)

                    k = cv2.waitKey(0)&0xff
                    if k == 27:
                        break

                    if k == 49:
                        num=1
                        numStr='One'
                        k = 13
                    elif k == 50:
                        num=2
                        numStr='Two'
                        k = 13
                    elif k == 51:
                        num=3
                        numStr='Three'
                        k = 13
                    elif k == 52:
                        num=4
                        numStr='Four'
                        k = 13
                    elif k == 53:
                        num=5
                        numStr='Five'
                        k = 13
                    elif k == 54:
                        num=6
                        numStr='Six'
                        k = 13
                    elif k == 55:
                        num=7
                        numStr='Seven'
                        k = 13
                    elif k == 56:
                        num=8
                        numStr='Eight'
                        k = 13
                    elif k == 57:
                        num=9
                        numStr='Nine'
                        k = 13
                    elif k == 48:
                        num=0
                        numStr='Zero'
                        k = 13

                if k == 13:
                    # numberLabel.append(number)
                    # number location
                    x = boxx[i]
                    y = boxy[i]
                    w = boxw[i]
                    h = boxh[i]

                    #location of outer frame
                    xx = x - 1
                    yy = y - 1
                    ww = w + 2
                    hh = h + 2
                    yy2 = yy + hh
                    xx2 = xx + ww
                    xmin = str(1)
                    ymin = str(1)
                    xmax = str(ww-1)
                    ymax = str(hh-1)
                    # minMax = [xmin, ymin, xmax, ymax]

                    # pospos
                    # minMax = [x,y, xx2, yy2]
                    print(img_num)
                    print(numStr)

                    indexStr = str(i)
                    cropframe=frame[yy:yy2, xx:xx2]
                    saveName = "Ninox"+img_num_str+'_'+indexStr+'_'+numStr
                    saveFig = savePathName+saveName+'.jpg'
                    saveXML = savePathName+saveName+'.xml'
                    print(saveFig)
                    cv2.imwrite(saveFig, cropframe)
                    print(saveXML)
                    print(savePathName)
                    # lni.exportNum2xml(img_num, savePathName, indexStr, cropframe, numStr, pos)

                    print("savePathName:")
                    print(savePathName)

                    vidheight, vidwidth, channels = cropframe.shape
                    img_num_str = str(img_num)
                    print(img_num_str)
                    vidwidth = str(vidwidth)
                    vidheight = str(vidheight)

                    folder = 'numbers'
                    filename = img_num_str+"_"+numStr+"_"+indexStr+".jpg"
                    path = '/home/greenroom/Desktop/pre_training_images/numbers/'+filename

                    ##################
                    ## generate XML ##
                    ##################

                    nl = "\n" #new line
                    tb = "\t" #tab
                    dt = "\t\t" #double tab
                    tt = "\t\t\t" #triple tab
                    f =open(saveXML,'w')
                    f.write("<annotation>"+nl+tb+"<folder>"+folder+"</folder>"+nl) #where does harry want it
                    f.write(tb+"<filename>"+filename+"</filename>"+nl)
                    f.write(tb+"<path>"+path+"</path>"+nl)
                    f.write(tb+"<source>"+nl+tb+tb+"<database>Unknown</database>"+nl+tb+"</source>"+nl)
                    f.write(tb+"<size>"+nl+dt+"<width>"+vidwidth+"</width>"+nl)
                    f.write(dt+"<height>"+vidheight+"</height>"+nl)
                    f.write(dt+"<depth>3</depth>"+nl+tb+"</size>"+nl)
                    f.write(tb+"<segmented>0</segmented>"+nl)

                    # xmin = str(pos[0])
                    # ymin = str(pos[1])
                    # xmax = str(pos[4])
                    # ymax = str(pos[5])

                    f.write(tb+"<object>"+nl)
                    f.write(dt+"<name>"+numStr+"</name>"+nl+dt+"<pose>Unspecified</pose>"+nl)
                    f.write(dt+"<truncated>0</truncated>"+nl+dt+"<difficult>0</difficult>"+nl)
                    f.write(dt+"<bndbox>"+nl)
                    f.write(tt+"<xmin>"+xmin+"</xmin>"+nl)
                    f.write(tt+"<ymin>"+ymin+"</ymin>"+nl)
                    f.write(tt+"<xmax>"+xmax+"</xmax>"+nl)
                    f.write(tt+"<ymax>"+ymax+"</ymax>"+nl)
                    f.write(dt+"</bndbox>"+nl+tb+"</object>"+nl)
                    f.write("</annotation>")
                    f.close()

            print("all done here, next frame ")

            img_num = img_num+framerate # next frame


            if k == 32: #space bar
                img_num = img_num+framerate
                print("skip to next frame")
                rewind == 'false'

            elif k == 8: #backspace
                img_num = img_num-framerate
                print("rewind")
                rewind = 'true'

            elif k == 27: #esc - exit
                break


# lni.whereAmIup2(img_num_str, flightFolderName, directoryName)
fname = flightFolderName+"_whichFrameAmIUpTo_Num.txt"
f = open(starting_point_file_name, 'w')
f.write(img_num_str)
f.close()

cv2.destroyAllWindows() ## all done. close everything.
