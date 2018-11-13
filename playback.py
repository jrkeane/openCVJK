#!/usr/local/bin/python3

# Greenroom playback by Jim Keane
# james@greenroomrobotics.com
# last update Fri, 20 Jul 18

# playback exists to speed up the process for getting training images out of videos for greenrooms ML
# there are two parts to playback:
# 1) grabbing frames from a video,
# 2) going back through those frames and targeting images for training

# stage 1: video playback
# stage 1 runs a Video.. watch it then pound your keyboard when you seen an animal
# this will save the frame with an animal into as a jpeg
# you can fast forward through boring spots, or go slow mo if you think somethings coming up.
# there's no rewind function yet
# follow the intructions

# stage 2: frame processing
# stage 2 goes through the frames that you saved from stage 1
# you can either skip frames, process them, or rewind
# if you process a frame you have options to add or delete contacts if the cv has done a shit job
# follow the bouncing ball
# sometimes you might have to double tap keys in to get a response so it's not really user friendly yet

# the end product is xml files with the onscreen location of animals
# the xml files are in a format that should work for training tensor flow

# playback_ini contains all the functions for playback to work


import cv2
import time
import numpy as np
import os
from os import path
import playback_ini as pl

# Directory Structure:
    # directoryName = parent folder (save playback & ini files here)
    # IMGpathName = folder within parent that images get stored to then called from

directoryName = '/home/ubuntu/openCV/GreenRoom/Ninox/' # currently running off a USB, but change it.
flightFolderName = 'flight1images' # images go into here
IMGpathName = directoryName+flightFolderName+'/'
saveDirectoryName = IMGpathName # change this if you want to save frames from vid processing to somewhere else
#config
screens = 1 # 1 or 2 if you have a monitor attached
framerate = 8 # cut down on the number of frames you're looking at in the vid (ninox frame rate 24fps)
VI = "img" #vid or img: use vid for saving frames for cv processing, then img (for saving outputs for ML)
save_vid_screens = False

if VI == "vid": # play video to find frames with likely animals
    starting_point = 0.0
    vidname = 'flight3friday2june.avi'
    full_fname = directoryName+vidname
    box_limit_max = 0
    david, waittime, totalframes = pl.setup_vid(vidname, full_fname, box_limit_max, starting_point)
    pl.printVideoPlaybackInstructions() #follow the bouncing ball

    while(1): ## start video playback

        framenum = david.get(1) # get next frame from video
        print(framenum)
        trackerok, frame = david.read()

        if not trackerok: #check frame loaded
            break

        if framenum % framerate == 0: # only show 1 in x frames (helps to avoid a seizure)
            print("frame " + str(framenum))
            cv2.imshow("Detection", frame)
            # closed_frame = pl.preprocessing(threshlim, frame) # convert img to greyscale then binary for processing
            # threshlim = cv2.getTrackbarPos('thrs1', 'Threshold') # if the slidebar is set up
            k = cv2.waitKey(waittime) & 0xff #micropause

            if k == 27: #hit esc, end loop
                break

            waittime = pl.keystrokes_during_video(k, vidname, frame,framenum, waittime, saveDirectoryName, save_vid_screens) # any UI?

        if framenum > totalframes: # break if there are no more frames to analyse
            print("end of vid. break.")
            break

    david.release() # stop reading video

elif VI=="img": # process frames with animals to and export an xml file for ml training

    rewind = 'false' # just accept it
    img_num, img_num_start, img_num_last = pl.starting_point(flightFolderName, directoryName) # pick up where you left off
    # print(img_num)
    print("starting from: " +str(img_num_start))
    print("end at: " + str(img_num_last))
    (out_of_bounds, threshlim, box_limit_max) = pl.setup_defaults() # set your defaults

    while(img_num <= img_num_last):

        img_num_str = str(img_num)
        imgname = "Ninox"+img_num_str+".jpg"
        full_fname = IMGpathName+imgname
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
                    # print("img_num")

            else: # go forward, find next
                img_num = int(img_num+framerate)
                # rewind = 'false'

        else: #if there's a frame to load

            rewind = 'false' # reset this
            vidheight, vidwidth, channels = frame.shape # get frame props
            pl.set_windows(box_limit_max, frame, screens) # set up the screen
            cv2.imshow("Detection",frame) # show raw image
            closed_frame = pl.preprocessing(threshlim, frame) # prep frame
            pl.printTextFrameLoad(full_fname, threshlim)
            pl.mlROI(frame) # demo for harry
            k = cv2.waitKey(0) & 0xff # wait for input: ent, space, del, or esc

            #######################################
            ## Here are your options now:
            # ent - process:
                # c - calibrate
                # r - reset to default calibrations
                # m - manually select targets
                # d - delete targets
                # ent - export
                # esc - cancel
            # space - continue
            # delete - rewind
            ########################################

            if k == 13: # return/enter

                closed_frame = pl.preprocessing(threshlim, frame) # prep frame
                num_contours, contours_list = pl.process_contours(closed_frame, box_limit_max, out_of_bounds, frame)
                k = 0

                while k is not 13:

                    pl.printTextOnInitialProcessing(contours_list)
                    k = cv2.waitKey(0) & 0xff # after initial process can retry or export.

                    if k == 114: # recalibrate
                        print("reset default limits")
                        out_of_bounds, threshlim, box_limit_max = pl.setup_defaults()
                        break

                    elif k == 99: #c - calibrate
                        threshlim, box_limit_max, num_contours, contours_list, frame = pl.on_keystrokes_calibrate(k, threshlim, box_limit_max, out_of_bounds, full_fname, screens)

                    elif k == 100: #d - manually select animals to delete
                        contours_list, num_contours, frame = pl.delete_targets(contours_list, num_contours, frame)
                        print(contours_list)

                    elif k == 115: #s - select images  manual
                        contours_list, num_contours = pl.manual_select(frame, k, num_contours, contours_list)

                    elif k == 121:
                        cropframe=frame[0:576, 0:748]
                        savename="NinoxDemo"
                        framenum = img_num_str
                        cropname = '/home/kookaburra/Desktop/'+str(savename)+img_num_str+".jpg"
                        cv2.imwrite(cropname, cropframe)
                        k = cv2.waitKey(50) & 0xff
                        break

                    elif k == 13: # enter - export
                        break

                    elif k == 27: # esc - exit
                        break

                if k == 13: #if loop was broken with enter then export
                    if num_contours is not 0:
                        pl.export2xml(img_num,flightFolderName,vidheight, vidwidth, contours_list, num_contours)
                        img_num = img_num+framerate # next frame

            elif k == 32: #space bar
                img_num = img_num+framerate
                print("skip to next frame")
                rewind == 'false'

            elif k == 8: #backspace
                img_num = img_num-framerate
                print("rewind")
                rewind = 'true'

            elif k == 27: #esc - exit
                break
if VI == "img":
    pl.whereAmIup2(img_num_str, flightFolderName, directoryName) # save current frame number so you can start from here next time
cv2.destroyAllWindows() ## all done. close everything.
