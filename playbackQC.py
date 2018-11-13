#!/usr/local/bin/python3

# Greenroom playback by Jim Keane
# james@greenroomrobotics.com
# last update Fri, 20 Jul 18
# ver 1.100 incorporated quality control function to recheck images and change
# xml files before ml training

# playback exists to speed up the process for getting training images out of
# videos for greenrooms ML
# there are two parts to playback:
# 1) grabbing frames from a video,
# 2) going back through those frames and targeting images for training

# stage 1: video playback
# stage 1 runs a Video.. watch it then pound your keyboard when
# you seen an animal
# this will save the frame with an animal into as a jpeg
# you can fast forward through boring spots, or go slow mo if you think
# somethings coming up.
# there's no rewind function yet
# follow the intructions

# stage 2: frame processing
# stage 2 goes through the frames that you saved from stage 1
# you can either skip frames, process them, or rewind
# if you process a frame you have options to add or delete contacts if the
# cv has done a shit job
# follow the bouncing ball
# sometimes you might have to double tap keys in to get a response so it's
 # not really user friendly yet

# the end product is xml files with the onscreen location of animals
# the xml files are in a format that should work for training tensor flow

# playback_ini contains all the functions for playback to work

import cv2
import time
import numpy as np
import os
from os import path
import playbackQC_ini as pl

# Directory Structure:
    # (save playback & ini files here)
    # directoryName = parent folder
    # IMGpathName = folder within parent that+
                    # images get stored to then called from

# currently running off a USB - JimsDisk, but change it here.
directoryName = '/home/ubuntu/openCV/GreenRoom/Ninox/'
flightFolderName = 'flight1images' # images go into here
IMGpathName = directoryName+flightFolderName+'/'
# change this if you want to save frames from vid processing to somewhere else
saveDirectoryName = IMGpathName

## config
screens = 1 # 1 or 2 if you have a monitor attached
# ninox frame rate 24fps
# cut down on the number of frames you're looking at in the vid
framerate = 8
#vid or img, QC or output:
# use vid for saving frames for cv processing,
# then img (for saving outputs for ML)

VI = "img"
save_vid_screens = False

if VI == "vid": # play video to find frames with likely animals
    starting_point = 0.0
    vidname = 'flight3friday2june.avi'
    full_fname = directoryName+vidname
    box_limit_max = 0
    david,waittime,totalframes = pl.setup_vid(vidname, full_fname,
                                              box_limit_max, starting_point)
    pl.printVideoPlaybackInstructions() #follow the bouncing ball

    ## start video playback
    while(1):

        framenum = david.get(1) # get next frame from video
        print(framenum)
        trackerok, frame = david.read()

        if not trackerok: #check frame loaded
            break

        # only show 1 in x frames (helps to avoid a seizure)
        if framenum % framerate == 0:
            print("frame " + str(framenum))
            cv2.imshow("Detection", frame)
            # closed_frame = pl.preprocessing(threshlim, frame)
            # convert img to greyscale then binary for processing
            # threshlim = cv2.getTrackbarPos('thrs1', 'Threshold')
            # if the slidebar is set up
            k = cv2.waitKey(waittime) & 0xff #micropause

            if k == 27: #hit esc, end loop
                break

            waittime = pl.keystrokes_during_video(k,vidname,frame,
                                                framenum, waittime,
                                                saveDirectoryName,
                                                save_vid_screens) # any UI?
        # break if there are no more frames to analyse
        if framenum > totalframes:
            print("end of vid. break.")
            break

    david.release() # stop reading video

# process frames with animals to and export an xml file for ml training
elif VI=="img":

    rewind = 'false' # just accept it
     # pick up where you left off
    img_num,img_num_start,img_num_last = pl.starting_point(flightFolderName,
                                                            directoryName)
    print("starting from: " +str(img_num_start))
    print("end at: " + str(img_num_last))
     # set your defaults
    (out_of_bounds, threshlim, box_limit_max) = pl.setup_defaults()

    while(img_num <= img_num_last):

        img_num_str = str(img_num)
        imgname = "Ninox"+img_num_str+".jpg"
        full_fname = IMGpathName+imgname
        frame = cv2.imread(full_fname) # read the frame from the folder

        if frame is None: # if no frame to load
             # rewind
            if rewind == 'true':
                # if you're at the start don't rewind further
                if img_num == img_num_start:
                    rewind = 'false'

                elif img_num < img_num_start:
                    img_num = img_num_start
                    rewind = 'false'

                else:
                    img_num = int(img_num-framerate)

            else: # go forward, find next
                img_num = int(img_num+framerate)

         #if there's a frame to load
        else:

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
                num_contours,contours_list = pl.process_contours(closed_frame,
                                                                box_limit_max,
                                                                out_of_bounds,
                                                                frame)
                k = 0

                while k is not 13:

                    pl.printTextOnInitialProcessing(contours_list)
                    # after initial process wait UI to retry or export.
                    k = cv2.waitKey(0) & 0xff

                    if k == 114: # recalibrate
                        print("reset default limits")
                        out_of_bounds,
                        threshlim,
                        box_limit_max = pl.setup_defaults()
                        break

                    elif k == 99: #c - calibrate
                        threshlim, box_limit_max, num_contours,
                        contours_list, frame = pl.on_keystrokes_calibrate(
                                                    k, threshlim, box_limit_max,
                                                    out_of_bounds, full_fname,
                                                    screens)

                    elif k == 100: #d - manually select animals to delete
                        contours_list,num_contours,frame = pl.delete_targets(
                                                            contours_list,
                                                            num_contours,
                                                            frame, VI)
                        print(contours_list)

                    elif k == 115: #s - select images  manual
                        contours_list,num_contours = pl.manual_select(
                                                    frame, k, num_contours,
                                                    contours_list, VI)

                    elif k == 121:
                        cropframe=frame[0:576, 0:748]
                        savename="NinoxDemo"
                        framenum = img_num_str
                        cropname = '/home/kookaburra/Desktop/'+\
                                    str(savename)+img_num_str+".jpg"

                        cv2.imwrite(cropname, cropframe)
                        k = cv2.waitKey(50) & 0xff
                        break

                    elif k == 13: # enter - export
                        break

                    elif k == 27: # esc - exit
                        break

                if k == 13: #if loop was broken with enter then export
                    if num_contours is not 0:
                        pl.export2xml(img_num,flightFolderName,vidheight,
                                        vidwidth, contours_list, num_contours,
                                        VI)
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

elif VI == "QC":

    oldPath = IMGpathName
    newPath = IMGpathName + 'newXML/'
    qc_text_file = newPath + "qc.txt"
    qc = open(qc_text_file, 'w')
    filenames = []
    imagenames = []
    files_already_checked = []

    for file in os.listdir(oldPath):
        if file.endswith(".xml"):
            filenames.append(file)
            file2 = (file[:-4]+'.jpg')
            imagenames.append(file2)
    print(filenames)
    im = 0
    total_im = len(filenames)
    # this is how many processed files there are to check
    print("number files: " + str(total_im))
    print("xml files: "+str(len(filenames))+", and images: "\
            +str(len(imagenames)))

    for file in os.listdir(newPath):
        if file.endswith(".xml"):
            files_already_checked.append(file)

    print(files_already_checked)
    qc.write("total frames: " + str(total_im)+'\n\n')

    while(im < total_im):

        print(" ")
        print(im)
        full_fname = IMGpathName+imagenames[im]
        xml_fname = IMGpathName+filenames[im]
        print(full_fname)

        if filenames[im] in files_already_checked:
            print("Already Checked")
            im += 1
            continue

        frame = cv2.imread(full_fname) # read the frame from the folder
        vidheight, vidwidth, channels = frame.shape
        if frame is None: # if no frame to load
            print("no frame to load")

        else: #if there's a frame to load
            cv2.namedWindow('qc', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('qc', int(frame.shape[1]*2), int(frame.shape[0]*2))
            # # cv2.moveWindow('qc', 3800, 200)
            xmlPath = oldPath+filenames[im]
            lines, linesOfInterest = pl.loadXML(xmlPath)

            #####################################
            ########### R & D ###################
            #####################################

            vw, vh, lines = pl.checkFrameSizes(lines, linesOfInterest, frame)
            contours_list = pl.getbbox(lines, linesOfInterest, frame)
            contours_list = pl.checkForZeroSizes(contours_list,
                                                lines, linesOfInterest)
            pl.qcTextDisplay()

            k = cv2.waitKey(0) & 0xff
            if k == 13: # 226 - shift key

                contours_list = pl.check_bboxes(contours_list, frame, vw, vh)
                pl.qcOptionsTextDisplay()
                k = 0

                while k is not 13:

                    k = cv2.waitKey(0) & 0xff

                    if k == 100: #d - manually select animals to delete
                        num_contours = len(contours_list[0])
                        print(num_contours)
                        contours_list,num_contours,frame = pl.delete_targets(
                                                            contours_list,
                                                            num_contours,
                                                            frame, VI)
                        print(contours_list)

                    elif k == 115: #s - select images  manual
                        num_contours=len(contours_list[0])
                        contours_list,num_contours = pl.manual_select(frame, k,
                                                        num_contours,
                                                        contours_list, VI)

                    elif k == 226: # shift key
                        break

                    elif k == 13: # enter - export
                        break

                    elif k == 27: # esc - exit
                        break

                if k == 13: #if loop was broken with enter then export
                    num_contours = len(contours_list[0])
                    thisfile = filenames[im]
                    img_num = int(thisfile[thisfile.find('Ninox')+5:\
                                            thisfile.rfind('.xml')])
                    print(img_num)
                    flightFolderName = flightFolderName

                    if num_contours is not 0:
                        pl.export2xml(img_num,flightFolderName,vidheight,
                                        vidwidth, contours_list, num_contours,
                                        VI)
                        img_num = img_num+framerate # next frame
                        qc.write(str(filenames[im]) + '\n')
                    im += 1

            # if k == 13: # run QC
            elif k == 100:  #d - delete entire xml
                print("ARE YOU SURE YOU WANT TO DELETE \
                        ENTIRE XML FOR THIS IMAGE?")
                print(full_fname)
                print("y for yes, any other key to pass")
                kk = cv2.waitKey(0) & 0xff

                if kk == ord('y'):
                    pl.deleteXML(xml_fname)
                    qc.write("deleted " + xml_fname + "\n")
                    im += 1
                else:
                    pass

            elif k == 32: #space bar - skip frame
                im += 1

            elif k == 8: #backspace - back frame
                im -= 1

            elif k == 27: #esc - quit
                break

            qc.write(str(filenames[im])+'/n')
            cv2.destroyAllWindows()

elif VI == 'output':

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('GreenroomNinoxCVdemo.avi',fourcc, 20.0, (732,560))

    oldPath = IMGpathName
    newPath = IMGpathName + 'newXML/'
    qc_text_file = newPath + "qc.txt"
    qc = open(qc_text_file, 'w')
    filenames = []
    imagenames = []
    files_already_checked = []

    for file in os.listdir(oldPath):
        if file.endswith(".xml"):
            filenames.append(file)
            file2 = (file[:-4]+'.jpg')
            imagenames.append(file2)
    print(filenames)
    im = 0
    total_im = len(filenames)
    # total_im = 30
    # this is how many processed files there are to check
    print("number files: " + str(total_im))
    print("xml files: " +str(len(filenames)) \
            + ", and images: " + str(len(imagenames)))

    while im < total_im:

        full_fname = IMGpathName+imagenames[im]
        frame = cv2.imread(full_fname)
        vidheight, vidwidth, channels = frame.shape

        xmlPath = oldPath+filenames[im]
        lines, linesOfInterest = pl.loadXML(xmlPath)
        contours_list = pl.getbbox(lines, linesOfInterest, frame)

        out.write(frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        im += 1
        print(im)


############################################################
############### close it all down ##########################
############################################################

if VI == "img":
    # save current frame number so you can start from here next time
    pl.whereAmIup2(img_num_str, flightFolderName, directoryName)
elif VI == "QC":
    qc.write(str(im)+'\n')
    qc.close()

    # cap.release()
elif VI == "output":
    out.release()

cv2.destroyAllWindows() ## all done. close everything.
