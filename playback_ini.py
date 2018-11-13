# playback_ini

# functions called in greenrooms playback.py
# grouped as functions for the video, then
# functions for the img processing
# preprocessing is the only function called in both

import os
import cv2
import numpy as np

def preprocessing(threshlim, frame): # convert frame to greyscale then binary

    imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  # convert coloured image to grayscale
    ret,thresh = cv2.threshold(imgray,threshlim,255, cv2.THRESH_BINARY)     # threshold the grayscale
    kernel = np.ones((4,4),np.uint8)
    closed_frame = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # morphological operations - opening to clear background fuzz
    cv2.imshow("Threshold", closed_frame)

    return closed_frame

##########################################
###### Functions for video playback ######
##########################################

def setup_vid(vidname, full_fname, box_limit_max, starting_point): # initial video set up required
    framenum = starting_point
    david = cv2.VideoCapture(full_fname) # load video
    if david.isOpened(): #check vid, get prpoerties
        pos_msec = david.get(2) # position of file in ms or timestamp
        vidwidth = david.get(3) #frame width
        vidheight = david.get(4) #frame height
        props = david.get(1) #frame
        print(props)
        totalframes = david.get(7) #total number of frames
        print(vidwidth) # video width in pixels
        print(vidheight) # hieght pixels
        print(totalframes)

    else: # something went wrong
        print("Video not opened properly....")
        print(" ")

    # set_windows(box_limit_max, frame, screens) # set up the screen
    newvidwidth = int(vidwidth*2)
    newvidheight = int(vidheight*2)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', newvidwidth, newvidheight)

    david.set(1,1128.0) #set starting frame of video for first imgrab

    waittime = 50 #wait time between frames (for video playback)

    return david, waittime, totalframes

def keystrokes_during_video(k, vidname, frame, framenum, waittime, saveDirectoryName, save_vid_screens): # options for running the video

    if k==ord('y'): #save frame as jpg
        if save_vid_screens == True:
            output_frame(vidname, frame, framenum, saveDirectoryName)

    elif k==ord('f'): # fast fwd
        waittime = 5
        print("f wait")

    elif k==ord('s'): #slow motion
        waittime = 200
        print("s wait")

    elif k==ord('d'): #play normal speed
        waittime = 50
        print("d wait")

    return waittime

def output_frame(vidname, frame, framenum, saveDirectoryName): # save frames from video playback as jpeg

    cropframe=frame[0:576, 0:748]
    savename="Ninox"
    framenum = int(framenum)
    cropname = saveDirectoryName+str(savename)+str(framenum)+".jpg"
    cv2.imwrite(cropname, cropframe)
    k = cv2.waitKey(50) & 0xff

    return

def printVideoPlaybackInstructions():

    print("")
    print("VIDEO PLAYBACK OPTIONS:")
    print("y - save frame as jpeg (either press this once or hold it down)")
    print("f - fast forward")
    print("s - slow motion")
    print("d - normal speed")
    print("i - show instructions again")
    print("esc - exit")
    print("hit any key to begin")
    cv2.waitKey(0) & 0xff

##########################################
###### Functions for img processing ######
##########################################

def set_windows(box_limit_max, frame, screens): # sets up the windows for jims computer

    vidheight, vidwidth, channels = frame.shape # get frame size
    if screens == 2:
        newvidwidth = int(vidwidth*1.5)
        newvidheight = int(vidheight*1)

    else:
        newvidwidth = int(vidwidth*2)
        newvidheight = int(vidheight*1)

    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', newvidwidth, newvidheight)
    cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Threshold', newvidwidth, newvidheight)
    if screens == 2:
        cv2.moveWindow('Detection', 3850, 10) # 1 screen 1500 2 screen 3800
        cv2.moveWindow('Threshold', 2900, 10) # 1 screen 200 2 screen 3000

    else:
        cv2.moveWindow('Detection', 1500, 10) # 1 screen 1500 2 screen 3800
        cv2.moveWindow('Threshold', 200, 10) # 1 screen 200 2 screen 3000

    demo_x = 20
    demo_y = 20
    demobbx = cv2.rectangle(frame, (demo_x, demo_y), (demo_x+box_limit_max, demo_y+box_limit_max), (0, 255,0),2)

    # cv2.createTrackbar('thrs1', 'Threshold', 150, 255, thrs_intensity_callback) #trackbar for intensity
    # cv2.createTrackbar('thresh_sizebar', 'Threshold', 10, 100, thrs_size_callback) #trackbar for animal size

    return

def process_contours(closed_frame, box_limit_max, out_of_bounds, frame): # just run it

    vidheight, vidwidth, channels = frame.shape
    horizon = out_of_bounds[0]
    top_left_text_limit = out_of_bounds[1]
    bottom_text_limit = out_of_bounds[2]
    crosshair_limit = out_of_bounds[3]
    mid_left_text_limit = out_of_bounds[4]
    xl = []
    yl = []
    wl = []
    hl = []

    thresh, contours, hierarchy = cv2.findContours(closed_frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # find contoured objects
    num_contours=0
    box_limit_min = int(box_limit_max*0.2)

    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)

        if h>box_limit_max or w>box_limit_max: # discard areas that are too large
            continue

        if h<box_limit_min and w<box_limit_min: # discard areas that are too small
            continue

        if y < horizon or y == (vidheight-h): #discard areas above the horizon
            continue

        if x == 0 or x+w == vidwidth: #discard anything touching the frameedge
            continue

        if y == 0 or y+h == vidheight: #discard anything touching the frameedge
            continue

        if x < top_left_text_limit[2] and y < top_left_text_limit[3]: #ignore the onscreen text
            continue

        if y+h > vidheight-bottom_text_limit:
            continue

        if y < int(crosshair_limit[1]+crosshair_limit[3]) and y > crosshair_limit[1] and x < int(crosshair_limit[0]+crosshair_limit[2]) and x > crosshair_limit[0]:
            continue

        if x < mid_left_text_limit[2] and y > mid_left_text_limit[1] and y < (mid_left_text_limit[1]+mid_left_text_limit[3]):
            continue

        if w < 8:
                w = int(w*4)
                h = int(h*3)
                x = int(x-w*1/2)
                y = int(y-h*1/3)

        else:
            w = int(w*2)
            h = int(h*2)
            x = int(x-w*1/4)
            y = int(y-h*1/4)

        boundimg = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0,0), 1)
        num_contours=num_contours+1
        xl.append(x)
        yl.append(y)
        wl.append(w)
        hl.append(h)

    cv2.imshow("Detection", frame)
    print(num_contours)
    contours_list = [xl, yl, wl, hl]

    return num_contours, contours_list

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

    fname = flightFolderName+"_whichFrameAmIUpTo.txt"
    text_file_name = directoryName+fname
    f = open(text_file_name, 'w')
    f.write(img_num_str)
    f.close()

    return

def starting_point(flightFolderName, directoryName): # pick up where you left off last time

    fname = flightFolderName+"_whichFrameAmIUpTo.txt"
    text_file_name = directoryName+fname
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

def export2xml(img_num, flightFolderName, vidheight,vidwidth, contours_list, num_contours): #exports all contacts for frame to xml

    img_num_str = str(img_num)
    vidwidth = str(vidwidth)
    vidheight = str(vidheight)
    path = '/media/kookaburra/JimsDisk/Ninox/'+flightFolderName+"/"
    text_file_name = path+"Ninox"+img_num_str+".xml"
    img_name="Ninox"+img_num_str+".jpg"
    print("exporting:" + text_file_name)
    nl = "\n" #new line
    tb = "\t" #tab
    dt = "\t\t" #double tab
    tt = "\t\t\t" #triple tab
    f =open(text_file_name,'w')
    f.write("<annotation>"+nl+tb+"<folder>"+flightFolderName+"</folder>"+nl)
    f.write(tb+"<filename>"+img_name+"</filename>"+nl)
    f.write(tb+"<path>"+path+img_name+"</path>"+nl)
    f.write(tb+"<source>"+nl+tb+tb+"<database>Unknown</database>"+nl+tb+"</source>"+nl)
    f.write(tb+"<size>"+nl+dt+"<width>"+vidwidth+"</width>"+nl)
    f.write(dt+"<height>"+vidheight+"</height>"+nl)
    f.write(dt+"<depth>3</depth>"+nl+tb+"</size>"+nl)
    f.write(tb+"<segmented>0</segmented>"+nl)

    xl = contours_list[0] # break contour list into individual components
    yl = contours_list[1]
    wl = contours_list[2]
    hl = contours_list[3]

    if xl:
        for i in range(num_contours):
            xmin = str(xl[i])
            ymin = str(yl[i])
            xmax = str(int(xl[i]+wl[i]))
            ymax = str(int(yl[i]+hl[i]))
            f.write(tb+"<object>"+nl)
            f.write(dt+"<name>animal</name>"+nl+dt+"<pose>Unspecified</pose>"+nl)
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

def mlROI(frame): # focuses on region of interst for ML to read to generate a log file
    # each box is a region of inerest
    #e.g set up to read heading, but add points for additional regions
    boxx = [22] # box start point (pixels from left of screen)
    boxy = [40] # box start point (pixels from Top! of screen)
    boxw = [40] # box width
    boxh = [20] # box height

    for i in range(len(boxx)):
        trackbox = (boxx[i],boxy[i],boxw[i], boxh[i])
        p1 = (int(trackbox[0]), int(trackbox[1]))
        p2 = (int(trackbox[0] + trackbox[2]), int(trackbox[1] + trackbox[3]))
        MLframe=frame[p1[1]:p2[1], p1[0]:p2[0]]
        popup = cv2.imshow("readMeMl", MLframe)

    return

def printTextFrameLoad(full_fname, threshlim): # displays text with instructions

    print("")
    print("file name: "+full_fname)
    print("thresh lim: " + str(threshlim))
    print("OPTIONS:")
    print("enter - process this frame")
    print("space - skip to next frame")
    print("backspace - back one frame")
    print("esc - exit")

    return

def printTextOnInitialProcessing(contours_list):  # displays text with instructions

    print("")
    print("PROCESSED")
    print("Picked up the following: ")
    print(contours_list)
    print("OPTIONS: ")
    print("ent - export to xml and open next frame")
    print("c - calibrate")
    print("r - reset thresholds to default")
    print('s - select targets manually')
    print("d - delete targets")
    print('esc - cancel')

    return

def thrs_intensity_callback(thrs1): #when i was running a slide bar

    threshlim = cv2.getTrackbarPos('thrs1', 'Threshold')
    closed_frame = preprocessing(threshlim, frame)
    cv2.imshow("Threshold", closed_frame)

    return threshlim

def thrs_size_callback(thresh_sizebar): #when i was running a slide bar

    box_limit_max = cv2.getTrackbarPos('thresh_sizebar', 'Threshold')
    process_contours(closed_frame, box_limit_max, out_of_bounds, frame)

    return box_limit_max
