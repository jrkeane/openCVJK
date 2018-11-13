# playback_ini

# functions called in greenrooms playback.py
# grouped as functions for the video, then
# functions for the img processing
# preprocessing is the only function called in both

import os
import cv2
import numpy as np

"""Preprocess images before analysis"""
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

# Prepare video to be read and set up windows on screen
def setup_vid(vidname, full_fname, box_limit_max, starting_point):
    # initial video set up required
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
        
    # set up the screen
    # set_windows(box_limit_max, frame, screens)
    newvidwidth = int(vidwidth*2)
    newvidheight = int(vidheight*2)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', newvidwidth, newvidheight)

    david.set(1,1128.0) #set starting frame of video for first imgrab

    waittime = 50 #wait time between frames (for video playback)

    return david, waittime, totalframes

# defines actions for keys when struck during video playback
def keystrokes_during_video(k, vidname, frame, framenum,
                            waittime, saveDirectoryName,
                            save_vid_screens):

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

# save frames from video playback as jpeg
def output_frame(vidname, frame, framenum, saveDirectoryName):

    cropframe=frame[0:576, 0:748]
    savename="Ninox"
    framenum = int(framenum)
    cropname = saveDirectoryName+str(savename)+str(framenum)+".jpg"
    cv2.imwrite(cropname, cropframe)
    k = cv2.waitKey(50) & 0xff

    return

# shows instructional text in command line
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

# sets up the windows for jims computer for img processing
def set_windows(box_limit_max, frame, screens):

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
    demobbx = cv2.rectangle(frame, (demo_x, demo_y),
                            (demo_x+box_limit_max, demo_y+box_limit_max),
                            (0, 255,0),2)

    # Trackbar for intensity
    # cv2.createTrackbar('thrs1', 'Threshold', 150, 255,
                        # thrs_intensity_callback)
    #trackbar for animal size
    # cv2.createTrackbar('thresh_sizebar', 'Threshold', 10, 100,
                        # thrs_size_callback)

    return

# Process contours found in image to look for animals
def process_contours(closed_frame, box_limit_max, out_of_bounds, frame):

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

    # find contoured objects
    thresh, contours, hierarchy = cv2.findContours(closed_frame,
                                                    cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
    num_contours=0
    box_limit_min = int(box_limit_max*0.2)

    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)

        # discard areas that are too large
        if h>box_limit_max or w>box_limit_max:
            continue

        # discard areas that are too small
        if h<box_limit_min and w<box_limit_min:
            continue

        #discard areas above the horizon
        if y < horizon or y == (vidheight-h):
            continue

        #discard anything touching the frame sides
        if x == 0 or x+w == vidwidth:
            continue

         #discard anything touching the frame top or bottom
        if y == 0 or y+h == vidheight:
            continue

        #ignore the onscreen text
        if x < top_left_text_limit[2] and y < top_left_text_limit[3]:
            continue

        # ignore onscreen text
        if y+h > vidheight-bottom_text_limit:
            continue

        # ignore the ninox crosshair
        if (y < int(crosshair_limit[1]+crosshair_limit[3])
            and y > crosshair_limit[1]
            and x < int(crosshair_limit[0]+crosshair_limit[2])
            and x > crosshair_limit[0]):
            continue

        # ignore onscreen text
        if (x < mid_left_text_limit[2]
            and y > mid_left_text_limit[1]
            and y < (mid_left_text_limit[1]+mid_left_text_limit[3])):
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

# sets limits of onscreen text, & intensity and size thresholds
def setup_defaults():

    horizon = 20 #pixel position of horizon (taken from top)
    top_left_text_limit = [0,0,100,340]
    bottom_text_limit = 20
    crosshair_limit = [350, 260, 40, 40]
    mid_left_text_limit = [0, 250, 65, 100]
    out_of_bounds = [horizon, top_left_text_limit,
                    bottom_text_limit, crosshair_limit,
                    mid_left_text_limit]
    # set default thresh value to recall if the adaptive one changes
    threshlim_default = 155
    box_limit_default = 10
    threshlim = threshlim_default
    box_limit_max = box_limit_default

    return out_of_bounds, threshlim, box_limit_max

# write frame number to text so you can pick up from here next time
def whereAmIup2(img_num_str, flightFolderName, directoryName):

    fname = flightFolderName+"_whichFrameAmIUpTo.txt"
    text_file_name = directoryName+fname
    f = open(text_file_name, 'w')
    f.write(img_num_str)
    f.close()

    return

# pick up where you left off last time
def starting_point(flightFolderName, directoryName):

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

    #if a file exists you'll pick up from where you left off last time
    if FileExists == True:
        f = open(text_file_name, 'r')
        string = f.read()
        img_num = int(string)
        f.close()

    print("starting at "+str(img_num_start))

    return img_num, img_num_start, img_num_last

#exports all contacts for frame to xml
def export2xml(img_num, flightFolderName, vidheight,
                vidwidth, contours_list, num_contours, VI):

    img_num_str = str(img_num)
    vidwidth = str(vidwidth)
    vidheight = str(vidheight)
    if VI == 'QC':
        flightFolderName = flightFolderName + "/newXML"

    path = '/media/kookaburra/JimsDisk/Ninox/'+flightFolderName+"/"
    text_file_name = path+"Ninox"+img_num_str+".xml"
    img_name="Ninox"+img_num_str+".jpg"
    print("exporting:" + text_file_name)
    nl = "\n" #new line
    tb = "\t" #tab
    dt = "\t\t" #double tab
    tt = "\t\t\t" #triple tab
    f = open(text_file_name,'w')
    f.write("<annotation>"+nl+tb+"<folder>"+flightFolderName+"</folder>"+nl)
    f.write(tb+"<filename>"+img_name+"</filename>"+nl)
    f.write(tb+"<path>"+path+img_name+"</path>"+nl)
    f.write(tb+"<source>"+nl+tb+tb+"<database>Unknown</database>"\
            +nl+tb+"</source>"+nl)
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
            f.write(dt+"<name>animal</name>"+nl+dt\
                    +"<pose>Unspecified</pose>"+nl)
            f.write(dt+"<truncated>0</truncated>"\
                    +nl+dt+"<difficult>0</difficult>"+nl)
            f.write(dt+"<bndbox>"+nl)
            f.write(tt+"<xmin>"+xmin+"</xmin>"+nl)
            f.write(tt+"<ymin>"+ymin+"</ymin>"+nl)
            f.write(tt+"<xmax>"+xmax+"</xmax>"+nl)
            f.write(tt+"<ymax>"+ymax+"</ymax>"+nl)
            f.write(dt+"</bndbox>"+nl+tb+"</object>")

    f.write("</annotation>")
    f.close()

    return

# for calibrating thresholds
def on_keystrokes_calibrate(k, threshlim, box_limit_max,
                            out_of_bounds, full_fname, screens):

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
        num_contours, contours_list = process_contours(closed_frame,
                                                        box_limit_max,
                                                        out_of_bounds, frame)

    return threshlim, box_limit_max, num_contours, contours_list, frame

# for the user to manually select targets
def manual_select(frame, k, num_contours, contours_list, VI):

    xl = contours_list[0]
    yl = contours_list[1]
    wl = contours_list[2]
    hl = contours_list[3]

    while k is not 13:

        print("Manually select targets")
        if VI == "QC":
            cali_box = cv2.selectROI("qc", frame, True, False)
        else:
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

# part of the manual_select function
def click_targets(event, x, y, flags, param):

    global clickx, clicky
    if event == cv2.EVENT_LBUTTONDOWN:
        clickx = x
        clicky = y
        print(str(x)+", "+str(y))

    return
 # if user wants to delete targets
def delete_targets(contours_list, num_contours, frame, VI):

    print(" ")
    print("CLICK target to delete then hit enter")
    xl = contours_list[0]
    yl = contours_list[1]
    wl = contours_list[2]
    hl = contours_list[3]

    while True:

        while True:

            kk = cv2.waitKey(1)
            if VI == 'QC':
                cv2.setMouseCallback("qc", click_targets)
            else:
                cv2.setMouseCallback("Detection", click_targets)
            if kk == 13:
                break

        delX = clickx
        delY = clicky
        deletePt = (delX, delY)
        contacts_list = range(num_contours)
        # print(contacts_list)
        deleted = 0
        for i in reversed(contacts_list):
            print("IIIIIII      " + str(i))
            x = int(xl[i]-5)
            y = int(yl[i]-5)
            w = int(wl[i]+10)
            h = int(hl[i]+10)
            if delX > x and delX < (x+w) and delY > y and delY < (y+h):
                if deleted > 0:
                    print("and this one ("+str(x)+", "+str(y)+")")

                else:
                    print("deleted this one ("+str(x)+", "+str(y)+")")

                boundimg = cv2.rectangle(frame, (xl[i], yl[i]),
                                        (xl[i]+wl[i], yl[i]+hl[i]),
                                        (255, 255,255), 2)
                if VI == "QC":
                    cv2.imshow('qc', frame)
                else:
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

# focuses on region of interst for ML to read to generate a log file
def mlROI(frame):
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

# displays text with instructions
def printTextFrameLoad(full_fname, threshlim):

    print("")
    print("file name: "+full_fname)
    print("thresh lim: " + str(threshlim))
    print("OPTIONS:")
    print("enter - process this frame")
    print("space - skip to next frame")
    print("backspace - back one frame")
    print("esc - exit")

    return

# displays text with instructions
def printTextOnInitialProcessing(contours_list):

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

#when i was running a slide bar
def thrs_intensity_callback(thrs1):

    threshlim = cv2.getTrackbarPos('thrs1', 'Threshold')
    closed_frame = preprocessing(threshlim, frame)
    cv2.imshow("Threshold", closed_frame)

    return threshlim

 #when i was running a slide bar
def thrs_size_callback(thresh_sizebar):

    box_limit_max = cv2.getTrackbarPos('thresh_sizebar', 'Threshold')
    process_contours(closed_frame, box_limit_max, out_of_bounds, frame)

    return box_limit_max


###############################################################
##################### qc ######################################
###############################################################

def loadXML(xmlPath):

    f = open(xmlPath, 'r')
    lines = f.readlines()
    f.close()
    # for i in range(len(lines)):
    #     print(lines[i])

    # print(len(lines))
    linesOfBndbox = []

    for line in range(len(lines)):
        if "<bndbox>" in lines[line]:
            linesOfBndbox.append(line)

        if "<width>" in lines[line]:
            lineOfVidwidth = int(line)
            # print(lineofvidwidth)
        if "<height>" in lines[line]:
            lineOfVidheight = int(line)

    print("line bndbox " + str(linesOfBndbox))

    linesOfInterest = [lineOfVidwidth, lineOfVidheight, linesOfBndbox]

    return lines, linesOfInterest

def checkFrameSizes(lines, linesOfInterest, frame):

    vidheight, vidwidth, channels = frame.shape
    lineOfVidwidth = linesOfInterest[0]
    lineOfVidheight = linesOfInterest[1]

    # item ='width'
    # child = '<'+item+'>'
    # child_len = 6
    vw = lines[lineOfVidwidth]
    vw = int(vw[vw.find('<width>')+7:vw.rfind('</width>')])
    vh = lines[lineOfVidheight]
    vh = int(vh[vh.find('<height>')+8:vh.rfind('</height>')])

    # print(vw)
    # print(vh)

    if vw == vidwidth and vh == vidheight:
        pass

    else:
        print("xml doesn't match frame")
        print('')
        print('xml width:'+str(vw))
        print('actual frame width: ' + str(vw))
        print('xml height' + str(vh))
        print('actual frame height: ' + str(vh))

        qc.write(current_frame+'\n')
        qc.write("xml doesn't match frame"+'\n')

    return vw,vh, lines


def getbbox(lines, linesOfInterest, frame):

    linesOfBndbox = linesOfInterest[2]
    x=[]
    y=[]
    w=[]
    h=[]

    for line in linesOfBndbox:

        xmin = lines[line+1]
        xmin = int(xmin[xmin.find('<xmin>')+6:xmin.rfind('</xmin>')])
        ymin = lines[line+2]
        ymin = int(ymin[ymin.find('<ymin>')+6:ymin.rfind('</ymin>')])
        xmax = lines[line+3]
        xmax = int(xmax[xmax.find('<xmax>')+6:xmax.rfind('</xmax>')])
        ymax = lines[line+4]
        ymax = int(ymax[ymax.find('<ymax>')+6:ymax.rfind('</ymax>')])
        width = xmax-xmin
        height = ymax - ymin
        x.append(xmin)
        y.append(ymin)
        w.append(width)
        h.append(height)

    contours_list = [x, y, w, h]

    for i in range(len(x)):
        boundimg = cv2.rectangle(frame, (x[i], y[i]),
                                (x[i]+w[i], y[i]+h[i]), (0, 200,0), 1)

        print('size: ' + str(w[i]) + ' x '+ str(h[i]))

    print(contours_list)
    print('number contacts = ' + str(len(x)))
    cv2.imshow('qc', frame)

    return contours_list

def checkForZeroSizes(contours_list, lines, linesOfInterest):

    x = contours_list[0]
    y = contours_list[1]
    w = contours_list[2]
    h = contours_list[3]
    num_contours = len(x)

    for i in reversed(range(num_contours)):
        if h[i] == 0 or w[i] == 0: # if non-object delete it
            del x[i]
            del y[i]
            del w[i]
            del h[i]
            print("deleted non-object")

    contours_list = [x, y, w, h]

    return contours_list


def check_bboxes(contours_list, frame,vw, vh):

    # print(contours_list)
    x = contours_list[0]
    y = contours_list[1]
    w = contours_list[2]
    h = contours_list[3]
    num_contours = len(x)
    # print(i_contours)
    min_size = 20
    edge_dist_req = 20
    bbox_outside_frame = []
    bbox_wrong_size = []

    # print("contacts:" +str(num_contours))
    # print(reversed(contact_range))

    for i in reversed(range(num_contours)):
        # print("i: "+str(i))
        too_small = 0
        near_edge = 0
        xmin = x[i]
        ymin = y[i]
        xmax = x[i]+w[i]
        ymax = y[i]+h[i]
        # cropframe = frame[y[i]:y[i]+h[i], x[i]:x[i]+w[i]]
        # cv2.imshow('cropframe', cropframe)
        # boundimg = cv2.rectangle(frame, (x[i], y[i]),
        #                        (x[i]+w[i], y[i]+h[i]), (0, 0,200), 2)
        # cv2.imshow('qc', frame)
        # cv2.waitKey(0) & 0xff
        if w[i] < min_size:
            # print("width < min size ")
            dif_w = min_size - w[i]
            # print("difference in x" + str(dif_w))
            move_x = int(dif_w/2)

            x[i]=xmin-move_x
            w[i]=min_size
            too_small = 1

        if h[i] < min_size:
            # print("height < min size ")
            dif_h = min_size - h[i]
            # print("difference in y" + str(dif_h))
            move_y = int(dif_h/2)

            y[i]=ymin-move_y
            h[i]=min_size
            too_small = 1


        # check objects are away from frame edge
        if (xmin < edge_dist_req
            or ymin < edge_dist_req
            or xmax > (vw-edge_dist_req)
            or ymax > (vh-edge_dist_req)):

            near_edge = 1
            print("image too close to edge... delete")
            boundimg = cv2.rectangle(frame, (x[i], y[i]),
                                    (x[i]+w[i], y[i]+h[i]), (0, 0, 0), 1)
            del x[i]
            del y[i]
            del w[i]
            del h[i]

        elif too_small == 1:
            boundimg = cv2.rectangle(frame, (x[i], y[i]),
                                    (x[i]+w[i], y[i]+h[i]), (0, 200, 0), 1)

        else:
            boundimg = cv2.rectangle(frame, (x[i], y[i]),
                                    (x[i]+w[i], y[i]+h[i]), (0, 0, 200), 1)

    print("new contours list:")
    print(contours_list)
    cv2.imshow('qc', frame)

    return contours_list

def qcTextDisplay():

    print("")
    print('Quality Control Image Options')
    print('enter - check bboxes')
    print('spacebar - skip frame')
    print('backspace - go back one frame')
    print('d - delete xml file')

    return

def qcOptionsTextDisplay():

    print("")
    print("checked bbox size and proximity to frame edge")
    print("d - select boxes to delete")
    print("s - select images to add")
    print("enter - export new xml")
    print("esc - cancel")

    return

def deleteXML(xml_fname):

    if os.path.isfile(xml_fname):
        os.remove(xml_fname)
        print("xml deleted")

        # print(xml_fname)

    return
