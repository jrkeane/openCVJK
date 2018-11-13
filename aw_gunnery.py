# target practice video
# jr keane
# SBLT, RAN
# developed for RAN innovation centre.
# August 2017
# splash detector...
# find the horizon
# zoom in on horizon
# rotate horizon to flat if needed
# detect splashes on the horizon
# solved.

'''functional core:
inputs don't change,
compose imperatives,

strategies for:
GOLANG
typescript

issues:
repeating myself
not abstracting things into functions
consistency
modularity

'''

import cv2
import numpy as np
import time
from sympy import *
from scipy import ndimage
import math
import init_gunnery3 as gun

def threshbar_hsv(thsv): #when i was running a slide bar

    hue_lower = cv2.getTrackbarPos('thsv', 'hsv')
    print(hue_lower)
    cv2.destroyWindow("mask")
    gun.red_rope(frame, middle_earth, hue_lower, weather)

    return hue_lower

#################################
############# init ##############
#################################

runningtime = 0 # initialise testbed timer
splashed = 0 # initialise counter
txtcount=0 # initialise counter
rotate = "false"
demo = 'n'
save_output = 'n'
weather = 'aw18'

if weather == 'good':
    pathName = '/home/ubuntu/openCV/Navy/Gunnery/'
    vidName = 'input_RANGE_GOOD_EXPLORE_3'
    fileName = vidName + '.mp4'

elif weather == 'bad':
    pathName = '/home/ubuntu/openCV/Navy/Gunnery/AMWC Gunnery Footage/'
    vidName = 'F150_20150121083643.mp4'
    fileName = pathName + vidName

elif weather == 'aw18':
    pathName = '/home/ubuntu/openCV/Navy/GRIM/'
    vidName = 'grim_test_examplesfirst.avi'
    fileName = pathName + vidName

# print(fileName)
playback = cv2.VideoCapture(fileName) # load video
cap = cv2.VideoCapture(0)

if save_output == 'y': # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_KEANE_DEMO2.mp4',fourcc, 20.0, (1440,1080)) #1440, 1080

if rotate == "true":
    framenum=60 #30 #80-125
    framestop=130

elif rotate == "false":
    framenum = 0 #80-127
    framestop = 500

if playback.isOpened(): #check vid, get properties
    pos_msec = playback.get(2) # position of file in ms or timestamp
    vidwidth = playback.get(3) #frame width
    vidheight = playback.get(4) #frame height
    framenum = playback.get(1) #frame
    totalframes = playback.get(7) #total number of frames
    print(vidwidth) # show video properties
    print(vidheight)
    print(totalframes)
    playback.set(1,framenum) #set starting frame of video for first imgrab
    newvidwidth = int(vidwidth*1) # set window size for testbed
    newvidheight = int(vidheight*1)
    cv2.namedWindow('target practice', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('target practice', newvidwidth, newvidheight)

    if demo == 'y':
        skyvidwidth=int(vidwidth*.8)
        skyvidheight=int(vidheight*.8)
        cv2.moveWindow('target practice', 10, 400)
        cv2.namedWindow('sky and sea', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('sky and sea', skyvidwidth, skyvidheight)
        cv2.moveWindow('sky and sea', 1600, 0)
    cv2.namedWindow('hsv', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('thsv', 'hsv', 130, 179, threshbar_hsv) #trackbar for intensity

start_time = time.time() # time it?
log_f = gun.openLogFile(vidName, pathName)

hue_lower = 130
#################################
########## vid loop #############
#################################


while(1): #run the video

    trackerok, frame = playback.read() #read frame from video
    print("Frame: " + str(framenum))
    if not trackerok: #check frame loaded
        print("tracker not ok. broken")
        break

    txtcount +=1 #counter for displaying onscreen text
    # gun.display_author_info(frame, txtcount) #show some stats
    if weather == 'good':
        thresh, imgray, bbox_sky, bbox_sea, rightmost, sea_bbox_min_y = gun.findingHorizon(frame, demo, weather) #find the horizon
        rot_angle, rotatey, middle_earth, hor_corr = gun.horizonStabilizer(vidwidth, vidheight, bbox_sky, bbox_sea, rightmost, sea_bbox_min_y) # get angle of boat/horizon
        if rotate =="false":
            ysky = bbox_sky[1]+bbox_sky[3]+5 #get some info about the horizon from position of sky and sea
            yocean = bbox_sea[1]-5
            cropx = int(0)
            cropx2= int(vidwidth)
            hor_ht=ysky-yocean

            if hor_ht<80: # manage the cropbox/search area around the horizon
                ysky=middle_earth+40
                yocean=middle_earth-40

            x_adj = 0
            y_adj = yocean
            x2adj = int(vidwidth) #int(vidwidth-vidwidth/5)
            y2adj = ysky
            horizon_boundaries = [x_adj, y_adj, x2adj, y2adj]
            zoom = frame[y_adj:y2adj,x_adj:x2adj]
            notex = int(0)
            notey = int(ysky+30) #track sea level for plotting text

            if txtcount>1 and txtcount<30: #display some info bout what the box is doing
                note = cv2.putText(frame, "TRACK HORIZON", (notex, notey), cv2.FONT_HERSHEY_SIMPLEX, 1, (250,250,250), 2)

            closing_c, splashbox = gun.splashDetect(horizon_boundaries, frame, zoom,hor_corr, demo) #detect splash on the horizon after horizon is detected and tracked
            horizon = cv2.rectangle(frame,(cropx,ysky),(cropx2,yocean),(255,255,255),2) #show box around the horizon

        elif rotate == "true": #rotate the horizon before splash detection

            flat = ndimage.rotate(frame,rot_angle,reshape=False)
            x_adj = int(vidwidth/4)
            y_adj = middle_earth-40
            x2adj = int(vidwidth-vidwidth/4)
            y2adj = middle_earth+40
            horizon_boundaries = [x_adj, y_adj, x2adj, y2adj]
            flatspin = flat[y_adj:y2adj, x_adj:x2adj] # Crop [y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
            closing_c, splashbox = gun.splashRotateDetect(flatspin, horizon_boundaries)
            cv2.imshow("flatspin", flatspin)
            cv2.imshow("black and white splash", closing_c)


    elif weather == 'bad':
        ############################
        ###### in development ######
        ############################
        frame = frame[250:850, 1:1439]
        middle_earth = int(vidheight/2)
        thresh, imgray, bbox_sky, bbox_sea, rightmost, sea_bbox_min_y = gun.findingHorizon(frame, demo, weather) #find the horizon
        print("bbox_sky:")
        print(bbox_sky)
        if bbox_sky:
            rot_angle, rotatey, middle_earth, hor_corr = gun.horizonStabilizer(vidwidth, vidheight, bbox_sky, bbox_sea, rightmost, sea_bbox_min_y)

        splashbox = []


        ###########################
        ###### end development ####
        ###########################

    target_x, target_y = gun.red_rope(frame, middle_earth, hue_lower, weather) #track red rope to target

    if splashbox:
        if splashed == 0:
            splash_text = gun.missingMissiles(frame, target_x, target_y, splashbox, y_adj)
            splashed = 1
            gun.writeLogFile(log_f, framenum, splash_text)

        else:
            notex = int(vidwidth*2/5)
            notey = int(vidheight/20)
            note = cv2.putText(frame, splash_text, (notex, notey), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    else:
            splashed = 0

    if save_output =='y': #save ouput video if required
        out.write(frame)

    cv2.imshow("target practice", frame)

    stop_time = time.time()
    framenum +=1

    if demo == 'y':
        k=cv2.waitKey(0) & 0xff

    else:
        k = cv2.waitKey(50) & 0xff

    if framenum > framestop: # end of vid
        print("end of vid. break.")
        break

    if k == 27: #hit esc to end playback
        print("broken")
        break

log_f.close()
cap.release() #all done, finish up
if save_output =='y':
    out.release()
cv2.destroyAllWindows()
