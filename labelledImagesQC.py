#!/usr/bin/python3
'''
Title: labelledImagesQC.py
Author: JR Keane
Created: 29 July 18
Last modified: 29 July 18
Version: 1.0
Version notes:
1.0     Quality Control of Labelled images before they go off to jam tensorflow
'''

import os
import numpy as np
import cv2
import re

# checking bndboxes are within frame boundaries
# checking that mins are less than maxes
# checking that no images are less than 5pix * 5pix

def check_bndboxes(lines, lineofbndbox, vw,vh):

    bbox_outside_frame = []
    bbox_wrong_size = []
    for line in lineofbndbox:
        xmin = lines[line+1]
        xmin = int(xmin[xmin.find('<xmin>')+6:xmin.rfind('</xmin>')])
        ymin = lines[line+2]
        ymin = int(ymin[ymin.find('<ymin>')+6:ymin.rfind('</ymin>')])
        xmax = lines[line+3]
        xmax = int(xmax[xmax.find('<xmax>')+6:xmax.rfind('</xmax>')])
        ymax = lines[line+4]
        ymax = int(ymax[ymax.find('<ymax>')+6:ymax.rfind('</ymax>')])

        bbox = [xmin, ymin, xmax, ymax]
        boundimg = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0,0), 1)
        if xmin > 10 and ymin>10  and xmax < (vw-10) and ymax < (vh-10):
            continue

        else:
            bbox_outside_frame.append(bbox)

        if xmin < xmax and ymin < ymax:
            continue

        else:
            bbox_wrong_size.append(bbox)

    if bbox_outside_frame:
        print("boxes outside frame")
        print(bbox_outside_frame)

    if bbox_wrong_size:
        print("mins greater than maxes")

    return bbox_wrong_size, bbox_outside_frame

oldPath = '/media/kookaburra/JimsDisk/Ninox/flight1images/' #where the processed lives
labelIMGPath = '/media/kookaburra/JimsDisk/Ninox/flight1images/label/' #QC sample of good files
newPath = '/media/kookaburra/JimsDisk/Ninox/flight1images/newXML/' #save changed files and outputs to here:
qc_text_file = newPath + "qc.txt"
qc = open(qc_text_file, 'w')
filenames = []
imagenames = []
show_images = 'true'
checked_im = 0

for file in os.listdir(oldPath):
    if file.endswith(".xml"):
        filenames.append(file)
        file2 = (file[:-4]+'.jpg')
        imagenames.append(file2)

total_im = len(filenames)
# filenames = ['Ninox10952.xml']
# imagenames = ['Ninox10952.jpg']
print("number files: " + str(total_im)) # this is how many processed files there are to check
qc.write("total frames to check: " + str(total_im)+'\n')

for i in range(len(filenames)):
    checked_im +=1

    imagePath = oldPath+imagenames[i]
    xmlPath = oldPath + filenames[i]
    current_frame = filenames[i]
    frame = cv2.imread(imagePath)
    vidheight, vidwidth, channels = frame.shape
    vidwidth = int(vidwidth)
    vidheight = int(vidheight)

    f = open(xmlPath, 'r')

    lines = f.readlines()
    # print(lines)
    # print(len(lines))
    lineofbndbox = []

    for line in range(len(lines)):
        if "<bndbox>" in lines[line]:
            lineofbndbox.append(line)

        if "<width>" in lines[line]:
            lineofvidwidth = int(line)
            # print(lineofvidwidth)
        if "<height>" in lines[line]:
            lineofvidheight = int(line)

    ###########################################
    ############ check frame ##################

    vw = lines[lineofvidwidth]
    vw = int(vw[vw.find('<width>')+7:vw.rfind('</width>')])
    vh = lines[lineofvidheight]
    vh = int(vh[vh.find('<height>')+8:vh.rfind('</height>')])

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

    #############################################
    ################ check bboxes ###############

    bbox_wrong_size, bbox_outside_frame = check_bndboxes(lines, lineofbndbox, vw, vh)

    #############################################
    ################ after check ################
    if bbox_wrong_size or bbox_outside_frame:
        qc.write(current_frame+'\n')
        qc.write("error:\n")
        print("error")
        print("")

        if bbox_wrong_size:
            print(bbox_wrong_size)
            qc.write("bboxes wrong size\n")

        elif bbox_outside_frame:
            print(bbox_outside_frame)
            qc.write("bboxes outside frame\n")

    if show_images == 'true':
        cv2.imshow('image', frame)
        k = cv2.waitKey(0) & 0xff
        cv2.destroyAllWindows()

        if k == 27:
            break

print("total of "+str(checked_im) + " checked")
qc.write("total images checked: "+str(checked_im)+'\n')
qc.close()
cv2.destroyAllWindows()

qc = open()
