# each box is a region of inerest
#e.g set up to read heading

def mlROI(frame):
    # each box is a region of inerest
    #e.g set up to read heading
    boxx = [32] # box start point (pixels from left of screen)
    boxy = [50] # box start point (pixels from Top! of screen)
    boxw = [35] # box width
    boxh = [19] # box height

    for i in range(len(boxx)):
        trackbox = (boxx[i],boxy[i],boxw[i], boxh[i])
        p1 = (int(trackbox[0]), int(trackbox[1]))
        p2 = (int(trackbox[0] + trackbox[2]), int(trackbox[1] + trackbox[3]))
        cropframe=frame[p1[1]:p2[1], p1[0]:p2[0]]
        popup = cv2.imshow("cropped", cropframe)
        k = cv2.waitKey(0) & 0xff

    return
