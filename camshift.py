import numpy as np
import cv2 as cv
import sys

def get_interested_object(cap: cv.VideoCapture):
    cap.read()
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print('Cannot read video file')
        sys.exit()

    bbox = cv.selectROI('roi', frame)
    cv.destroyWindow('roi')

    return frame, bbox


if __name__ == '__main__':
    cap = cv.VideoCapture(1)

    frame, bbox = get_interested_object(cap)

    # take first frame of the video
    # ret,frame = cap.read()

    # setup initial location of window
    x, y, w, h = bbox
    track_window = (x, y, w, h)

    # set up the ROI for tracking
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

    while(1):
        ret, frame = cap.read()

        if ret == True:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            # apply camshift to get the new location
            ret, track_window = cv.CamShift(dst, track_window, term_crit)

            # Draw it on image
            pts = cv.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv.polylines(frame,[pts],True, 255,2)
            cv.imshow('img2',img2)

            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break