import numpy as np
import cv2 as cv
import sys


def initialize_tracking_alg():
    # Set up tracker.
    # Instead of MIL, you can also use
    # works: MIL,KCF, GOTURN, CSRT
    tracker_types = ['CSRT', 'MIL', 'KCF', 'GOTURN', 'BOOSTING', 'MEDIANFLOW', 'TLD', 'MOSSE']
    tracker_type = tracker_types[0]
    tracker = None

    match tracker_type:
        case 'BOOSTING':
            tracker = cv.TrackerBoosting_create()
        case 'MIL':
            tracker = cv.TrackerMIL_create()
        case 'KCF':
            tracker = cv.TrackerKCF_create()
        case 'TLD':
            tracker = cv.TrackerTLD_create()
        case 'MEDIANFLOW':
            tracker = cv.TrackerMedianFlow_create()
        case 'GOTURN':
            tracker = cv.TrackerGOTURN_create()
        case 'MOSSE':
            tracker = cv.TrackerMOSSE_create()
        case 'CSRT':
            tracker = cv.TrackerCSRT_create()

    return tracker, tracker_type


def get_interested_object(cap: cv.VideoCapture):
    cap.read()
    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # if frame is read correctly ret is True
    if not ret:
        print('Cannot read video file')
        sys.exit()

    bbox = cv.selectROI('roi', frame)
    cv.destroyWindow('roi')

    return frame, bbox


def read_key():
    return cv.waitKey(1) & 0xff


if __name__ == '__main__':
    isTrackerInitialized = False

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    tracker, tracker_type = initialize_tracking_alg()

    while True:
        # Read a new frame
        ok, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if not ok:
            break

        if isTrackerInitialized:
            # Start timer
            timer = cv.getTickCount()

            # Update tracker
            ok, bbox = tracker.update(frame)

            # Calculate Frames per second (FPS)
            fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv.rectangle(frame, p1, p2, (255, 0, 0), 6, 1)
            else:
                # Tracking failure
                cv.putText(frame, "Tracking failure detected", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # Display tracker type on frame
            cv.putText(frame, tracker_type + " Tracker", (100, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display FPS on frame
            cv.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display point
            cv.putText(frame, "point : " + str((int(bbox[0]), int(bbox[1]))), (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75,
                       (160, 0, 0), 2)

        # Display result
        cv.imshow("Tracking", frame)

        # Exit if ESC pressed
        match read_key():
            case 27:  # ESC
                cap.release()
                cv.destroyAllWindows()
                break
            case 115:  # s
                # get one for tracking
                frame, bbox = get_interested_object(cap)
                # Initialize tracker with first frame and bounding box
                ok = tracker.init(frame, bbox)
                isTrackerInitialized = True

