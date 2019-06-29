"""
Script to be run against recorded videos to test detection
algorithms and settings.
"""

import argparse
import numpy as np
import cv2

import shared

# import motionprocess as mp


def weightaccum(old_frame, new_frame, weight):
    """ Function calculates an accumulated weighted average. This function
    serves as a wrapper to the basic script to recast the arrays in the
    needed datatype. """
    # Recasting in float32 as accumulateWeighted requires this datatype
    old_frame = old_frame.astype(np.float32)
    new_frame = new_frame.astype(np.float32)
    # Doing the weighted average
    cv2.accumulateWeighted(new_frame, old_frame, weight)
    # Returning the result cast back in uint8
    return old_frame.astype(np.uint8)


def detect(frame, avg, accum):
    # Process Frame
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Resize grayscale image for faster image processing (reducing each dimension by a factor of 2)
    gray = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # Average Frame
    if avg is None:
        avg = gray.copy()
    # Using a very small weighted average to vastly prefer older
    # frames that do not include any current fireballs,
    # thereby creating our background
    avg = weightaccum(avg, gray, 0.01)

    # Subtract and Threshold
    delta = cv2.subtract(gray, avg)
    cv2.imshow("Delta", delta)
    thresh = cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY)[1]
    # kernel = np.ones([3,3])
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Thresh", thresh)

    # Accumulated binaries to create line for Hough transform to find
    if accum is None:
        accum = thresh.copy()
    # Using a larger weighted average to create a composite of the
    # more recent frames, which should include the fireball trail
    accum = weightaccum(accum, thresh, 0.2)
    # Thresholding the accumulated image for Hough Transforming
    accum_thresh = cv2.threshold(accum, 10, 255, cv2.THRESH_BINARY)[1]
    # The Hough Transform
    lines = cv2.HoughLinesP(
        accum_thresh,
        rho=shared.DETECT.LENGTH,
        theta=shared.DETECT.ANGLES,
        threshold=shared.DETECT.THRESHOLD,
        minLineLength=shared.DETECT.MINLINE,
        maxLineGap=shared.DETECT.LINESKIP,
    )
    # If we detect lines, draw a box around each one and initialize or perpetuate recording
    marked = frame.copy()
    avgpt = None
    if lines is not None:
        for eachline in lines:
            for x1, y1, x2, y2 in eachline:
                cv2.line(frame, (x1 * 2, y1 * 2), (x2 * 2, y2 * 2), (0, 0, 255))
        x1, y1, x2, y2 = lines[0, 0]
        avgpt = ((x1 + x2) / 2, (y1 + y2) / 2)
        linefound = True
    else:
        linefound = False

    return linefound, avg, accum, accum_thresh, marked, avgpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to desired video file")
    args = vars(ap.parse_args())

    vid = cv2.VideoCapture(args["video"])
    grabbed, frame = vid.read()

    avg = None
    accum = None

    cv2.namedWindow("Frame")
    cv2.moveWindow("Frame", 100, 400)
    cv2.namedWindow("Background")
    cv2.moveWindow("Background", 95, 150)
    cv2.namedWindow("Delta")
    cv2.moveWindow("Delta", 105 + 360, 150)

    cv2.namedWindow("Marked")
    cv2.moveWindow("Marked", 900, 400)
    cv2.namedWindow("Thresh")
    cv2.moveWindow("Thresh", 895, 150)
    cv2.namedWindow("Accumulated")
    cv2.moveWindow("Accumulated", 905 + 360, 150)

    while grabbed:
        linefound, avg, accum, accum_thresh, marked, avgpt = detect(frame, avg, accum)

        cv2.imshow("Frame", frame)
        cv2.imshow("Background", avg)
        cv2.imshow("Accumulated", accum_thresh)
        cv2.imshow("Marked", marked)
        if linefound:
            print(avgpt)

        key = cv2.waitKey()
        if key == ord("q"):
            break

        grabbed, frame = vid.read()


if __name__ == "__main__":
    main()
