#===================================================
#
# File Name: motionprocess.py
# 
# Purpose: Find and record fireball events in grayscale
#
# Creation Date: 16-01-2017
#
# Last Modified: Wed 28 Jun 2017 04:25:58 PM PDT
#
# Created by: Jed Rembold
#
#===================================================

import argparse
from VidUtils import KeyClipWriter
import datetime
import time
import os
import sys
import logging
import shutil
import numpy as np
import cv2
import cProfile
from threading import Thread

import shared



def takeSnapshot(path):
    '''
    Function to be run in a thread to take periodic snapshots of the
    night sky. Mainly to serve as a confirmation for the sky state
    at any particular hour of the night
    '''
    while True:
        while (grabbed and shared.ANALYZE_ON):
            date_num = datetime.datetime.utcnow()
            date_str = date_num.strftime("%Y%m%d_%H%M%S")
            # cv2.imwrite(path + "/Snap_" + date_str + ".png", frame)
            if gray is not None:
                savefile = gray
                for i in range(10):
                    newframe = gray
                    savefile = cv2.add(cv2.subtract(savefile,savefile.mean()), cv2.subtract(newframe, newframe.mean()))
                savefile = cv2.cvtColor(savefile, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(path + '/'+ date_str + "_Snap.png", savefile)
                logging.info('Snapshot!!')
            time.sleep(30) #Sleep 30 mins
        time.sleep(10) #In analysis off, wait 10 minutes then check again

def weightaccum( old_frame, new_frame, weight ):
    ''' Function calculates an accumulated weighted average. This function
    serves as a wrapper to the basic script to recast the arrays in the
    needed datatype. '''
    # Recasting in float32 as accumulateWeighted requires this datatype
    old_frame = old_frame.astype(np.float32)
    new_frame = new_frame.astype(np.float32)
    # Doing the weighted average
    cv2.accumulateWeighted(new_frame, old_frame, weight)
    # Returning the result cast back in uint8
    return old_frame.astype(np.uint8)


def drawBoundingHough(frame, x1, x2, y1, y2 ):
    ''' Function to draw a red box around all detected objects in a scene
    '''
    size = 40
    midx = int((x2+x1)/2)
    midy = int((y2+y1)/2)
    p1x = max(midx-size,0)
    p1y = max(midy-size,0)
    p2x = min(midx+size, frame.shape[1])
    p2y = min(midy+size, frame.shape[0])
    cv2.rectangle(frame, (p1x, p1y), (p2x,p2y), 255, 1)


def analyze(buffsize, savepath, headless, vpath=None ):
    '''
    This is the main analysis script. Reads in the data from the camera,
    checks for objects, opens threads to save the data if an object is
    found. '''

    global grabbed, gray

    # Saved video or live video?
    if vpath is None:
        # Must be live video
        cam = cv2.VideoCapture(shared.SRC)
        time.sleep(1.0)
    else:
        # Must be saved video
        cam = cv2.VideoCapture(vpath)

    # Initializing everything
    avg = None
    accum = None
    framenum = 0
    kcw = KeyClipWriter(bufSize=buffsize)
    consecFrames = 0
    logging.info("New Observation Run Started")
    logging.info("Detection params -- Length: {}, Threshold: {}, MinLine: {}, LineSkip: {}".format(shared.DETECT.LENGTH, shared.DETECT.THRESHOLD, shared.DETECT.MINLINE, shared.DETECT.LINESKIP))
    logging.info("Clips saved to {}".format(savepath))

    disk_full = False

    # Grab one frame to initialize 
    (grabbed, frame) = cam.read()

    # Start snapshoting thread
    snapshots = Thread(target=takeSnapshot, args=(savepath,))
    snapshots.daemon = True
    snapshots.start()

    while grabbed:
        # Check the time to see what hour of the day it is
        curr_hour = time.localtime().tm_hour

        if vpath is not None:
            shared.ANALYZE_ON = True
        else:
            # If it is nighttime turn on analyzing, else sleep for 5 mins before checking again
            if not shared.STARTTIME <= curr_hour < shared.ENDTIME:
                shared.ANALYZE_ON = True
                logging.info("A new night has arrived! Frame analysis beginning!")
            else:
                # print("Daylight! Sleeping...")
                time.sleep(5)

        # When analyzing is turned on
        while shared.ANALYZE_ON:
            # Grab the current time for fps purposes
            startframetime = time.time()

            # Grab the latest frame
            (grabbed, frame) = cam.read()

            # If no frame to grab, either we have an issue or we are at the end
            if not grabbed:
                logging.warning("No frame was grabbed. Exiting run loop.")
                break

            # Initialize frame as no containing an event, so that consecutive non-event frame counter should augment
            updateConsecFrames = True

            # Process Frame
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            B = gray.copy()


            # Resize grayscale image for faster image processing (reducing each dimension by a factor of 2)
            gray = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            # Average Frame
            if avg is None:
                avg = gray.copy()
            # Using a very small weighted average to vastly prefer older 
            # frames that do not include any current fireballs, 
            # thereby creating our background
            avg = weightaccum( avg, gray, 0.05 )

            # Subtract and Threshold
            delta = cv2.subtract(gray, avg)
            thresh = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1]
            kernel = np.ones([3,3])
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Accumulated binaries to create line for Hough transform to find
            if accum is None:
                accum = thresh.copy()
            # Using a larger weighted average to create a composite of the 
            # more recent frames, which should include the fireball trail
            accum = weightaccum( accum, thresh, 0.1 )
            # Thresholding the accumulated image for Hough Transforming
            accum_thresh = cv2.threshold(accum, 10, 255, cv2.THRESH_BINARY)[1]

            # Writing date and time in UTC to lower left corner of output frame in green
            date_num = datetime.datetime.utcnow()
            date_str = date_num.strftime("%Y%m%d %H%M%S.%f")
            cv2.putText(B, date_str + " UTC", (10, B.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255)

            # The Hough Transform
            lines = cv2.HoughLinesP( accum_thresh, shared.DETECT.LENGTH, shared.DETECT.ANGLES, shared.DETECT.THRESHOLD, shared.DETECT.MINLINE, shared.DETECT.LINESKIP )
            # If we detect lines, draw a box around each one and initialize or perpetuate recording
            if lines is not None:
                # for eachline in lines:
                    # for x1, y1, x2, y2 in eachline:
                        # Bounding box edges multiplied by 2 to account for dimension reduction earlier
                        #drawBoundingHough(R, 2*x1, 2*x2, 2*y1, 2*y2)
                updateConsecFrames = False
                consecFrames = 0


                # If not already recording, start the recording!
                if not kcw.recording:
                    # Get the current free space on the disk in megabytes
                    free_space = shutil.disk_usage(savepath).free*1E-6
                    # If we have more than 500MB available, go ahead and start the recording, else stop program
                    if free_space > 500:
                        # print("New event found at time: {}".format(date_str))
                        p = "{}/{}.avi".format(savepath, date_num.strftime("%Y%m%d_%H%M%S_%f"))
                        # kcw.start(p, cv2.VideoWriter_fourcc(*'H264'), 30)
                        kcw.start(p, cv2.VideoWriter_fourcc(*'FFV1'), 30)
                        # kcw.start(p, cv2.VideoWriter_fourcc(*'HFYU'), 30)
                        logging.info("New event detected. Video name: {}".format(p))
                    else:
                        logging.warning("Terminating observation run due to lack of storage")
                        disk_full = True
                        shared.ANALYZE_ON = False

            # Create output image with grayscale image saved as blue color
            # output = cv2.merge([B, G, R])
            output = cv2.cvtColor(B, cv2.COLOR_GRAY2RGB)

            #Wrapping things up
            if updateConsecFrames:
                consecFrames += 1

            #Update buffer with latest frame
            kcw.update(output)

            #If too many frames w/o an event, stop recording
            if kcw.recording and consecFrames == buffsize:
                kcw.finish()
                logging.info("Event completed and video recording finished") 

            # Show windows if desired
            if not headless:
                cv2.imshow("Output", output)
                # cv2.imshow("Timestamp",G)
                # cv2.imshow("Box", R)
                cv2.imshow("Background", avg)
                cv2.imshow("Subtracted",delta)
                cv2.imshow("Accumulated", accum_thresh)
                
                key = cv2.waitKey(1)

                # Exit script early. This does not work if running in headless mode
                if key == ord("q"):
                    shared.ANALYZE_ON = False
                    logging.info("Analysis manually stopped!")

            framenum += 1
            endframetime = time.time()

            # Occasionally print out the current framerate
            if not framenum % 5:
                shared.FRAMERATE = 1/(endframetime-startframetime)
                # print("Operating at {} frames per second".format(1/(endframetime-startframetime)))

            # Check time
            if shared.STARTTIME <= time.localtime().tm_hour < shared.ENDTIME:
                shared.ANALYZE_ON = False
                logging.info("Day has come. Analysis going to sleep.")

            if not shared.RUNNING:
                break

        if kcw.recording:
            kcw.finish()

        # if key == ord("q"):
            # break

        if not shared.RUNNING:
            shared.ANALYZE_ON = False
            break

        if disk_full:
            break

    logging.info("Observing session finished.")

if __name__ == "__main__":
    #Setting up logging formatting and location
    logging.basicConfig(
            #Logging to Log.txt in same directory as script
            filename = 'Observation_Log.txt',
            level = logging.DEBUG,
            style = '{',
            format = '[{asctime}.{msecs:<3.0f}] [{levelname:^8}]: {message}',
            # datefmt = '%H:%M:%S',
            datefmt = '%Y/%m/%d %H:%M:%S',
            # filemode = 'w',
            )

    shared.RUNNING = True
    grabbed = False
    shared.ANALYZE_ON = False
    gray = []
    shared.STARTTIME = 4
    shared.ENDTIME = 12

    # Construct the Argument Parser and Parse away!
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to desired video file")
    ap.add_argument("-b", "--buffer-size", type=int, default=32, help="Buffer size of video clip writer (defaults to 32)")
    ap.add_argument("-o", "--output", help="Path to desired output files")
    ap.add_argument("-n", "--headless", action='store_true', help="Pass option to suppress monitoring output")
    args = vars(ap.parse_args())
    # print(args)

    # cProfile.run('main()')
    analyze(args['buffer_size'], args['output'], args['headless'],args['video'])
