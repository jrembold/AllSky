#===================================================
#
# File Name: timedRecording.py
# 
# Purpose: To record footage during a set time interval 
#
# Creation Date: 05-06-2017
#
# Last Modified: Mon 05 Jun 2017 05:08:07 PM PDT
#
# Created by: Jed Rembold
#
#===================================================

'''
Script should check the local clock to a set time interval and
if within that time interval, should record. Written with the
intention of being used to capture known events (Iridium
flares) for finetuning of analysis and capture software.
'''

import argparse
from VidUtils import KeyClipWriter
from VidUtils import VideoStream
from datetime import datetime, timedelta
from dateutil import parser
import cv2
import time

def initializeVideo():
    kcw = KeyClipWriter()
    cam = VideoStream()
    return kcw, cam

def checkTimeInInterval(starttime, endtime):
    start = parser.parse(starttime)
    end = parser.parse(endtime)
    if start < datetime.now() < end:
        return True
    else:
        return False

def isTimeBeforeStart(starttime):
    start = parser.parse(starttime)
    return datetime.now() < start

def isTimeAfterEnd(endtime):
    end = parser.parse(endtime)
    return datetime.now() > end

def printTimeRemaining(endtime):
    end = parser.parse(endtime)
    timeleft = (end - datetime.now()).total_seconds()
    if not int(timeleft) % 5:
        print('\rTime left: {} seconds'.format(int(timeleft)), end='', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--start', help='Time and date to start recording')
    ap.add_argument('-e', '--end', help='Time and date to end recording')
    ap.add_argument('-o', '--output', help='Path to save video to')
    args = vars(ap.parse_args())

    #Sleep until starting time, periodically checking time
    while isTimeBeforeStart(args['start']):
        time.sleep(5)

    #Wake up and start!
    kcw, cam = initializeVideo()
    (grabbed, frame) = cam.read()

    while not isTimeAfterEnd(args['end']):
        (grabbed, frame) = cam.read()
        kcw.update(frame)

        if not kcw.recording:
            path = "{}/{}.avi".format(args['output'], datetime.now().strftime('%Y%m%d_%H%M%S'))
            kcw.start(path, cv2.VideoWriter_fourcc(*'FFV1'), 30)

        printTimeRemaining(args['end'])


    #All done! Shut things down!
    kcw.finish()
    cam.stop()


if __name__ == '__main__':
    main()
    


