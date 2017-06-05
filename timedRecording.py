#===================================================
#
# File Name: timedRecording.py
# 
# Purpose: To record footage during a set time interval 
#
# Creation Date: 05-06-2017
#
# Last Modified: Mon 05 Jun 2017 04:16:07 PM PDT
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
from KeyClipWriter import KeyClipWriter
from datetime import datetime
from dateutil import parser
import cv2

def initializeVideo():
    kcw = KeyClipWriter()
    cam = cv2.VideoCapture(0)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--start', help='Time and date to start recording')
    ap.add_argument('-e', '--end', help='Time and date to end recording')
    args = vars(ap.parse_args())

    #Sleep until starting time, periodically checking time
    while isTimeBeforeStart(args['start']):
        time.sleep(5)

    #Wake up and start!
    kcw, cam = initializeVideo()
    (grabbed, frame) = cam.read()

    while grabbed:
        kcw.upd

    print(checkTimeInInterval(args['start'], args['end']))

if __name__ == '__main__':
    main()
    


