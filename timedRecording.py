#===================================================
#
# File Name: timedRecording.py
# 
# Purpose: To record footage during a set time interval 
#
# Creation Date: 05-06-2017
#
# Last Modified: Wed 30 May 2018 02:42:55 PM PDT
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
from datetime import datetime, timedelta
from dateutil import parser
import cv2
import time

def initializeVideo():
    kcw = KeyClipWriter()
    cam = cv2.VideoCapture(0)
    return kcw, cam

# def checkTimeInInterval(starttime, endtime):
    # start = parser.parse(starttime)
    # end = parser.parse(endtime)
    # if start < datetime.now() < end:
        # return True
    # else:
        # return False

# def isTimeBeforeStart(starttime):
    # start = parser.parse(starttime)
    # return datetime.now() < start

# def isTimeAfterEnd(endtime):
    # end = parser.parse(endtime)
    # return datetime.now() > end

def printTimeRemaining(endtime):
    timeleft = (endtime - datetime.now()).total_seconds()
    if not int(timeleft) % 5:
        print(f'\rRecording Time Left: {int(timeleft):5} seconds', end='', flush=True)
        # print('\rTime left: {} seconds'.format(int(timeleft)), end='', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--start', help='Time and date to start recording')
    ap.add_argument('-e', '--end', help='Time and date to end recording')
    ap.add_argument('-o', '--output', help='Path to save video to')
    args = vars(ap.parse_args())

    starttime = parser.parse(args['start'])
    endtime = parser.parse(args['end'])
    print(f'Recording to start at: {datetime.strftime(starttime,"%Y/%m/%d %I:%M:%S")}')
    print(f'Recording to end at:   {datetime.strftime(endtime,"%Y/%m/%d %I:%M:%S")}')

    #Sleep until starting time, periodically checking time
    while starttime > datetime.now():
        timetill = (starttime - datetime.now()).total_seconds()
        print(f'\rTime until recording begins: {int(timetill):5} seconds', end='', flush=True)
        time.sleep(5)

    #Wake up and start!
    print('\n---- Video recording starting! ----')
    kcw, cam = initializeVideo()
    for i in range(10):
        (grabbed, frame) = cam.read()

    while datetime.now() <= endtime:
        (grabbed, frame) = cam.read()
        kcw.update(frame)

        if not kcw.recording:
            path = "{}/{}.avi".format(args['output'], datetime.now().strftime('%Y%m%d_%H%M%S'))
            kcw.start(path, cv2.VideoWriter_fourcc(*'FFV1'), 30)

        printTimeRemaining(endtime)


    #All done! Shut things down!
    kcw.finish()

if __name__ == '__main__':
    main()
    


