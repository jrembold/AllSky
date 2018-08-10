#===================================================
#
# File Name: timedRecording.py
# 
# Purpose: To record footage during a set time interval 
#
# Creation Date: 05-06-2017
#
# Last Modified: Thu 31 May 2018 05:26:32 PM PDT
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

def initializeVideo(src):
    kcw = KeyClipWriter()
    cam = VideoStream(src)
    cam.start()
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
    if not round(timeleft,1) % 1:
        # print(f'\rRecording Time Left: {int(timeleft):5} seconds', end='', flush=True)
        print('\rTime left: {:5} seconds'.format(int(timeleft)), end='', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--start', help='Time and date to start recording')
    ap.add_argument('-e', '--end', help='Time and date to end recording')
    ap.add_argument('-o', '--output', help='Path to save video to')
    ap.add_argument('--src', default=0, help='Video Source Number. Defaults to 0')
    args = vars(ap.parse_args())

    starttime = parser.parse(args['start'])
    endtime = parser.parse(args['end'])
    src = int(args['src'])
    # print(f'Recording to start at: {datetime.strftime(starttime,"%Y/%m/%d %I:%M:%S")}')
    # print(f'Recording to end at:   {datetime.strftime(endtime,"%Y/%m/%d %I:%M:%S")}')
    print('Recording to start at: {}'.format(datetime.strftime(starttime,"%Y/%m/%d %I:%M:%S")))
    print('Recording to end at: {}'.format(datetime.strftime(endtime,"%Y/%m/%d %I:%M:%S")))

    kcw, cam = initializeVideo(src)

    #Sleep until starting time, periodically checking time
    while starttime > datetime.now():
        timetill = (starttime - datetime.now()).total_seconds()
        # frame = cam.read()
        if not round(timetill,1) % 1:
            # print(f'\rTime until recording begins: {int(timetill):5} seconds', end='', flush=True)
            print('\rTime until recording begins: {:5} seconds'.format(int(timetill)), end='', flush=True)

    #Wake up and start!
    print('\n---- Video recording starting! ----')
    cam.start()

    while datetime.now() <= endtime:
        frame = cam.read()
        kcw.update(frame)

        if not kcw.recording:
            path = "{}/{}.avi".format(args['output'], datetime.now().strftime('%Y%m%d_%H%M%S'))
            kcw.start(path, cv2.VideoWriter_fourcc(*'XVID'), 30)

        printTimeRemaining(endtime)

    cam.stop()
    while not cam.Q.empty():
        kcw.update(frame)



    #All done! Shut things down!
    print('')
    kcw.finish()


if __name__ == '__main__':
    main()
    


