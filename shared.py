#===================================================
#
# File Name: shared.py
# 
# Purpose: To give a shared globals module 
#
# Creation Date: 29-05-2017
#
# Last Modified: Wed 31 May 2017 05:21:17 PM PDT
#
# Created by: Jed Rembold
#
#===================================================

RUNNING = False
ANALYZE_ON = False
SAVELOC = None
STARTTIME = None
ENDTIME = None
FRAMERATE = None

class DETECT:
    LENGTH = 4
    ANGLES = 0.10
    THRESHOLD = 20
    MINLINE = 10
    LINESKIP = 3
