#####################################
#
#
# Identifying Objects & Their Magnitudes
# 
# Luke Galbraith Russell
#
# Willamette University
#
#
#####################################

import numpy as np
import cv2
import argparse
import math


####################################
# Creating function for mouse click
# Left Click is for Reference Star
# Right Click is for the Object
####################################

def click(event, x, y, flags, param):
    global ReferenceStarLoc
    global ReferenceBackgroundMag
    global ReferenceBackgroundLoc
    global ObjectBackgroundMag
    global ObjectBackgroundLoc
    global ObjectLoc

    if event == cv2.EVENT_LBUTTONDOWN:
        ReferenceStarLoc = (x,y)

    elif event == cv2.EVENT_LBUTTONUP:
        ReferenceBackgroundMag = img[y,x]
        ReferenceBackgroundLoc = (x,y)
        cv2.circle(img2, ReferenceStarLoc, int(math.sqrt((ReferenceStarLoc[0] - ReferenceBackgroundLoc[0])**2 + (ReferenceStarLoc[1] - ReferenceBackgroundLoc[1])**2)),100,5)
        cv2.imshow("window", img2*10)

    elif event == cv2.EVENT_RBUTTONDOWN:
        ObjectLoc = (x,y)
    
    elif event == cv2.EVENT_RBUTTONUP:
        ObjectBackgroundMag = img[y,x]
        ObjectBackgroundLoc = (x,y) 
        cv2.circle(img2, ObjectLoc, int(math.sqrt((ObjectLoc[0] - ObjectBackgroundLoc[0])**2 + (ObjectLoc[1] - ObjectBackgroundLoc[1])**2)),99,5)
        cv2.imshow("window", img2*10)

####################################
# argument use in order to identify picture in command line
# open with "python /path/to/script.py --image /path/to/picture.jpg"
# only works with JPG as of now
####################################

ap=argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
args= vars(ap.parse_args())

####################################
# reading in the jpg, cloning a copy, and then converting that copy to grayscale
# creating window for it to pop up in
# activating mouse function from above
# and ESC on the picture will exit it out
####################################

image = cv2.imread(args["image"])
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(0,0), fx=2, fy=2)
img2 = img.copy()
cv2.namedWindow("window")
cv2.setMouseCallback("window", click)
cv2.imshow("window", img*10)
cv2.waitKey(0)
cv2.destroyAllWindows()

####################################
# This is a way to find the brightest area near the pixel you clicked
# Take the distance to background in either direction, search grid for brightest point
# sum all the values of the square area selected
####################################

def MagnitudeFinder(Loc, BackgroundMag, BackgroundLoc):

    Distance = (np.absolute(np.subtract(Loc, BackgroundLoc)))
    Radius = np.amax(Distance)
    print('The radius of the reference star is', Radius)

    Range=[]
    for i in range(-Radius,Radius):
        for j in range(-Radius,Radius):
            if i**2 + j**2  <  Radius**2:
                Range.append((i + Loc[0], j + Loc[1])[::-1])
    print(Range)
    MagValue = 0
    for i in Range:
        MagValue = MagValue + (img[i] - BackgroundMag)
    print('The magnitude value is',MagValue)
    return(MagValue)

ReferenceMagValue = MagnitudeFinder(ReferenceStarLoc, ReferenceBackgroundMag, ReferenceBackgroundLoc)

ObjectMagValue = MagnitudeFinder(ObjectLoc, ObjectBackgroundMag, ObjectBackgroundLoc)

####################################
#Photometry for finding the catalog value
####################################
InstrumentalMagnitude = -2.5*np.log(ReferenceMagValue)
CatalogMagnitude = float(input('Enter catalog magnitude: '))
Offset = InstrumentalMagnitude - CatalogMagnitude 

ObjectCatalogValue = -2.5*np.log(ObjectMagValue) - Offset
print('The catalog value of the object is *maybe*',  ObjectCatalogValue)

