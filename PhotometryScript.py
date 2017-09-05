#####################################
#
#
# Identifying Objects and Their Magnitudes
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
import sympy as sp

#pixel = img[200,500]
#print(pixel)

####################################
# Creating function for mouse click
# Left Click is for Reference Star
# Right Click is for Object
####################################

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global ReferenceStarMag
        global ReferenceStarLoc
        ReferenceStarMag = img[x,y]
        ReferenceStarLoc = [x,y]
        return ReferenceStarMag, ReferenceStarLoc

    elif event == cv2.EVENT_LBUTTONUP:
        global ReferenceBackgroundMag
        global ReferenceBackgroundLoc
        ReferenceBackgroundMag = img[x,y]
        ReferenceBackgroundLoc = [x,y]
        return ReferenceBackgroundMag, ReferenceBackgroundLoc

    elif event == cv2.EVENT_RBUTTONDOWN:
        global ObjectMag
        global ObjectLoc
        ObjectMag = img[x,y]
        ObjectLoc = [x,y]
        return ObjectMag, ObjectLoc
    
    elif event == cv2.EVENT_RBUTTONUP:
        global ObjectBackgroundMag
        global ObjectBackgroundLoc
        ObjectBackgroundMag = img[x,y]
        ObjectBackgroundLoc = [x,y]
        return ObjectBackgroundMag, ObjectBackgroundLoc

####################################
# argument use in order to identify picture in command line
# Open with "python /path/to/script.py --image /path/to/picture.jpg"
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

img = cv2.imread(args["image"])
clone=img.copy()
img = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("window")
cv2.setMouseCallback("window", click)
cv2.imshow("window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

####################################
# This is a way to find the brightest area near the pixel you clicked
# Take the distance to background in either direction, search grid for brightest point
# sum all the values of the square area selected
####################################

#print(ReferenceStarLoc, ReferenceBackgroundLoc)
ReferenceDistance = (np.absolute(np.subtract(ReferenceStarLoc, ReferenceBackgroundLoc)))
ReferenceRadius = np.amax(ReferenceDistance)
print('The radius of the reference star is', ReferenceRadius)

ReferenceRange = np.arange(-ReferenceRadius,ReferenceRadius+1,1)
XReferenceRange = ReferenceStarLoc[0] + ReferenceRange
YReferenceRange = ReferenceStarLoc[1] + ReferenceRange

FullReferenceRange=[]
for i in range(0,2*ReferenceRadius+1):
    ColumnShape = []
    ColumnValues = ((np.full((2*ReferenceRadius+1), YReferenceRange[i])))
    ColumnShape  = np.column_stack((XReferenceRange,ColumnValues))
    FullReferenceRange.append(ColumnShape)

FullReferenceRange = np.array(np.concatenate((np.array(FullReferenceRange))))
#FullestRange looks gross, but this was the only way to get it in one long 121 by 2 matrix
#But yes, it is gross

ReferenceMagValues = []
for i in range(0,ReferenceRadius**2):
    xVal = FullReferenceRange[i,0]
    yVal = FullReferenceRange[i,1]
    ReferencePixelValue=img[xVal,yVal]
    ReferenceMagValues.append(ReferencePixelValue)

ReferenceMagValues = (np.array(ReferenceMagValues)).astype(np.int16)
ReferenceMagValues = (ReferenceMagValues - ReferenceBackgroundMag)
ReferenceMagValues = ReferenceMagValues.clip(0)
ReferenceMagValue = np.sum(ReferenceMagValues)
print('The magnitude value of the reference star is',ReferenceMagValue)

####################################
#Same thing but for Object
####################################

ObjectDistance = (np.absolute(np.subtract(ObjectLoc, ObjectBackgroundLoc)))
ObjectRadius = np.amax(ObjectDistance)
print('The radius of the object is',ObjectRadius)

ObjectRange = np.arange(-ObjectRadius,ObjectRadius+1,1)
XObjectRange = ObjectLoc[0] + ObjectRange
YObjectRange = ObjectLoc[1] + ObjectRange
FullObjectRange=[]
for i in range(0,2*ObjectRadius+1):
    ColumnShape = []
    ColumnValues = ((np.full((2*ObjectRadius+1), YObjectRange[i])))
    ColumnShape  = np.column_stack((XObjectRange,ColumnValues))
    FullObjectRange.append(ColumnShape)

FullObjectRange = np.array(np.concatenate((np.array(FullObjectRange))))

ObjectMagValues = []
for i in range(0,ObjectRadius**2):
    xVal = FullObjectRange[i,0]
    yVal = FullObjectRange[i,1]
    ObjectPixelValue=img[xVal,yVal]
    ObjectMagValues.append(ObjectPixelValue)
ObjectMagValues = (np.array(ObjectMagValues)).astype(np.int16)
ObjectMagValues = (ObjectMagValues - ObjectBackgroundMag)
ObjectMagValues = ObjectMagValues.clip(0)
ObjectMagValue = np.sum(ObjectMagValues)
print('The Magnitude Value of the object is',ObjectMagValue)

####################################
#Photometry for finding the catalog value
####################################
Exposure = 1

InstrumentalMagnitude = -2.5*sp.log(ReferenceMagValue/Exposure)
CatalogMagnitude = float(input('Enter catalog magnitude: '))
Offset = InstrumentalMagnitude - CatalogMagnitude 

ObjectCatalogValue = -2.5*sp.log(ObjectMagValue/Exposure) - Offset
print('The catalog value of the object is *maybe*',  ObjectCatalogValue)

