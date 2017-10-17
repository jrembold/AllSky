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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
####################################
# Creating function for mouse click
# Left Click is for Reference Star
# Right Click is for the Object
####################################

def click(event, x, y, flags, param):
    global ReferenceStarLoc
    global ReferenceStarX
    global ReferenceStarY
    global ReferenceBackgroundMag
    global ReferenceBackgroundLoc
    global ObjectBackgroundMag
    global ObjectBackgroundLoc
    global ObjectLoc
    global ObjectX
    global ObjectY

    if event == cv2.EVENT_LBUTTONDOWN:
        ReferenceStarLoc = (x,y)
        ReferenceStarX = x
        ReferenceStarY = y

    elif event == cv2.EVENT_LBUTTONUP:
        ReferenceBackgroundMag = img[y,x]
        ReferenceBackgroundLoc = (x,y)
        cv2.circle(img2, ReferenceStarLoc, int(math.sqrt((ReferenceStarLoc[0] - ReferenceBackgroundLoc[0])**2 + (ReferenceStarLoc[1] - ReferenceBackgroundLoc[1])**2)),100,1)
        cv2.imshow("window", img2*20)

    elif event == cv2.EVENT_RBUTTONDOWN:
        ObjectLoc = (x,y)
        ObjectX = x
        ObjectY = y
    
    elif event == cv2.EVENT_RBUTTONUP:
        ObjectBackgroundMag = img[y,x]
        ObjectBackgroundLoc = (x,y) 
        cv2.circle(img2, ObjectLoc, int(math.sqrt((ObjectLoc[0] - ObjectBackgroundLoc[0])**2 + (ObjectLoc[1] - ObjectBackgroundLoc[1])**2)),99,1)
        cv2.imshow("window", img2*20)
print('Instructions')
print('Reference Star = left click ||| Object = right click')
print('Click on item. It does not need to be exact. Continue to hold and drag cursor to area of empty space next to item and release.')
print('If circle does not satisfy what you were trying to do, feel free to repeat the previous instruction.')
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
cv2.imshow("window", img2*20)
cv2.waitKey(0)
cv2.destroyAllWindows()

####################################
# This is a way to find the brightest area near the pixel you clicked
# Take the distance to background in either direction, search grid for brightest point
# sum all the values of the square area selected
####################################
def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def FindingMax(X,Y,choice):
    if choice == 'x':
        SearchRange = np.arange(X-15,X+15)
    if choice == 'y':
        SearchRange = np.arange(Y-15,Y+15)
    Coordinates = []
    for i in np.arange(-15,15):
        if choice == 'x':
            Coordinates.append(img[Y, X+i])
        if choice == 'y':
            Coordinates.append(img[Y+i, X])

    MaxValue = np.argmax(np.array(Coordinates))
    MaxLoc = SearchRange[MaxValue]

    return(MaxLoc)

XMaxLocRef = FindingMax(ReferenceStarX, ReferenceStarY,'x')
YMaxLocRef = FindingMax(ReferenceStarX, ReferenceStarY,'y')
XMaxLocObj = FindingMax(ObjectX, ObjectY, 'x')
YMaxLocObj = FindingMax(ObjectX, ObjectY, 'y')

def FindingGaussian(MaxLoc,choice,selection):
    if selection == 'ReferenceStar':
        YMaxLoc = YMaxLocRef
        XMaxLoc = XMaxLocRef
    if selection == 'Object':
        YMaxLoc = YMaxLocObj
        XMaxLoc = XMaxLocObj
    Range = np.arange(MaxLoc-15,MaxLoc+15)
    Coordinates = []
    for i in np.arange(-15,15):
        if choice == 'x':
            Coordinates.append(img[YMaxLoc, MaxLoc+i]) 
        if choice == 'y':
            Coordinates.append(img[MaxLoc+i, XMaxLoc])
    mean = MaxLoc
    sigma = 3
    popt, pcov = curve_fit(gauss_function, Range, Coordinates, p0 = [np.max(Coordinates), mean, sigma])
    return(popt)

XpoptRef = FindingGaussian(XMaxLocRef, 'x','ReferenceStar')
YpoptRef = FindingGaussian(YMaxLocRef, 'y', 'ReferenceStar')
XpoptObj = FindingGaussian(XMaxLocObj, 'x', 'Object')
YpoptObj = FindingGaussian(YMaxLocObj, 'y', 'Object')
##################################################
# Magnitude Finder
##################################################

def MagnitudeFinder(Loc, BackgroundMag, BackgroundLoc, Xpopt, Ypopt):
    global ReferenceStarAvgRadius
    global ObjectAvgRadius

    Distance = (np.absolute(np.subtract(Loc, BackgroundLoc)))
    YRadius = int(np.ceil(3*Xpopt[2]))
    XRadius = int(np.ceil(3*Ypopt[2]))

    Range=[]
    for i in range(-XRadius,XRadius):
        for j in range(-YRadius,YRadius):
            if i**2 + j**2  <  YRadius**2 and XRadius**2:
                Range.append((i + Loc[0], j + Loc[1])[::-1])

    BackgroundRange=[]
    BackgroundRadius = 3 #arbitrary number
    for i in range(-BackgroundRadius,BackgroundRadius):
        for j in range(-BackgroundRadius,BackgroundRadius):
            if i**2 + j**2  <  BackgroundRadius**2:
                BackgroundRange.append((i + BackgroundLoc[0], j + BackgroundLoc[1])[::-1])

    BackgroundValues = []
    for i in BackgroundRange:
        BackgroundValues.append(img[i])
    AvgBackgroundMag = sum(BackgroundValues)/len(BackgroundValues)

    MagValue = 0
    for i in Range:
        MagValue = MagValue + (img[i] - AvgBackgroundMag)

    
    if Loc == ReferenceStarLoc:
        print('The average radius of the reference star is ',(XRadius**2 + YRadius**2)**.5, 'and the magnitude is', MagValue)
        ReferenceStarAvgRadius = (XRadius**2 + YRadius**2)**.5
    if Loc == ObjectLoc:
        print('The average radius of the object is', (XRadius**2 + YRadius**2)**.5, 'and the magnitude is', MagValue)
        ObjectAvgRadius = (XRadius**2 + YRadius**2)**.5
    
    return(MagValue)

ReferenceMagValue = MagnitudeFinder(ReferenceStarLoc, ReferenceBackgroundMag, ReferenceBackgroundLoc, XpoptRef, YpoptRef)

ObjectMagValue = MagnitudeFinder(ObjectLoc, ObjectBackgroundMag, ObjectBackgroundLoc,XpoptObj, YpoptObj)

####################################
#Photometry for finding the catalog value
####################################
InstrumentalMagnitude = -2.5*np.log10(ReferenceMagValue)
CatalogMagnitude = float(input('Enter catalog magnitude: '))
Offset = InstrumentalMagnitude - CatalogMagnitude 

ObjectCatalogValue = -2.5*np.log10(ObjectMagValue) - Offset
print('The catalog value of the object is *maybe*',  ObjectCatalogValue)



##################################################
# Subplots
##################################################
def PlottingCurve(XMaxLoc, YMaxLoc, Xpopt, Ypopt, Radius):
    subf = img[-15+YMaxLoc:15+YMaxLoc,-15+XMaxLoc:15+XMaxLoc]
    f, axarr = plt.subplots(2,2)
    axarr[0,0].plot(np.arange(-15+XMaxLoc,15+XMaxLoc), np.array(gauss_function(np.arange(-15+XMaxLoc,15+XMaxLoc),*Xpopt)))
    axarr[1,0].imshow(subf,cmap='gray')
    axarr[1,0].add_artist(plt.Circle((15,15),Radius,color='cyan', alpha=0.2))
    axarr[0,1].axis('off')
    axarr[1,1].plot((gauss_function(np.arange(15+YMaxLoc,-15+YMaxLoc,-1),*Ypopt)),np.arange(15+YMaxLoc,-15+YMaxLoc,-1))
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    plt.tight_layout()
    plt.draw()

PlottingCurve(XMaxLocRef, YMaxLocRef, XpoptRef, YpoptRef, ReferenceStarAvgRadius)
PlottingCurve(XMaxLocObj, YMaxLocObj, XpoptObj, YpoptObj, ObjectAvgRadius)

plt.show()
