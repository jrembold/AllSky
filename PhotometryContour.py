##################################################
#
#
# Identifying Objects & Their Magnitudes
# 
# Luke Galbraith Russell
#
# Willamette University
#
#
##################################################

import seaborn as sns
import numpy as np
import cv2
import argparse
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
import matplotlib

matplotlib.rcParams['figure.figsize']=(10,10)

##################################################
# Creating function for mouse click
# Left Click is for Reference Star
# Right Click is for the Object
##################################################

def click(event, x, y, flags, param):
    global REFERENCESTARLOC
    global REFERENCESTARX
    global REFERENCESTARY
    global OBJECTLOC
    global OBJECTX
    global OBJECTY

    if event == cv2.EVENT_LBUTTONDOWN:
        REFERENCESTARLOC = (x,y)
        REFERENCESTARX = x
        REFERENCESTARY = y

    elif event == cv2.EVENT_RBUTTONDOWN:
        OBJECTLOC = (x,y)
        OBJECTX = x
        OBJECTY = y
    
##################################################
# Instructions for user
##################################################

print('Instructions')
print('Reference Star = left click ||| Object = right click')
print('Click on item. It does not need to be exact. Continue to hold and drag cursor to area of empty space next to item and release.')
print('If circle does not satisfy what you were trying to do, feel free to repeat the previous instruction.')

##################################################
# argument use in order to identify picture in command line
# open with "python /path/to/script.py --image /path/to/picture.jpg"
# only works with JPG as of now
##################################################

ap=argparse.ArgumentParser()
ap.add_argument("-i", "--video", required=True)
args= vars(ap.parse_args())

##################################################
# reading in the jpg, cloning a copy, and then converting that to grayscale
# creating window for it to pop up in
# activating mouse function from above
# and key press on the picture will exit it out
##################################################
record = True
frame_no = -1

vid = cv2.VideoCapture(args["video"])
(Grabbed,img) = vid.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(img, (5, 5), 0)
Thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]


img2 = img.copy()
cv2.namedWindow("window")
cv2.setMouseCallback("window", click)
cv2.imshow("window", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


def FindingInitialMax(X,Y):
    global cX,cY
    # Radius of search
    r = 10
    # Accessing array so Y values first
    thresh = Thresh[Y-r:Y+r,X-r:X+r]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    M = cv2.moments(cnts)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    MaxLocShifted = (X-r+cX, Y-r+cY)
    return MaxLocShifted 
    
XMAXLOCREF,YMAXLOCREF = FindingInitialMax(REFERENCESTARX, REFERENCESTARY)
InitialXMaxLocObj,InitialYMaxLocObj = FindingInitialMax(OBJECTX, OBJECTY)

REFERENCESTARLOC = (XMAXLOCREF, YMAXLOCREF)
#OBJECTLOC = (XMaxLocObj, YMaxLocObj)


X = InitialXMaxLocObj
Y = InitialYMaxLocObj


ObjectMagValueList = []

##################################################
# Finding Max
#
# This function searches through a range of values
# along each axis from the point you clicked
# and it finds the max for the column and for the row
# and thus centering any future data at the center 
# as the center is the brightest
##################################################

Iteration = 100
while Grabbed == True:

    (Grabbed, img) = vid.read()
    if Grabbed == False:
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def FindingMax(X,Y):
        global MaxLocShifted
        # Radius of search
        r = 10
        # Accessing array so Y values first
        thresh = Thresh[Y-r:Y+r,X-r:X+r]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        M = cv2.moments(cnts)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        MaxLocShifted = (X-r+cX, Y-r+cY)
        return MaxLocShifted

    ##################################################
    # In order to center the x and y of both items
    # we call the FindingMax function for them
    # And this center we will use in the FindingGaussian function
    ##################################################

    XMaxLocObj,YMaxLocObj = FindingMax(OBJECTX, OBJECTY)
    OBJECTLOC = (XMaxLocObj, YMaxLocObj)

    #XList.append(X)
    #YList.append(Y)

    X = XMaxLocObj
    Y = YMaxLocObj
    ##################################################
    # Finding Gaussian
    #
    # Starting at the previously identified center,
    # We find the pixel values of a range around that center
    # and create a list of pixel values with the index values
    # We then say the max is the ampliteude, 
    # and guess the standard deviation
    # And we fit the data to the Gaussian function. 
    ##################################################

    def Gaussian(x, a, x0, Sigma):
        return a*np.exp(-(x-x0)**2/(2*Sigma**2))


    def FindingGaussian(MaxLoc):
        global OBJECTXPIXELS
        global OBJECTYPIXELS
        global REFERENCEXPIXELS
        global REFERENCEYPIXELS

        Coordinates = []
        Range = np.arange(MaxLoc-10,MaxLoc+10)

        if MaxLoc == XMAXLOCREF:
            Coordinates = img[YMAXLOCREF, MaxLoc-10:MaxLoc+10]
        if MaxLoc == YMAXLOCREF:
            Coordinates = img[MaxLoc-10:MaxLoc+10, XMAXLOCREF]
        if MaxLoc == XMaxLocObj:
            Coordinates = img[YMaxLocObj, MaxLoc-10:MaxLoc+10]
        if MaxLoc == YMaxLocObj:
            Coordinates = img[MaxLoc-10:MaxLoc+10, XMaxLocObj]

        Mean = MaxLoc
        Sigma = 3
        try:
            FitParameters, pcov = curve_fit(Gaussian, Range, Coordinates, p0 = [np.max(Coordinates), Mean, Sigma])
        except RuntimeError: 
            FitParameters = [np.max(Coordinates),Mean, Sigma]

        return(FitParameters)
    ##################################################
    # FitParameters contains the amplitude, standard deviation and mean
    # The standard deviation is needed later so we call this function 
    # for both axes for both items
    ##################################################

    XFitParametersRef = FindingGaussian(XMAXLOCREF)
    YFitParametersRef = FindingGaussian(YMAXLOCREF)
    XFitParametersObj = FindingGaussian(XMaxLocObj)
    YFitParametersObj = FindingGaussian(YMaxLocObj)

    def FindingDataMax(NewParameters):
        Coordinates = []

        Search = np.arange(-10,10)

        if NewParameters == XFitParametersRef:
            for i in Search:
                Coordinates.append(img[YFitParametersRef[1],XFitParametersRef[1]+i])
            REFERENCEXPIXELS = np.array(Coordinates)
        

        if NewParameters == YFitParametersRef:
            for i in Search:
                Coordinates.append(img[YFitParametersRef[1]+i,XFitParametersRef[1]])
            REFERENCEYPIXELS = np.array(Coordinates)

        if NewParameters == XFitParametersObj:
            for i in Search:
                Coordinates.append(img[YFitParametersObj[1],XFitParametersObj[1]+i])
            OBJECTXPIXELS = np.array(Coordinates)
        

        if NewParameters == YFitParametersObj:
            for i in Search:
                Coordinates.append(img[YFitParametersObj[1]+i,XFitParametersObj[1]])
            OBJECTYPIXELS = np.array(Coordinates)


    ##################################################
    # Magnitude Finder
    # 
    ##################################################

    def MagnitudeFinder(Loc, XFitParameters, YFitParameters):
        global REFERENCESTARAVGRADIUS
        global OBJECTBACKAVGRADIUS
        global OBJECTBACKGROUNDRADIUS
        global REFERENCEBACKGROUNDRADIUS

        YRadius = int(np.ceil(2*XFitParameters[2])) #This rounds up to the nearest integer
        XRadius = int(np.ceil(2*YFitParameters[2]))
        Radius = max(XRadius,YRadius)

        Range=[]
        if Radius > 100:
            Radius = 100
        for i in range(-Radius,Radius):
            for j in range(-Radius,Radius):
                if i**2 + j**2  <  Radius**2:
                    Range.append((i + Loc[0], j + Loc[1])[::-1])

        #print(Radius)
        BackgroundRadius = Radius+5
        BackgroundRange=[]
        for i in range(-BackgroundRadius,BackgroundRadius):
            for j in range(-BackgroundRadius,BackgroundRadius):
                if i**2 + j**2  <  BackgroundRadius**2 and \
                    i**2 + j**2 > Radius**2:
                    BackgroundRange.append((i + Loc[0], j + Loc[1]))#[::-1])

        BackgroundValues = []
        for i in BackgroundRange:
            BackgroundValues.append(img[i])
        AvgBackgroundMag = sum(BackgroundValues)/len(BackgroundValues)
        #print(AvgBackgroundMag)
        MagValue = 0
        RawMagValue= 0
        for i in Range:
            #print(X,Y)
            #print(img[i])
            MagValue = MagValue + (img[i] - AvgBackgroundMag)
            RawMagValue = RawMagValue + img[i]
            #print('VALUES',MagValue, RawMagValue)
        if Loc == REFERENCESTARLOC:
            #print('The average radius of the reference star is ',(XRadius**2 + YRadius**2)**.5, 'and the magnitude is', MagValue)
            REFERENCESTARAVGRADIUS = Radius
            REFERENCEBACKGROUNDRADIUS = Radius+5
        if Loc == OBJECTLOC:
            #print('The average radius of the object is', (XRadius**2 + YRadius**2)**.5, 'and the magnitude is', MagValue)
            OBJECTBACKAVGRADIUS = Radius
            OBJECTBACKGROUNDRADIUS = Radius+5
        
        return(MagValue)

    ReferenceMagValue = MagnitudeFinder(REFERENCESTARLOC, XFitParametersRef, YFitParametersRef)

    ObjectMagValue = MagnitudeFinder(OBJECTLOC, XFitParametersObj, YFitParametersObj)
    ObjectMagValueList.append(ObjectMagValue)

    ##################################################
    #Photometry for finding the catalog value
    ##################################################

    InstrumentalMagnitude = -2.5*np.log10(ReferenceMagValue)
    CatalogMagnitude = 10
    Offset = InstrumentalMagnitude - CatalogMagnitude 
    #CatalogMagnitude = float(input('Enter catalog magnitude: '))

    ObjectCatalogValue = -2.5*np.log10(ObjectMagValue) - Offset
    #print('The catalog value of the object is *maybe*',  ObjectCatalogValue)

    ##################################################
    # Subplots
    ##################################################

    def PlottingCurve(XMaxLoc, YMaxLoc, XFitParameters, YFitParameters, Radius):
        print('Ahhhh!')
        sns.set()
        sns.set_style("dark")
        sns.set_context("poster") 
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2,1])
        ax = plt.subplot(gs[0, 0]) 

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
        ax4 = plt.subplot(gs[3])
        

        PlotRange = 20

        def getPlotSlice(XCent, YCent, Dir, PlotRange):
            '''Function to return a single column or row sliced at the desired location
            and in the desired direction. XCent and YCent are integers and Dir is a string
            of either "x" or "y"

            I realized later you may have already made something similar in FindingDataMax?
            '''
            # Gaussian positions are not integers so converting to integers
            XCent = int(XCent)
            YCent = int(YCent)
            # Extracting desired slice
            if Dir == 'x':
                return img[YCent,XCent-PlotRange:XCent+PlotRange]
            else:
                return img[YCent-PlotRange:YCent+PlotRange, XCent]

        #Top Left
        inverseimg = cv2.bitwise_not(img)
        ax1.imshow(img,cmap='gray')
        ax1.set_xlim(-PlotRange+XMaxLoc,PlotRange+XMaxLoc)
        ax1.set_ylim(-PlotRange+YMaxLoc,PlotRange+YMaxLoc)
        if XMaxLoc == XFitParametersObj[1]:
            ax1.set_title('Magnitude of %s'%(round(ObjectCatalogValue,3)))
            ax1.add_artist(plt.Circle((XMaxLoc,YMaxLoc),OBJECTBACKGROUNDRADIUS, color = 'yellow', alpha=.2))
            ax1.add_artist(plt.Circle((XMaxLoc,YMaxLoc),Radius,color='red', alpha=0.2))
            ax1.add_artist(plt.Circle((MaxLocShifted),2, color = 'white',))
        if XMaxLoc == XFitParametersRef[1]:
            ax1.set_title('Magnitude of %s' %(CatalogMagnitude))
            ax1.add_artist(plt.Circle((XMaxLoc,YMaxLoc),Radius,color='blue', alpha=0.2))
            ax1.add_artist(plt.Circle((XMaxLoc,YMaxLoc),REFERENCEBACKGROUNDRADIUS, color = 'yellow', alpha=.2))
        # ax1.grid()
        ax1.axis('off')
        
        #Top Right
        if YMaxLoc == YFitParametersObj[1]:
            ax2.plot(getPlotSlice(XMaxLoc,YMaxLoc,'y',PlotRange),
                    np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),
                    label='Data',color='orange')
            ax2.plot((Gaussian(np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),*YFitParameters)),
                    np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),
                    label='Gaussian Fit',color='red')
        if YMaxLoc == YFitParametersRef[1]:
            ax2.plot(getPlotSlice(XMaxLoc,YMaxLoc,'y',PlotRange),
                    np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),
                    label='Data')
            ax2.plot((Gaussian(np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),*YFitParameters)),
                    np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),
                    label='Gaussian Fit')
        #ax2.legend(loc="lower right")
        ax2.yaxis.set_visible(False)
       # ax2.set_xlabel('Pixel Values')
       # ax2.xaxis.set_label_position('top')
       # ax2.xaxis.set_label_coords(.34,1.11)
        ax2.xaxis.tick_top()
        
        #Bottom Left
        if XMaxLoc == XFitParametersObj[1]:
            ax3.plot(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),
                    getPlotSlice(XMaxLoc, YMaxLoc,'x',PlotRange),
                    label='Data',color='orange')
            ax3.plot(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),
                    Gaussian(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),*XFitParameters),
                    label='Gaussian Fit',color='red')
        if XMaxLoc == XFitParametersRef[1]:
            ax3.plot(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),
                    getPlotSlice(XMaxLoc, YMaxLoc, 'x', PlotRange),
                    label='Data')
            ax3.plot(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc), 
                    Gaussian(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),*XFitParameters),
                    label='Gaussian Fit')
        ax3.legend(bbox_to_anchor=(0., -.2, 1., .102), loc=3,ncol=2, borderaxespad=0.)
       # ax3.set_ylabel('Pixel Values')
       # ax3.yaxis.set_label_coords(-0.11,.34)
        ax3.invert_yaxis()
        ax3.xaxis.set_visible(False)
        # plt.xticks([])
        
        #Bottom Right
        ax4.imshow(img*20,cmap='gray')
        if XMaxLoc == XFitParametersObj[1]:
            ax4.axvline(OBJECTX, color='red')
            ax4.axhline(OBJECTY, color='red')
        if XMaxLoc == XFitParametersRef[1]:
            ax4.axvline(REFERENCESTARX, color='blue')
            ax4.axhline(REFERENCESTARY, color='blue')
        ax4.axis('off')

        plt.subplots_adjust(wspace=.1)
        plt.subplots_adjust(hspace=.1)
        plt.draw()

        #Title
        if XMaxLoc == XFitParametersObj[1]:
            plt.suptitle('Object')
        if XMaxLoc == XFitParametersRef[1]:
            plt.suptitle('Reference Star')

        plt.savefig(f'testplot{Iteration}.pdf')

    PlottingCurve(XFitParametersObj[1], YFitParametersObj[1], XFitParametersObj, YFitParametersObj, OBJECTBACKAVGRADIUS)
    Iteration += 1

plt.plot(np.arange(0,len(ObjectMagValueList)),ObjectMagValueList)
plt.show()
