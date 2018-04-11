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


REFERENCESTARLOC = (0,0)
OBJECTLOC = (0,0)
OBJECTAVGRADIUS = 0
OBJECTBACKGROUNDRADIUS = 0
REFERENCESTARAVGRADIUS = 0
REFERENCEBACKGROUNDRADIUS = 0
XFitParametersRef = []
YFitParametersRef = []
XFitParametersObj = []
YFitParametersObj = []

matplotlib.rcParams['figure.figsize']=(10,10)

##################################################
# Creating function for mouse click
# Left Click is for Reference Star
# Right Click is for the Object
##################################################
def click(event, x, y, flags, param):
    global REFERENCESTARLOC
    global OBJECTLOC
    if event == cv2.EVENT_LBUTTONDOWN:
        REFERENCESTARLOC = (x,y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        OBJECTLOC = (x,y)

def Threshold(theimage, ThresholdRatio):
    threshold = cv2.threshold(theimage, ThresholdRatio, 255, cv2.THRESH_BINARY)[1]
    return(threshold)

def MaxFinder(X, Y, img):
    r = 10
    # Accessing array so Y values first
    subimg = img[Y-r:Y+r, X-r:X+r]
    blur = cv2.GaussianBlur(subimg, (5,5), 0)
    thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)[1]
    thresh = Threshold(blur, THRESHOLDNUMBER)
    im2, contours, heir = cv2.findContours(thresh, cv2.RETR_TREE,
	cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    max_idx = np.argmax(areas)
    cnt = contours[max_idx]

    M = cv2.moments(cnt)
    M["m00"] != 0
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    MaxLocShifted = (X-r+cX, Y-r+cY)
    # else:
        # MaxLocShifted = (X-r,Y-r)
    return MaxLocShifted 

def Gaussian(x, a, x0, Sigma):
    return a*np.exp(-(x-x0)**2/(2*Sigma**2))

def GaussianFinder(MaxLoc, img):
    CoordinatesX = []
    CoordinatesY = []
    Range = np.arange(MaxLoc-10,MaxLoc+10)

    if MaxLoc == REFERENCESTARLOC[0]:
        Coordinates = img[REFERENCESTARLOC[1], MaxLoc-10:MaxLoc+10]
    if MaxLoc == REFERENCESTARLOC[1]:
        Coordinates = img[MaxLoc-10:MaxLoc+10, REFERENCESTARLOC[0]]
    if MaxLoc == OBJECTLOC[0]:
        Coordinates = img[OBJECTLOC[1], MaxLoc-10:MaxLoc+10]
    if MaxLoc == OBJECTLOC[1]:
        Coordinates = img[MaxLoc-10:MaxLoc+10, OBJECTLOC[0]]

    Mean = MaxLoc
    Sigma = 3
    try:
        FitParameters, pcov = curve_fit(Gaussian, Range, Coordinates, 
                p0 = [np.max(Coordinates), Mean, Sigma])
    except RuntimeError: 
        FitParameters = [np.max(Coordinates),Mean, Sigma]

    return(FitParameters)

def MagnitudeFinder(Loc, XFitParameters, YFitParameters, img):
    global Radius, OBJECTBACKGROUNDRADIUS, REFERENCEBACKGROUNDRADIUS, OBJECTAVGRADIUS, REFERENCESTARAVGRADIUS
    YRadius = int(np.ceil(2*XFitParameters[2])) 
    XRadius = int(np.ceil(2*YFitParameters[2]))
    Radius = max(XRadius,YRadius)

    Range=[]
    if Radius > 100:
        Radius = 100
    for i in range(-Radius,Radius):
        for j in range(-Radius,Radius):
            if i**2 + j**2  <  Radius**2:
                Range.append((i + Loc[0], j + Loc[1])[::-1])

    BackgroundRadius = Radius+5
    BackgroundRange=[]
    for i in range(-BackgroundRadius,BackgroundRadius):
        for j in range(-BackgroundRadius,BackgroundRadius):
            if i**2 + j**2  <  BackgroundRadius**2 and \
                i**2 + j**2 > Radius**2:
                BackgroundRange.append((i + Loc[0], j + Loc[1])[::-1])

    # BackgroundRange = np.array(BackgroundRange)
    # for i in BackgroundRange[:,0]:
        # if i > 479:
            # BackgroundRange[:,0] = 479
    # print(BackgroundRange)
    # BackgroundRange.tolist()
    # #print(BackgroundRange)

    BackgroundValues = []
    for i in BackgroundRange:
        BackgroundValues.append(img[i])
    AvgBackgroundMag = sum(BackgroundValues)/len(BackgroundValues)
    MagValue = 0
    RawMagValue= 0
    for i in Range:
        MagValue = MagValue + (img[i] - AvgBackgroundMag)
        RawMagValue = RawMagValue + img[i]
    if Loc == REFERENCESTARLOC:
        REFERENCESTARAVGRADIUS = Radius
        REFERENCEBACKGROUNDRADIUS = Radius+5
    if Loc == OBJECTLOC:
        OBJECTAVGRADIUS = Radius
        OBJECTBACKGROUNDRADIUS = Radius+5
    return(MagValue)

def PlottingCurve(XFitParameters, YFitParameters, Radius, img,Folder):

    def getPlotSlice(XCent, YCent, Dir, PlotRange):
        """Function to return a single column or row sliced at 
        the desired location
        and in the desired direction.
        XCent and YCent are integers and Dir is a stringof either "x" or "y"
        """
        # Gaussian positions are not integers so converting to integers
        # Extracting desired slice
        if Dir == 'x':
            return img[YCent,XCent-PlotRange:XCent+PlotRange]
        else:
            return img[YCent-PlotRange:YCent+PlotRange, XCent]

    XMaxLoc = int(XFitParameters[1])
    YMaxLoc = int(YFitParameters[1])

    # sns.set()
    # sns.set_style("dark")
    sns.set_context("poster")
    matplotlib.style.use("dark_background")
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2,1])
    ax = plt.subplot(gs[0, 0]) 

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
    
    PlotRange = 20

    # #Top Left
    # ax1.imshow(img,cmap='gray')
    # ax1.set_xlim(-PlotRange+OBJECTLOC[0],PlotRange+OBJECTLOC[0])
    # ax1.set_ylim(-PlotRange+OBJECTLOC[1],PlotRange+OBJECTLOC[1])
    # if XMaxLoc == int(XFitParametersObj[1]):
        # #ax1.set_title('Magnitude of %s'%(round(ObjectCatalogValue,3)))
        # ax1.add_artist(plt.Circle((OBJECTLOC),OBJECTBACKGROUNDRADIUS, 
            # color = 'yellow', alpha=.2))
        # ax1.add_artist(plt.Circle((OBJECTLOC),Radius,color='red', alpha=0.2))
        # ax1.add_artist(plt.Circle((OBJECTLOC),.1, color = 'black',))
    # # if XMaxLoc == int(XFitParametersRef[1]):
        # # ax1.set_title('Magnitude of %s' %(CatalogMagnitude))
        # # ax1.add_artist(plt.Circle((REFERENCESTARLOC),Radius,
            # # color='blue', alpha=0.2))
        # # ax1.add_artist(plt.Circle((REFERENCESTARLOC),REFERENCEBACKGROUNDRADIUS, 
            # # color = 'yellow', alpha=.2))
    # ax1.axis('off')
   
    ax1.imshow(img,cmap='gray')
    if XMaxLoc == int(XFitParametersObj[1]):
        ax1.set_xlim(-PlotRange+OBJECTLOC[0],PlotRange+OBJECTLOC[0])
        ax1.set_ylim(-PlotRange+OBJECTLOC[1],PlotRange+OBJECTLOC[1])
        # ax1.set_title('Magnitude of %s'%(round(ObjectCatalogValue,3)))
        ax1.add_artist(plt.Circle((OBJECTLOC),OBJECTBACKGROUNDRADIUS, 
            color = 'yellow', alpha=.2))
        ax1.add_artist(plt.Circle((OBJECTLOC),OBJECTAVGRADIUS,color='red',alpha=.2))
        # ax1.add_artist(plt.Circle((OBJECTLOC),.1, color = 'black',))
    if XMaxLoc == int(XFitParametersRef[1]):
        ax1.set_xlim(-PlotRange+REFERENCESTARLOC[0],PlotRange+REFERENCESTARLOC[0])
        ax1.set_ylim(-PlotRange+REFERENCESTARLOC[1],PlotRange+REFERENCESTARLOC[1])
        # ax1.set_title('Magnitude of %s' %(CatalogMagnitude))
        ax1.add_artist(plt.Circle((REFERENCESTARLOC),5,
            color='blue', alpha=0.2))
        ax1.add_artist(plt.Circle((REFERENCESTARLOC),REFERENCEBACKGROUNDRADIUS, 
            color = 'yellow', alpha=.2))
    ax1.axis('off')

    #Top Right
    if YMaxLoc == int(YFitParametersObj[1]):
        ax2.plot(getPlotSlice(OBJECTLOC[0],OBJECTLOC[1],'y',PlotRange),
                np.arange(-PlotRange+OBJECTLOC[1],PlotRange+OBJECTLOC[1],1),
                label='Data',color='orange')
        ax2.plot((Gaussian(np.arange(-PlotRange+OBJECTLOC[1],
            PlotRange+OBJECTLOC[1],1),*YFitParameters)),
                np.arange(-PlotRange+OBJECTLOC[1],PlotRange+OBJECTLOC[1],1),
                label='Gaussian Fit',color='red')
        ax2.set_xlabel('Pixel Value')
    # if YMaxLoc == int(YFitParametersRef[1]):
        # ax2.plot(getPlotSlice(REFERENCESTARLOC[0],REFERENCESTARLOC[1],
            # 'y', PlotRange),
                # np.arange(-PlotRange+REFERENCESTARLOC[1],
                    # PlotRange+REFERENCESTARLOC[1],1),
                # label='Data')
        # ax2.plot((Gaussian(np.arange(-PlotRange+REFERENCESTARLOC[1],
            # PlotRange+REFERENCESTARLOC[1],1),*YFitParameters)),
                # np.arange(-PlotRange+REFERENCESTARLOC[1],
                    # PlotRange+REFERENCESTARLOC[1],1),
                # label='Gaussian Fit')
    ax2.yaxis.set_visible(False)
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top')

    #Bottom Left
    if XMaxLoc == int(XFitParametersObj[1]):
        ax3.plot(np.arange(-PlotRange+OBJECTLOC[0],PlotRange+OBJECTLOC[0]),
                getPlotSlice(OBJECTLOC[0],OBJECTLOC[1],'x',PlotRange),
                label='Data',color='orange')
        ax3.plot(np.arange(-PlotRange+OBJECTLOC[0],PlotRange+OBJECTLOC[0]),
                Gaussian(np.arange(-PlotRange+OBJECTLOC[0],
                    PlotRange+OBJECTLOC[0]),*XFitParameters),
                label='Gaussian Fit',color='red')
        ax3.set_ylabel('Pixel Value')
    # if XMaxLoc == int(XFitParametersRef[1]):
        # YMaxLoc = REFERENCESTARLOC[1]
        # XMaxLoc = REFERENCESTARLOC[0]
        # ax3.plot(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),
                # getPlotSlice(XMaxLoc, YMaxLoc, 'x', PlotRange),
                # label='Data')
        # ax3.plot(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc), 
                # Gaussian(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),
                    # *XFitParameters),
                # label='Gaussian Fit')
    ax3.legend(bbox_to_anchor=(0.,-.2, 1., .102), loc=3,ncol=2, borderaxespad=0.)
    ax3.invert_yaxis()
    ax3.xaxis.set_visible(False)
    
    #Bottom Right
    ax4.imshow(img,cmap='gray')
    if XMaxLoc == int(XFitParametersObj[1]):
        ax4.axvline(OBJECTLOC[0], color='red')
        ax4.axhline(OBJECTLOC[1], color='red')
    # if XMaxLoc == int(XFitParametersRef[1]):
        # ax4.axvline(REFERENCESTARLOC[0], color='blue')
        # ax4.axhline(REFERENCESTARLOC[1], color='blue')
    ax4.axis('off')


    plt.subplots_adjust(wspace=.1)
    plt.subplots_adjust(hspace=.1)
    plt.draw()

    #Title
    # if XMaxLoc == int(XFitParametersObj[1]):
        # plt.suptitle('Object')
    # if XMaxLoc == int(XFitParametersRef[1]):
        # plt.suptitle('Reference Star')
    if XFitParameters[0] == XFitParametersObj[0]:
        plt.savefig(f'Data/{Folder}/ObjectPlot{Iteration:03d}.png', bbox_inches='tight')
    if XFitParameters[0] == XFitParametersRef[0]:
        plt.savefig(f'Data/{Folder}/ReferencePlot.png', bbox_inches='tight')

##################################################
# Instructions for user
##################################################
def InitialRead(VideoName):
    Array = np.array([],ndmin=3)
    Video = cv2.VideoCapture(VideoName)
    Grabbed, Image = Video.read()
    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    ImageStack = Image
    (Grabbed,Image) = Video.read()
    while Grabbed == True:
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        ImageStack = np.dstack([ImageStack,Image])
        print(np.shape(ImageStack))
        (Grabbed,Image) = Video.read()
    return(ImageStack)



def main(vid_name,Folder,StartFrame, objectlocation, referencestarlocation, ThresholdNumber):
    global REFERENCESTARLOC
    global OBJECTLOC
    global REFERENCESTARAVGRADIUS
    global OBJECTAVGRADIUS
    global OBJECTBACKGROUNDRADIUS
    global REFERENCEBACKGROUNDRADIUS
    global THRESHOLDNUMBER
    global XFitParametersRef
    global YFitParametersRef
    global XFitParametersObj
    global YFitParametersObj
    global CatalogMagnitude
    global Iteration
    global ObjectMagValue 

    record = True
    frame_no = 100
        
    REFERENCESTARLOC = referencestarlocation
    OBJECTLOC = objectlocation
    THRESHOLDNUMBER = ThresholdNumber

    vid = cv2.VideoCapture(vid_name)
    vid.set(1,StartFrame); # Where frame_no is the frame you want
    Grabbed, img = vid.read() # Read the frame
    #(Grabbed,img) = vid.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    Thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]


    # img2 = img.copy()
    # cv2.namedWindow("window")
    # cv2.setMouseCallback("window", click)
    # cv2.imshow("window", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    REFERENCESTARLOC = MaxFinder(REFERENCESTARLOC[0], REFERENCESTARLOC[1], img)
    XFitParametersRef = GaussianFinder(REFERENCESTARLOC[0], img)
    YFitParametersRef = GaussianFinder(REFERENCESTARLOC[1], img)
    ReferenceMagValue = MagnitudeFinder(REFERENCESTARLOC, 
            GaussianFinder(REFERENCESTARLOC[0], img), 
            GaussianFinder(REFERENCESTARLOC[1], img), img)
    XFitParameters = (0,0,0)

    InstrumentalMagnitude = -2.5*np.log10(ReferenceMagValue)
    CatalogMagnitude = -2.65
    Offset = InstrumentalMagnitude - CatalogMagnitude 
    #CatalogMagnitude = float(input('Enter catalog magnitude: '))
    ####TEMP####
    Offset = +12


    ObjectMagValueList = []
    CatalogList = []
    Instrument = []
    Iteration = 1
    Round = 0
    while Grabbed == True:
        
        (Grabbed, img) = vid.read()
        if Grabbed == False:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        OBJECTLOC = MaxFinder(OBJECTLOC[0], OBJECTLOC[1], img) 
        XFitParametersObj = GaussianFinder(OBJECTLOC[0], img)
        YFitParametersObj = GaussianFinder(OBJECTLOC[1], img)
        ObjectMagValue=MagnitudeFinder(OBJECTLOC,XFitParametersObj,
                YFitParametersObj, img)
        ObjectMagValueList.append(ObjectMagValue)


        ObjectCatalogValue = -2.5*np.log10(ObjectMagValue) - Offset
        CatalogList.append(ObjectCatalogValue)
        if Iteration > 1:
            # plt.figure()
            # plt.plot(np.arange(0,len(ObjectMagValueList)),ObjectMagValueList)
            # plt.xlabel('Frame')
            # plt.ylabel('Pixel Value')
            # plt.yticks([])
            # plt.savefig(f"/home/luke/{Folder}/LightCurvePixel.png",
            # bbox_inches='tight')

            plt.figure()
            plt.plot(np.arange(0,len(CatalogList)),CatalogList)
            plt.xlabel('Frame')
            plt.ylabel('Magnitude')
            plt.gca().invert_yaxis()
            plt.savefig(f"Data/{Folder}/LightCurve.png",
            bbox_inches='tight')


        PlottingCurve(XFitParametersObj, YFitParametersObj, OBJECTAVGRADIUS, 
                img,Folder)
        if Iteration == 1:
            PlottingCurve(XFitParametersRef, YFitParametersRef, REFERENCESTARAVGRADIUS, img,Folder)
        Iteration += 1
    plt.figure()
    # plt.plot(np.arange(0,len(ObjectMagValueList)),ObjectMagValueList)
    plt.plot(np.arange(0,len(CatalogList)),CatalogList)
    plt.xlabel('Frame')
    plt.ylabel('Pixel Values')
    # plt.yticks([])
    plt.gca().invert_yaxis()
    plt.savefig(f"Data/{Folder}/LightCurve.png", bbox_inches='tight')
##################################################
# open with "python /path/to/script.py --image /path/to/picture.jpg"
##################################################
if __name__ == "__main__":
    print('Instructions')
    print('Reference Star = left click ||| Object = right click')
    print('Click on item. It does not need to be exact. Continue to hold and drag cursor to area of empty space next to item and release.')
    print('If circle does not satisfy what you were trying to do, feel free to repeat the previous instruction.')
    ap=argparse.ArgumentParser()
    ap.add_argument("-i", "--video", required=True)
    args= vars(ap.parse_args())
    main(args['video'])
