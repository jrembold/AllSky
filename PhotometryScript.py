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
import csv

##################################################
# Creating function for mouse click
# Left Click is for Reference Star
# Right Click is for the Object
##################################################

def click(event, x, y, flags, param):
    global ReferenceStarLoc
    global ReferenceStarX
    global ReferenceStarY
    global ObjectLoc
    global ObjectX
    global ObjectY

    if event == cv2.EVENT_LBUTTONDOWN:
        ReferenceStarLoc = (x,y)
        ReferenceStarX = x
        ReferenceStarY = y

    elif event == cv2.EVENT_RBUTTONDOWN:
        ObjectLoc = (x,y)
        ObjectX = x
        ObjectY = y
    
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
#image = cv2.imread(args["image"])
record = True
frame_no = -1

vid = cv2.VideoCapture(args["video"])
(grabbed,img) = vid.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#img = cv2.resize(img,(0,0), fx=1.5, fy=1.5)
img2 = img.copy()
cv2.namedWindow("window")
cv2.setMouseCallback("window", click)
cv2.imshow("window", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

##################################################
# Finding Max
#
# This function searches through a range of values
# along each axis from the point you clicked
# and it finds the max for the column and for the row
# and thus centering any future data at the center 
# as the center is the brightest
##################################################

def FindingInitialMax(X,Y,choice):
    if choice == 'x':
        SearchRange = np.arange(X-10,X+10)
    if choice == 'y':
        SearchRange = np.arange(Y-10,Y+10)
    Coordinates = []
    for i in np.arange(-10,10):
        if choice == 'x':
            Coordinates.append(img[Y, X+i])
        if choice == 'y':
            Coordinates.append(img[Y+i, X])

    MaxValue = np.argmax(np.array(Coordinates))
    MaxLoc = SearchRange[MaxValue]

    return(MaxLoc)

    ##################################################
    # In order to center the x and y of both items
    # we call the FindingMax function for/ them
    # And this center we will use in the FindingGaussian function
    ##################################################

XMaxLocRef = FindingInitialMax(ReferenceStarX, ReferenceStarY,'x')
YMaxLocRef = FindingInitialMax(ReferenceStarX, ReferenceStarY,'y')
InitialXMaxLocObj = FindingInitialMax(ObjectX, ObjectY, 'x')
InitialYMaxLocObj = FindingInitialMax(ObjectX, ObjectY, 'y')

ReferenceStarLoc = (XMaxLocRef, YMaxLocRef)
InitialObjectLoc = (InitialXMaxLocObj, InitialYMaxLocObj)

X = InitialXMaxLocObj
Y = InitialYMaxLocObj

XList = []
YList = []
ObjectMagValueList = []
CatalogList = []
RadiusList = []
RefRadiusList = []
# with open('values.csv') as csvfile:
    # reader = csv.DictReader(csvfile)
    # for row in reader:
        # XList.append(row['XList'])
        # YList.append(row['YList'])
        # ObjectMagValueList.append(row['ObjectMagValue'])
        # CatalogList.append(row['Catalog'])
        # RadiusList.append(row['Radius'])
        # RefRadiusList.append(row['RefRadius'])

num = 0
while grabbed == True:

    (grabbed,img) = vid.read()
    if grabbed == False:
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def FindingMax(X,Y,choice):
        if choice == 'x':
            SearchRange = np.arange(X-10,X+10)
        if choice == 'y':
            SearchRange = np.arange(Y-10,Y+10)
        Coordinates = []
        for i in np.arange(-10,10):
            if choice == 'x':
                Coordinates.append(img[Y, X+i])
            if choice == 'y':
                Coordinates.append(img[Y+i, X])

        MaxValue = np.argmax(np.array(Coordinates))
        MaxLoc = SearchRange[MaxValue]

        return(MaxLoc)

        ##################################################
        # In order to center the x and y of both items
        # we call the FindingMax function for/ them
        # And this center we will use in the FindingGaussian function
        ##################################################

    XMaxLocObj = FindingMax(ObjectX, ObjectY, 'x')
    YMaxLocObj = FindingMax(ObjectX, ObjectY, 'y')

    ObjectLoc = (XMaxLocObj, YMaxLocObj)

    XList.append(X)
    YList.append(Y)

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

    def FindingGaussian(MaxLoc,choice,selection):
        global ObjectXPixels
        global ObjectYPixels
        global ReferenceXPixels
        global ReferenceYPixels

        if selection == 'ReferenceStar':
            YMaxLoc = YMaxLocRef
            XMaxLoc = XMaxLocRef
        if selection == 'Object':
            YMaxLoc = YMaxLocObj
            XMaxLoc = XMaxLocObj

        Range = np.arange(MaxLoc-10,MaxLoc+10)
        Coordinates = []
        for i in np.arange(-10,10):
            if choice == 'x':
                Coordinates.append(img[YMaxLoc, MaxLoc+i]) 
                if selection == 'Object':
                    ObjectXPixels = np.array(Coordinates)
                if selection == 'ReferenceStar':
                    ReferenceXPixels = np.array(Coordinates)
            if choice == 'y':
                Coordinates.append(img[MaxLoc+i, XMaxLoc])
                if selection == 'Object':
                    ObjectYPixels = np.array(Coordinates)
                if selection == 'ReferenceStar':
                    ReferenceYPixels = np.array(Coordinates) 

        Mean = MaxLoc
        Sigma = 2
        FitParameters, pcov = curve_fit(Gaussian, Range, Coordinates, p0 = [np.max(Coordinates), Mean, Sigma])
        return(FitParameters)

    ##################################################
    # FitParameters contains the amplitude, standard deviation and mean
    # The standard deviation is needed later so we call this function 
    # for both axes for both items
    ##################################################

    XFitParametersRef = FindingGaussian(XMaxLocRef, 'x','ReferenceStar')
    YFitParametersRef = FindingGaussian(YMaxLocRef, 'y', 'ReferenceStar')
    XFitParametersObj = FindingGaussian(XMaxLocObj, 'x', 'Object')
    YFitParametersObj = FindingGaussian(YMaxLocObj, 'y', 'Object')

    ##################################################
    # Magnitude Finder
    # 
    ##################################################

    def MagnitudeFinder(Loc, XFitParameters, YFitParameters):
        global ReferenceStarAvgRadius
        global ObjectAvgRadius
        global ObjectBackgroundRadius
        global ReferenceBackgroundRadius

        YRadius = int(np.ceil(3*XFitParameters[2])) #This rounds up to the nearest integer
        XRadius = int(np.ceil(3*YFitParameters[2]))
        Radius = max(XRadius,YRadius)
        Range=[]
        # print('COORDINATES',Loc[0],Loc[1])
        for i in range(-Radius,Radius):
            for j in range(-Radius,Radius):
                if i**2 + j**2  <  Radius**2:
                    Range.append((i + Loc[0], j + Loc[1])[::-1])

        # print('Fit: ', XFitParameters[0])
        # print('Radius of Star:', Radius)
        RadiusList.append(Radius)

        BackgroundRadius = Radius+5
        BackgroundRange=[]
        for i in range(-BackgroundRadius,BackgroundRadius):
            for j in range(-BackgroundRadius,BackgroundRadius):
                if i**2 + j**2  <  BackgroundRadius**2 and i**2 + j**2 > Radius**2:
                    BackgroundRange.append((i + Loc[0], j + Loc[1]))
        # print('New Background')
        # print(BackgroundRange)
        # print("BACKGROUND",BackgroundRange)
        BackgroundValues = []
        for i in BackgroundRange:
            BackgroundValues.append(img[i])
        AvgBackgroundMag = sum(BackgroundValues)/len(BackgroundValues)

        MagValue = 0
        for i in Range:
            MagValue = MagValue + (img[i] - AvgBackgroundMag)

        
        if Loc == ReferenceStarLoc:
            #print('The average radius of the reference star is ',(XRadius**2 + YRadius**2)**.5, 'and the magnitude is', MagValue)
            RefRadiusList.append(Radius)
            ReferenceStarAvgRadius = Radius
            ReferenceBackgroundRadius = Radius+5
        if Loc == ObjectLoc:
            #print('The average radius of the object is', (XRadius**2 + YRadius**2)**.5, 'and the magnitude is', MagValue)
            RadiusList.append(Radius)
            ObjectAvgRadius = Radius
            ObjectBackgroundRadius = Radius+5
       
            cv2.circle(img,(Loc[0],Loc[1]),Radius,(0,0,255))
            cv2.circle(img,(Loc[0],Loc[1]),BackgroundRadius,(0,0,255))
            cv2.imshow("testwindow", img)
            cv2.waitKey(0)
        return(MagValue)

    ReferenceMagValue = MagnitudeFinder(ReferenceStarLoc, XFitParametersRef, YFitParametersRef)

    ObjectMagValue = MagnitudeFinder(ObjectLoc, XFitParametersObj, YFitParametersObj)
    ObjectMagValueList.append(ObjectMagValue)

    ##################################################
    #Photometry for finding the catalog value
    ##################################################

    InstrumentalMagnitude = -2.5*np.log10(ReferenceMagValue)
    #CatalogMagnitude = float(input('Enter catalog magnitude: '))
    CatalogMagnitude = 10 #Testing out before splitting up Ref and Obj
    Offset = InstrumentalMagnitude - CatalogMagnitude 

    ObjectCatalogValue = -2.5*np.log10(ObjectMagValue) - Offset
    #print('The catalog value of the object is *maybe*',  ObjectCatalogValue)



    def PlottingCurve(XMaxLoc, YMaxLoc, XFitParameters, YFitParameters, Radius):
        sns.set()
        sns.set_style("dark")
        sns.set_context("poster")  
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2,1])
        ax = plt.subplot(gs[0, 0]) 

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
        ax4 = plt.subplot(gs[3])
        

        PlotRange = 10

        #Top Left
        inverseimg = cv2.bitwise_not(img)
        ax1.imshow(img,cmap='gray')
        ax1.set_xlim(-PlotRange+XMaxLoc,PlotRange+XMaxLoc)
        ax1.set_ylim(-PlotRange+YMaxLoc,PlotRange+YMaxLoc)
        if XMaxLoc == XFitParametersObj[1]:
            ax1.set_title('Magnitude of %s'%(round(ObjectCatalogValue,3)))
            ax1.add_artist(plt.Circle((XMaxLoc,YMaxLoc),ObjectBackgroundRadius, color = 'yellow', alpha=.2))
            ax1.add_artist(plt.Circle((XMaxLoc,YMaxLoc),Radius,color='red', alpha=0.2))
        if XMaxLoc == XFitParametersRef[1]:
            ax1.set_title('Magnitude of %s' %(CatalogMagnitude))
            ax1.add_artist(plt.Circle((XMaxLoc,YMaxLoc),Radius,color='blue', alpha=0.2))
            ax1.add_artist(plt.Circle((XMaxLoc,YMaxLoc),ReferenceBackgroundRadius, color = 'yellow', alpha=.2))
        # ax1.grid()
        ax1.axis('off')
        
        #Top Right
        if XMaxLoc == XFitParametersObj[1]:
            ax2.plot(ObjectYPixels,np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),label='Data',color='orange')
            ax2.plot((Gaussian(np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),*YFitParameters)),
            np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),label='Gaussian Fit',color='red')
        if XMaxLoc == XFitParametersRef[1]:
            ax2.plot(ReferenceYPixels,np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),label='Data')
            ax2.plot((Gaussian(np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),*YFitParameters)),
            np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),label='Gaussian Fit')
        #ax2.legend(loc="lower right")
        ax2.yaxis.set_visible(False)
       # ax2.set_xlabel('Pixel Values')
       # ax2.xaxis.set_label_position('top')
       # ax2.xaxis.set_label_coords(.34,1.11)
        ax2.xaxis.tick_top()
        
        #Bottom Left
        if XMaxLoc == XFitParametersObj[1]:
            ax3.plot(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),ObjectXPixels,label='Data',color='orange')
            ax3.plot(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),
            Gaussian(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),*XFitParameters),label='Gaussian Fit',color='red')
        if XMaxLoc == XFitParametersRef[1]:
            ax3.plot(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),ReferenceXPixels,label='Data')
            ax3.plot(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc), 
            Gaussian(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),*XFitParameters),label='Gaussian Fit')
        ax3.legend(bbox_to_anchor=(0., -.2, 1., .102), loc=3,ncol=2, borderaxespad=0.)
       # ax3.set_ylabel('Pixel Values')
       # ax3.yaxis.set_label_coords(-0.11,.34)
        ax3.invert_yaxis()
        ax3.xaxis.set_visible(False)
        # plt.xticks([])
        
        #Bottom Right
        ax4.imshow(img*20,cmap='gray')
        if XMaxLoc == XFitParametersObj[1]:
            ax4.axvline(ObjectX, color='red')
            ax4.axhline(ObjectY, color='red')
        if XMaxLoc == XFitParametersRef[1]:
            ax4.axvline(ReferenceStarX, color='blue')
            ax4.axhline(ReferenceStarY, color='blue')
        ax4.axis('off')

        plt.subplots_adjust(wspace=.1)
        plt.subplots_adjust(hspace=.1)
        plt.draw()

        #Title
        if XMaxLoc == XFitParametersObj[1]:
            plt.suptitle('Object')
        if XMaxLoc == XFitParametersRef[1]:
            plt.suptitle('Reference Star')

        plt.savefig(f'testplot{num}.pdf')

    PlottingCurve(XFitParametersObj[1], YFitParametersObj[1], XFitParametersObj, YFitParametersObj, ObjectAvgRadius)
    print(num)
    num += 1
plt.plot(np.arange(0,len(ObjectMagValueList)),ObjectMagValueList)
plt.show()
##################################################
# Subplots
##################################################
'''
def PlottingCurve(XMaxLoc, YMaxLoc, XFitParameters, YFitParameters, Radius):
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

    #Top Left
    inverseimg = cv2.bitwise_not(img)
    ax1.imshow(img,cmap='gray')
    ax1.set_xlim(-PlotRange+XMaxLoc,PlotRange+XMaxLoc)
    ax1.set_ylim(-PlotRange+YMaxLoc,PlotRange+YMaxLoc)
    if XMaxLoc == XFitParametersObj[1]:
        ax1.set_title('Magnitude of %s'%(round(ObjectCatalogValue,3)))
        ax1.add_artist(plt.Circle((XMaxLoc,YMaxLoc),ObjectBackgroundRadius, color = 'yellow', alpha=.2))
        ax1.add_artist(plt.Circle((XMaxLoc,YMaxLoc),Radius,color='red', alpha=0.2))
    if XMaxLoc == XFitParametersRef[1]:
        ax1.set_title('Magnitude of %s' %(CatalogMagnitude))
        ax1.add_artist(plt.Circle((XMaxLoc,YMaxLoc),Radius,color='blue', alpha=0.2))
        ax1.add_artist(plt.Circle((XMaxLoc,YMaxLoc),ReferenceBackgroundRadius, color = 'yellow', alpha=.2))
    # ax1.grid()
    ax1.axis('off')
    
    #Top Right
    if XMaxLoc == XFitParametersObj[1]:
        ax2.plot(ObjectYPixels,np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),label='Data',color='orange')
        ax2.plot((Gaussian(np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),*YFitParameters)),
        np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),label='Gaussian Fit',color='red')
    if XMaxLoc == XFitParametersRef[1]:
        ax2.plot(ReferenceYPixels,np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),label='Data')
        ax2.plot((Gaussian(np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),*YFitParameters)),
        np.arange(-PlotRange+YMaxLoc,PlotRange+YMaxLoc,1),label='Gaussian Fit')
    #ax2.legend(loc="lower right")
    ax2.yaxis.set_visible(False)
   # ax2.set_xlabel('Pixel Values')
   # ax2.xaxis.set_label_position('top')
   # ax2.xaxis.set_label_coords(.34,1.11)
    ax2.xaxis.tick_top()
    
    #Bottom Left
    if XMaxLoc == XFitParametersObj[1]:
        ax3.plot(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),ObjectXPixels,label='Data',color='orange')
        ax3.plot(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),
        Gaussian(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),*XFitParameters),label='Gaussian Fit',color='red')
    if XMaxLoc == XFitParametersRef[1]:
        ax3.plot(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),ReferenceXPixels,label='Data')
        ax3.plot(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc), 
        Gaussian(np.arange(-PlotRange+XMaxLoc,PlotRange+XMaxLoc),*XFitParameters),label='Gaussian Fit')
    ax3.legend(bbox_to_anchor=(0., -.2, 1., .102), loc=3,ncol=2, borderaxespad=0.)
   # ax3.set_ylabel('Pixel Values')
   # ax3.yaxis.set_label_coords(-0.11,.34)
    ax3.invert_yaxis()
    ax3.xaxis.set_visible(False)
    # plt.xticks([])
    
    #Bottom Right
    ax4.imshow(img*20,cmap='gray')
    if XMaxLoc == XFitParametersObj[1]:
        ax4.axvline(ObjectX, color='red')
        ax4.axhline(ObjectY, color='red')
    if XMaxLoc == XFitParametersRef[1]:
        ax4.axvline(ReferenceStarX, color='blue')
        ax4.axhline(ReferenceStarY, color='blue')
    ax4.axis('off')

    plt.subplots_adjust(wspace=.1)
    plt.subplots_adjust(hspace=.1)
    plt.draw()

    #Title
    if XMaxLoc == XFitParametersObj[1]:
        plt.suptitle('Object')
    if XMaxLoc == XFitParametersRef[1]:
        plt.suptitle('Reference Star')


PlottingCurve(XFitParametersRef[1], YFitParametersRef[1], XFitParametersRef, YFitParametersRef, ReferenceStarAvgRadius)
plt.figure()
PlottingCurve(XFitParametersObj[1], YFitParametersObj[1], XFitParametersObj, YFitParametersObj, ObjectAvgRadius)

plt.show()
'''

