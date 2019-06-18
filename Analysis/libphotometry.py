##########################################
# Library for photometric functions
# Author(s): Jed Rembold, Luke Russell
##########################################

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os

plt.ioff()

RADIUS_SIGNAL = 3
RADIUS_BGINNER, RADIUS_BGOUTER = 5, 10


def readGrayImage(filename):
    """Reads in an image file and return a grayscale version."""
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)


def selectPoint(image):
    """Displays an image in matplotlib upon which a point can be selected."""
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    plt.show()
    x = int(input("Desired x coordinate? "))
    y = int(input("Desired y coordinate? "))
    plt.close()
    return (x, y)


def threshold(img, thresh):
    """Return a quick binary image at a certain threshold."""
    return cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]


def blur(img, kern=(3, 3)):
    """Do a quick gaussian blur on an image."""
    return cv2.GaussianBlur(img, kern, 0)


def find_largest_contour(bin_img, img):
    """Returns the largest brightest contour in an binary image given the actual
    image as well. Sums the pixel values in the actual image for each given contour
    and returns the contour which results in the largest sum.
    """
    conts, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(conts) < 1:
        # raise ValueError("No contours were found in the image.")
        return False, conts
    bestcont = conts[0]
    maxsum = 0
    for i, cont in enumerate(conts):
        mask = cv2.drawContours(
            np.zeros(bin_img.shape, dtype=np.uint8), conts, i, (1,), -1
        )
        thesum = img[mask.astype(bool)].sum()
        if thesum > maxsum:
            bestcont = cont
            maxsum = thesum
    return True, bestcont


def find_center_mass(contour):
    """Returns the center point of a given contour."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        (x, y), _ = cv2.minEnclosingCircle(contour)
        cR = int(y)
        cC = int(x)
        # raise ValueError("Contour too small to find a new center.")
    else:
        cR = int(M["m01"] / M["m00"])
        cC = int(M["m10"] / M["m00"])
    return (cR, cC)


def create_circle_mask(img, center, radius):
    """Broadcasting method of creating a boolean mask of all points within
    a particular radius of some center point in a image.
    """
    h, w = img.shape
    # Flipping center ordering here to account for differences in how opencv returns
    # coordinates (standard x,y) and how numpy wants them (r,c)
    act_center = np.array(center[::-1])
    r, c = np.ogrid[0:h, 0:w] - np.array(center[::-1])
    return (r ** 2 + c ** 2) < radius ** 2


def compute_obj_intensity(img, big_cont):
    """Primary function to compute the total measured brightness of the largest bright
    object in an image.
    """
    # # Computing center and radius
    # bin_img = threshold(blur(img), threshold_)

    # cont_found, big_cont = find_largest_contour(bin_img, blur(img))
    # if cont_found:
    center, radius = cv2.minEnclosingCircle(big_cont)

    # Creating masks
    sig_r = RADIUS_SIGNAL
    bg_r1, bg_r2 = RADIUS_BGINNER, RADIUS_BGOUTER
    signal = create_circle_mask(img, center, radius + sig_r)
    background = np.logical_xor(
        create_circle_mask(img, center, radius + bg_r1),
        create_circle_mask(img, center, radius + bg_r2),
    )
    summed_intensity = np.sum(img[signal] - np.mean(img[background]))
    return (
        summed_intensity,
        radius + sig_r,
        radius + bg_r1,
        radius + bg_r2,
        np.mean(img[background]),
    )


def imageframe(sf, identifier, loc="."):
    """
    Saves an image of the subfield of interest with relevant circles
    drawn atop to indicate what was considered part of the signal and
    what was considered part of the background. Will be saved in the
    folder loc.

    subframe: SubFrame, image data to be saved
    identifier: unique code or string to identify this particular image
    loc: string, full folder location to store images. Defaults to cwd
    """
    from matplotlib.patches import Circle

    # Computing center and radius
    blurred = blur(sf.data)
    bin_img = threshold(blurred, sf.calcthresh(blurred))
    cont_found, big_cont = find_largest_contour(bin_img, blurred)
    if cont_found:
        _, radius = cv2.minEnclosingCircle(big_cont)
        center = (sf.c, sf.r)
        sig_r = RADIUS_SIGNAL
        bg_r1, bg_r2 = RADIUS_BGINNER, RADIUS_BGOUTER
        # -----------------------
        # Temporary and just for display and testing
        fig, ax = plt.subplots()
        ax.imshow(
            sf.data,
            cmap="gray",
            extent=[
                sf.c - sf.radius,
                sf.c + sf.radius,
                sf.r + sf.radius,
                sf.r - sf.radius,
            ],
        )
        p = Circle(center, radius + sig_r, fill=False, ec="green", lw=2)
        p2 = Circle(center, radius + bg_r1, fill=False, ec="cyan", lw=2)
        p3 = Circle(center, radius + bg_r2, fill=False, ec="cyan", lw=2)
        ax.add_patch(p)
        ax.add_patch(p2)
        ax.add_patch(p3)
        fig.suptitle(f"Frame {identifier}", size=12, weight="bold")
        ax.set_title(sf.time.strftime("%Y%m%d %H:%M:%S.%f %Z"))
        # plt.imshow(signal | background)  # Shows masks
        if os.path.exists(loc):
            fig.savefig(f"{loc}/Frame{identifier}.png")
        else:
            os.makedirs(loc)
            fig.savefig(f"{loc}/Frame{identifier}.png")
        #plt.close(fig)
        # -----------------------


def compute_obj_mag(obj_flux, candle_flux, candle_mag):
    """Calculates the apparent magnitude of an object given the measured
    and catalog values of some standard candle.
    """
    instrument_mag = -2.5 * np.log10(candle_flux)
    offset = instrument_mag - candle_mag
    return -2.5 * np.log10(obj_flux) - offset


def follow_and_flux(vid, startframe, icent, radius=25, loc=None, verbose=False):
    """
    Function to follow a bright point throughout a video and calculate the
    flux through all subsequent frames. Function will error gracefully if no bright
    object is found in the frame any longer.

        vid: Video, object of type video class
        startframe: int, the frame number to start the analysis at
        icent: tuple (of ints), the initial location of the object in (r,c) format
        radius: int, the number of pixels in each direction which comprise the subframe
        loc: string, if given, will save snapshots to given folder location. No snapshots
            are saved otherwise.
        verbose: bool, prints results as they are found
    """
    results = []
    for i in range(startframe, vid.length):
        if i == startframe:
            cent_pt = icent
        else:
            cent_pt = sf.center
        sf = SubFrame(vid.get_frame(i), cent_pt, radius)
        sf.autocenter()
        if not sf.contour_exists:
            break
        flux, sig_r, bg_r1, bg_r2, _ = sf.get_flux()
        if loc is not None:
            imageframe(sf, i, loc=loc)
        data = {
            "Frame": i,
            "Time": sf.time,
            "X": sf.center[1],
            "Y": sf.center[0],
            "R_Sig": sig_r,
            "R_BG1": bg_r1,
            "R_BG2": bg_r2,
            "Flux": flux,
        }
        results.append(data)
        if verbose:
            print(
                f"Frame {i:>3.0f}: At {sf.center} with flux {flux:>9.2f}."
            )
    return pd.DataFrame(
        results, columns=["Frame", "Time", "X", "Y", "R_Sig", "R_BG1", "R_BG2", "Flux"]
    ).set_index("Frame")


class Frame(object):
    """Class to keep track of individual frame (or full image) information."""

    def __init__(self, data, time=None):
        self.data = data
        self.time = time
        self.shape = self.data.shape

    def show(self):
        plt.close()
        plt.imshow(self.data, cmap="gray")
        plt.title(self.time.strftime("%Y%m%d %H:%M:%S.%f %Z"))
        plt.show()

    def __repr__(self):
        return "{}(Shape: {}, Avg: {:0.2f}, Time: {})".format(
            __class__.__name__,
            self.data.shape,
            np.mean(self.data),
            self.time.strftime("%Y%d%m %H:%M:%S"),
        )


class SubFrame(object):
    """Class to form subframes from frames and keep track of their center."""

    def __init__(self, frame, cent, radius):
        self.frame = frame
        self.img = frame.data
        self.time = frame.time
        self.center = cent
        self.r, self.c = cent
        self.radius = radius
        self.data = None
        self.contour_exists = False
        self.largest_cont = None
        self.updatedata()

    def calcdata(self):
        """Computes the relevant part of the image and sets data attribute to that portion."""
        bot = max(0, self.r - self.radius)
        top = min(self.img.shape[0], self.r + self.radius)
        left = max(0, self.c - self.radius)
        right = min(self.img.shape[1], self.c + self.radius)
        self.data = self.img[bot:top, left:right]

    def calcthresh(self, img):
        """Computes what should be an acceptable threshold above the background noise."""
        mu = np.mean(img.ravel())
        sd = np.std(img.ravel())
        return mu + 3 * sd

    def updatedata(self):
        self.calcdata()
        self.thresh = self.calcthresh(self.data)
        self.get_biggest_contour()

    def new_cent(self, newcent):
        """Provides a new center to the subimage and recomputes the data.

        newcent should be of the form (row, column)
        """
        self.r, self.c = newcent
        self.center = (self.r, self.c)
        self.updatedata()

    def autocenter(self):
        """Function to recenter a subimage on the largest bright object in the scene."""
        if self.contour_exists:
            corr_r, corr_c = find_center_mass(self.largest_cont)
            self.new_cent(
                (self.r - self.radius + corr_r, self.c - self.radius + corr_c)
            )

    def get_flux(self):
        """Compute the summed intensity or flux of the subimage."""
        if self.contour_exists:
            return compute_obj_intensity(self.data, self.largest_cont)
        else:
            raise ValueError("No large bright object exists to measure the flux of!")

    def get_biggest_contour(self):
        """Return whether a contour exists and what the largest one is."""
        blurred = blur(self.data)
        bin_img = threshold(blurred, self.calcthresh(blurred))
        self.contour_exists, self.largest_cont = find_largest_contour(
            bin_img, self.data
        )

    def show(self):
        """Plots subimage to a window."""
        plt.close()  # Remove any existing plot
        plt.imshow(
            self.data,
            extent=[
                self.c - self.radius,
                self.c + self.radius,
                self.r + self.radius,
                self.r - self.radius,
            ],
        )
        plt.colorbar()
        plt.title(self.time.strftime("%Y%m%d %H:%M:%S.%f %Z"))
        plt.show()

    def __repr__(self):
        return "SubFrame(Center=({},{}), Radius={}) of \n{}".format(
            self.r, self.c, self.radius, self.frame
        )


class Video(object):
    """Class to facilitate working with video files."""

    def __init__(self, fname):
        self.fname = fname
        self.starttime = self.get_start_time()
        self.frame_dt = 1 / 29.97
        self.frames = None
        self.length = 0
        self.isdelaced = False
        self.centers = None
        self.fluxes = None
        self.readVideo()

    def readVideo(self):
        """Reads in a video file and saves the data in a 3D matrix."""
        vid = cv2.VideoCapture(self.fname)
        imgstack = []
        # grab = True
        grab, img = vid.read()
        while grab:
            imgstack.append(
                Frame(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    self.starttime
                    + datetime.timedelta(seconds=self.frame_dt * self.length),
                )
            )
            self.length += 1
            grab, img = vid.read()
        self.frames = imgstack

    def showframe(self, frame):
        """Plots a quick image of the desired frame from the video. Frame should
        be an integer.
        """
        self.frames[frame].show()

    def get_frame(self, frame):
        """Helper function to return a 2D desired frame from the video stack."""
        return self.frames[frame]

    def delace(self):
        """Deinterlaces the video by extracting even and odd lines and then splicing
        them back together. The overall effect is to half the vertical resolution
        while doubling the frame rate.
        """
        if not self.isdelaced:
            newframes = []
            self.frame_dt /= 2
            for i, frame in enumerate(self.frames):
                newframes.append(
                    Frame(
                        frame.data[1::2, :],
                        self.starttime
                        + datetime.timedelta(seconds=self.frame_dt * (2 * i)),
                    )
                )
                newframes.append(
                    Frame(
                        frame.data[0::2, :],
                        self.starttime
                        + datetime.timedelta(seconds=self.frame_dt * (2 * i + 1)),
                    )
                )
            self.isdelaced = True
            self.frames = newframes
            self.length = len(newframes)
        else:
            print("Video already deinterlaced.")

    def get_start_time(self):
        """Parses the filename to set a starting time and date for the video."""
        vidname = self.fname.split("/")[-1]
        date_, time_ = vidname.split("_")
        year = int(date_[:4])
        mon = int(date_[4:6])
        day = int(date_[6:])
        hour = int(time_[:2])
        min_ = int(time_[2:4])
        sec = int(time_[4:6])
        return datetime.datetime(
            year, mon, day, hour, min_, sec, tzinfo=datetime.timezone.utc
        )
