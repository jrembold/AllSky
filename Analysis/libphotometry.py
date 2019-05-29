##########################################
# Library for photometric functions
# Author(s): Jed Rembold, Luke Russell
##########################################

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt


def readGrayImage(filename):
    """Reads in an image file and return a grayscale version."""
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)


def selectPoint(image):
    """Displays an image in matplotlib upon which a point can be selected."""
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    plt.show()
    x = input("Desired x coordinate? ")
    y = input("Desired y coordinate? ")
    plt.close()
    return (x, y)


def threshold(img, thresh):
    """Return a quick binary image at a certain threshold."""
    return cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]


def find_largest_contour(bin_img):
    """Returns the largest bright contour in a binary image."""
    conts, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(conts) < 1:
        return None
    largest_cont = conts[np.argmax([cv2.contourArea(c) for c in conts])]
    return largest_cont


def find_center_mass(contour):
    """Returns the center point of a given contour."""
    M = cv2.moments(contour)
    cR = int(M["m01"] / M["m00"])
    cC = int(M["m10"] / M["m00"])
    return (cR, cC)


def compute_obj_intensity(img, threshold_=40):
    """Primary function to compute the total measured brightness of the largest bright
    object in an image.
    """
    # Computing center and radius
    bin_img = threshold(img, threshold_)
    big_cont = find_largest_contour(bin_img)
    center, radius = cv2.minEnclosingCircle(big_cont)

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

    # Creating masks
    sig_r = 2
    bg_r1, bg_r2 = 5, 9
    signal = create_circle_mask(img, center, radius + sig_r)
    background = np.logical_xor(
        create_circle_mask(img, center, radius + bg_r1),
        create_circle_mask(img, center, radius + bg_r2),
    )
    summed_intensity = np.sum(img[signal] - np.mean(img[background]))

    # -----------------------
    # Temporary and just for display and testing
    plt.close()
    plt.imshow(img, cmap="gray")
    from matplotlib.patches import Circle

    p = Circle(center, radius + sig_r, fill=False, ec="green", lw=2)
    p2 = Circle(center, radius + bg_r1, fill=False, ec="cyan", lw=2)
    p3 = Circle(center, radius + bg_r2, fill=False, ec="cyan", lw=2)
    plt.gca().add_patch(p)
    plt.gca().add_patch(p2)
    plt.gca().add_patch(p3)
    # plt.imshow(signal | background)  # Shows masks
    plt.show()
    # -----------------------

    return summed_intensity


def compute_obj_mag(obj_flux, candle_flux, candle_mag):
    """Calculates the apparent magnitude of an object given the measured
    and catalog values of some standard candle.
    """
    instrument_mag = -2.5 * np.log10(candle_flux)
    offset = instrument_mag - candle_mag
    return -2.5 * np.log10(obj_flux) - offset


class SubImage(object):
    """Class to form subimages and keep track of their center."""

    def __init__(self, img, cent, radius, thresh=40):
        self.img = img
        self.r, self.c = cent
        self.radius = radius
        self.thresh = thresh
        self.data = None
        self.calcdata()
        self.calcthresh()

    def calcdata(self):
        """Computes the relevant part of the image and sets data attribute to that portion."""
        bot = max(0, self.r - self.radius)
        top = min(self.img.shape[0], self.r + self.radius)
        left = max(0, self.c - self.radius)
        right = min(self.img.shape[1], self.c + self.radius)
        self.data = self.img[bot:top, left:right]

    def calcthresh(self):
        """Computes what should be an acceptable threshold above the background noise."""
        mu = np.mean(self.data.ravel())
        sd = np.std(self.data.ravel())
        self.thresh = mu + 3 * sd

    def new_cent(self, newcent):
        """Provides a new center to the subimage and recomputes the data.

        newcent should be of the form (row, column)
        """
        self.r, self.c = newcent
        self.calcdata()

    def autocenter(self):
        """Function to recenter a subimage on the largest bright object in the scene."""
        big_contour = find_largest_contour(threshold(self.data, self.thresh))
        corr_r, corr_c = find_center_mass(big_contour)
        self.new_cent((self.r - self.radius + corr_r, self.c - self.radius + corr_c))

    def get_flux(self):
        """Compute the summed intensity or flux of the subimage."""
        return compute_obj_intensity(self.data, self.thresh)

    def show(self):
        """Plots subimage to a window."""
        plt.close()  # Remove any existing plot
        plt.imshow(self.data)
        plt.colorbar()
        plt.show()

    def __repr__(self):
        return "A SubImage centered at row {} and column {} with radius {}.".format(
            self.r, self.c, self.radius
        )
