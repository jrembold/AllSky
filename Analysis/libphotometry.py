####################################
# Library for photometric functions
####################################

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt


def readGrayImage(filename):
    """Reads in an image file and return a grayscale version."""
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)


def selectPoint(image):
    """Displays an image in matplotlib upon which a point can be selected."""
    plt.style.use("seaborn-darkgrid")

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


def compute_exp_mag(img, threshold_=40):
    """Primary function to compute the experimental magnitude of the largest bright
    object in a scene.
    """
    bin_img = threshold(img, threshold_)
    big_cont = find_largest_contour(bin_img)
    (r, c), radius = cv2.minEnclosingCircle(big_cont)
    center = (round(r), round(c))
    radius = round(radius)

    print(r, c, radius)

    # Temporary and just for display and testing
    plt.imshow(img, cmap="gray")
    from matplotlib.patches import Circle

    p = Circle(center, radius * 1.2, fill=False, ec="green", lw=2)
    p2 = Circle(center, radius * 1.5, fill=False, ec="cyan", lw=2)
    p3 = Circle(center, radius * 2, fill=False, ec="cyan", lw=2)
    plt.gca().add_patch(p)
    plt.gca().add_patch(p2)
    plt.gca().add_patch(p3)
    plt.show()


class SubImage(object):
    """Class to form subimages and keep track of their center."""

    def __init__(self, img, cent, radius, thresh=40):
        self.img = img
        self.r, self.c = cent
        self.radius = radius
        self.thresh = thresh
        self.data = None
        self.calcdata()

    def calcdata(self):
        """Computes the relevant part of the image and sets data attribute to that portion."""
        bot = max(0, self.r - self.radius)
        top = min(self.img.shape[0], self.r + self.radius)
        left = max(0, self.c - self.radius)
        right = min(self.img.shape[1], self.c + self.radius)
        self.data = self.img[bot:top, left:right]

    def update(self, newcent):
        """Provides a new center to the subimage and recomputes the data

        newcent should be of the form (row, column)
        """
        self.r, self.c = newcent
        self.calcdata()

    def center(self):
        """Function to recenter a subimage on the largest bright object in the scene."""
        big_contour = find_largest_contour(threshold(self.data, self.thresh))
        corr_r, corr_c = find_center_mass(big_contour)
        self.update((self.r - self.radius + corr_r, self.c - self.radius + corr_c))

    def __repr__(self):
        return "A SubImage centered at row {} and column {} with radius {}.".format(
            self.r, self.c, self.radius
        )
