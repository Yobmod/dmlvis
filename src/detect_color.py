from pyimagesearch.shapedetector import ShapeDetector
from pyimagesearch.colorlabeler import ColorLabeler
import argparse
import imutils
import cv2

from typing import Tuple, List
from typing_extensions import Literal
from mytypes import imageType

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image: imageType = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# blur the resized image slightly, then convert it to both
# grayscale and the L*a*b* color spaces
blurred = cv2.GaussianBlur(resized, (5, 5), 0)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# initialize the shape detector and color labeler
sd = ShapeDetector()
cl = ColorLabeler()

# loop over the contours
for c in cnts:
    # compute the center of the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)

    # detect the shape of the contour and label the color
    shape = sd.detect(c)
    color = cl.label(lab, c)

    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape and labeled
    # color on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    text = "{} {}".format(color, shape)
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(image, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def get_image_color_all(image: imageType, number: int = 0):
    """get array of all colors in an image"""
    from collections import Counter
    color_list = [tuple(color) for row in image for color in row]
    color_count = Counter(color_list)
    color_set = {tuple(color) for row in image for color in row}
    return color_set

def get_image_color_average():
    """get average color of image"""


def UV_to_rgb(wavelength: Literal[365, 395]) -> Tuple[int, int, int]:
    """get tuple of central color of UV wavelength image on white background"""
    if wavelength == 365:
        return (0, 0, 0)
    elif wavelength == 395:
        return (1, 1, 1)


def filter_for_color(image: imageType, color: Tuple[int, int, int], variation: float = 0.1) -> imageType:
    """remove all colors from image except given one plus some variation"""
