import numpy as np
import cv2 as cv
from .binarization import binarize_wolf

__all__ = [
    'isolate_lines',
    'invert_img'
]


def _erode(img, kernel_shape=(50,1)):
    return cv.erode(img, cv.getStructuringElement(cv.MORPH_RECT, kernel_shape))

def invert_img(img):
    return np.bitwise_not(img)

def isolate_lines(img):
    # Color to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Increase contrast
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_gray = clahe.apply(img_gray)
    # Binarization
    img_binary = binarize_wolf(img_gray)
    # Erosion
    return invert_img(_erode(invert_img(img_binary), (50,1)))
