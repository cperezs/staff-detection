import itertools
import numpy as np
from scipy import ndimage as ndi
from skimage import util
from skimage.transform import integral_image
from skimage.filters import (threshold_otsu, threshold_sauvola)

__all__ = [
    'binarize_otsu',
    'binarize_sauvola',
    'binarize_wolf'
]

# Copied from https://github.com/scikit-image/scikit-image/blob/v0.13.1/skimage/filters/thresholding.py
def _mean_std(image, w):
    """Return local mean and standard deviation of each pixel using a
    neighborhood defined by a rectangular window with size w times w.
    The algorithm uses integral images to speedup computation. This is
    used by threshold_niblack and threshold_sauvola.
    Parameters
    ----------
    image : ndarray
        Input image.
    w : int
        Odd window size (e.g. 3, 5, 7, ..., 21, ...).
    Returns
    -------
    m : 2-D array of same size of image with local mean values.
    s : 2-D array of same size of image with local standard
        deviation values.
    References
    ----------
    .. [1] F. Shafait, D. Keysers, and T. M. Breuel, "Efficient
           implementation of local adaptive thresholding techniques
           using integral images." in Document Recognition and
           Retrieval XV, (San Jose, USA), Jan. 2008.
           DOI:10.1117/12.767755
    """
    if w == 1 or w % 2 == 0:
        raise ValueError(
            "Window size w = %s must be odd and greater than 1." % w)

    left_pad = w // 2 + 1
    right_pad = w // 2
    padded = np.pad(image.astype('float'), (left_pad, right_pad),
                    mode='reflect')
    padded_sq = padded * padded

    integral = integral_image(padded)
    integral_sq = integral_image(padded_sq)

    kern = np.zeros((w + 1,) * image.ndim)
    for indices in itertools.product(*([[0, -1]] * image.ndim)):
        kern[indices] = (-1) ** (image.ndim % 2 != np.sum(indices) % 2)

    sum_full = ndi.correlate(integral, kern, mode='constant')
    m = util.crop(sum_full, (left_pad, right_pad)) / (w ** image.ndim)
    sum_sq_full = ndi.correlate(integral_sq, kern, mode='constant')
    g2 = util.crop(sum_sq_full, (left_pad, right_pad)) / (w ** image.ndim)
    s = np.sqrt(g2 - m * m)
    return m, s

def threshold_wolf(image, window_size=15, k=0.2):
    m, s = _mean_std(image, window_size)
    return m + k * (s / s.max() - 1) * (m - image.min())

def binarize_otsu(img):
    return np.uint8((img > threshold_otsu(img)) * 255)

def binarize_sauvola(img, window_size=101, k=0.5):
    return np.uint8((img > threshold_sauvola(img, window_size=window_size, k=k)) * 255)

def binarize_wolf(img, window_size=101, k=0.5):
    return np.uint8((img > threshold_wolf(img, window_size=window_size, k=k)) * 255)
