import math
import colorsys
import pickle

import numpy as np

from scipy.ndimage.interpolation import rotate

"""
misc.tools contains a bunch of nice misc python tools 
"""

def psave(var, filename):
    """
    psave pickles and save var to filename
    """
    pickle.dump(var, open(filename, 'wb'))


def pload(filename):
    """
    pload opens and unpickles content of filename
    """
    return pickle.load(open(filename, 'rb'))


def crop_center(im, ps):
    """
    crops the center of input image im. If rounding prevents a centered patch, always grabs the UPPER LEFT.
    """
    if not type(ps) == int:
        raise TypeError('INPUT ps must be a scalar')
    if ps < 1:
        raise ValueError('INPUT ps must be larger than 0')

    upper = math.floor(im.shape[0] / 2.0 - ps / 2.0) # first row
    left = math.floor(im.shape[1] / 2.0 - ps / 2.0) # first column
    return(im[upper : upper + ps, left : left + ps, :])

def crop_and_rotate(im, center, ps, angle, tile = False):
    """
    crop_and_rotate returns a rotated and cropped patch from input image im.
    To save comp. power, it first cropps a larger patch, then rotates that, and finally calls crop_center to get the desired cropped patch
    """    
    assert type(ps) == int, 'INPUT ps must be a scalar'

    if tile:
        im = np.pad(im, ((ps * 2, ps * 2), (ps * 2, ps * 2), (0, 0)), mode='reflect')
        center = [center[0] + ps * 2, center[1] + ps * 2]

    # round up and make odd. Make at least size 5.
    psbig = int(1 + (math.ceil(ps * 2**.5) // 2 ) * 2) 
    upper = center[0] - (psbig / 2)
    left = center[1] - (psbig / 2)

    # crop patch with margins
    bigpatch = im[upper : upper + psbig, left : left + psbig, :] 
    return crop_center(rotate(bigpatch, angle), ps)


def int_to_rgb(im, bg_color = 0, ignore = 255, nmax = None):
    """
    Converts integer valued np image array to an rgb color image.

    Takes
    im: (w x h) uint8, nparray image

    Gives
    (w x h x 3) uint8, nparray image
    """
    gt_vals = list(set(im.flatten()) - set([ignore]))
    gt_vals = [int(g) for g in gt_vals]
    
    if nmax is None:
        nmax = np.max(gt_vals) + 1
    
    elif nmax <= np.max(gt_vals):
        raise ValueError('nmax must be as 1 larger than the largest class value.')
    
    RGB_tuples = get_good_colors(nmax)

    w, h = im.shape
    ret = np.ones((w, h, 3), dtype=np.uint8) * bg_color
   
    for label in gt_vals:
        ret[im == label] = RGB_tuples[label, :]
        
    return ret


def softmax(w):
    """
    Softmax operator on vector or matrices (along rows.)
    """
    w = np.asarray(w)
    e = np.exp(w)
    if len(w.shape) == 1:
        return e / np.sum(e)
    else:
        row_sums = e.sum(axis=1)
        return e / row_sums[:, np.newaxis]
     

def get_good_colors(N):
    """
    This nifty function returns optimally different N colors.
    """
    HSV_tuples = [(x*1.0/N, 0.5, 1) for x in range(N)]
    return(255 * np.array(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)))


def hist_stretch(im):
    """
    performs simple histogram stretch
    """

    hist,bins = np.histogram(im.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    return cdf[im]


def acc(gt, est):
    """
    Calculate the accuracy of (agreement between) two interger valued list.
    """
    if len(gt) == 0 or len(est) == 0:
        raise TypeError('Inputs can not be empty')

    if not len(gt) == len(est):
        raise ValueError('Input gt and est must have the same length')
    
    for g in gt:
        if not isinstance(g, int):
            raise TypeError('Input gt must be an array of ints')

    for e in est:
        if not isinstance(e, int):
            raise TypeError('Input est must be an array of ints')

    return float(sum([(g == e) for (g,e) in zip(gt, est)])) / len(gt)