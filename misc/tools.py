import glob
import os
import math
import colorsys
import pickle

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


"""
beijbom_misc_tools contains a bunch of nice misc python tools 
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
    crops the center of input image im.
    """
    if not type(ps) == int:
        raise TypeError('INPUT ps must be a scalar')
    center = [s/2 for s in im.shape[:2]]
    el = [ps / 2, ps / 2] if ps % 2 == 0 else [ps / 2, ps/2 + 1] # edge length
    return(im[center[0] - el[0] : center[0] + el[1], center[1] - el[0] : center[1] + el[1], :])

    
def crop_and_rotate(im, center, ps, angle, tile = False):
    """
    crop_and_rotate returns a rotated and cropped patch from input image im.
    To save comp. power, it first cropps a larger patch, then rotates that, and finally calls crop_center to get the desired cropped patch
    """    
    assert type(ps) == int, 'INPUT ps must be a scalar'

    if tile:
        offset = np.asarray(im.shape[:2])
        center = [offset[0] + center[0], offset[1] + center[0]]
        im = tile_image(im)
    tmp = ((math.ceil(ps * 2**.5) + 1) // 2 ) * 2 # round up and make even
    psbig = int(tmp) if ps % 2 == 0 else int(tmp) + 1 
    el = [psbig / 2, psbig/2] if psbig % 2 == 0 else [psbig / 2, psbig/2 + 1] # edge length
    bigpatch = im[center[0] - el[0] : center[0] + el[1], center[1] - el[0]:center[1] + el[1], :] # crop big patch
    return(crop_center(rotate_with_PIL(bigpatch, angle), ps))

def rotate_with_PIL(im, angle):
    """
    Helper function to crop_and_rotate. Image rotation is quicker with PIL than with numpy.
    """
    im = Image.fromarray(im)
    im.rotate(angle)
    return np.asarray(im)

def tile_image(im):
    """
    tiles input image so that all edges are mirrored
    """
    r1 = np.concatenate((im[::-1,::-1], im[::-1], im[::-1, ::-1]), 1)
    r2 = np.concatenate((im[:,::-1], im, im[:, ::-1]), 1)
    r3 = np.concatenate((im[::-1,::-1], im[::-1], im[::-1, ::-1]), 1)
    return(np.concatenate((r1, r2,r3), 0))


def vis_square(data, padsize=1, padval=0, scale = 1):
    """
    This function is from the caffe tutorials. 

    Takes 
    data: an array of shape (n, height, width) or (n, height, width, channels)
    
    Gives
    Visualizes each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    """

    data -= data.min()
    data *= scale
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)

def int_to_rgb(im, bg_color = 0, ignore = 255, nclasses = None):
    """
    Converts integer valued np image array to an rgb color image.

    Takes
    im: (w x h) uint8, nparray image

    Gives
    (w x h x 3) uint8, nparray image
    """
    gt_vals = list(set(np.unique(im)) - set([ignore]))
    gt_vals = [int(g) for g in gt_vals]
    
    if nclasses is None:
        nclasses = np.max(gt_vals) + 1
    RGB_tuples = get_good_colors(nclasses)

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
    if not len(gt) == len(est):
        raise ValueError('input gt and est must have the same length')
    return float(sum([(g == e) for (g,e) in zip(gt, est)])) / len(gt)


def liblin_cleanscores(liblin_scores, liblin_labellist, nclasses):
    """
    This arcane class restructures the scores output of liblinear.predict. 
    It makes it so that each score is ordered naturally, and that each score vector is of length nclasses.
    """
    lowscore = np.min(liblin_scores) - 1
    scores = []
    for lls in liblin_scores:
        s = np.ones(nclasses) * lowscore
        s[liblin_labellist] = lls
        scores.append(s)
    return scores
        

def make_coral_gt(imsize, annotations, blobsize = 5, ignore=255):
    """
    gt: H x W array of class values that make a label image
    """
    gt = np.ones(imsize, dtype=np.uint8) * ignore
    for (r, c, label) in annotations:
        for r_offset in range(-blobsize, 1 + blobsize, 1):
            for c_offset in range(-blobsize, 1 + blobsize, 1):
                if r+r_offset < gt.shape[0] and r+r_offset > -1 and c+c_offset < gt.shape[1] and c+c_offset > -1:
                    gt[r+r_offset, c+c_offset] = label
        return gt