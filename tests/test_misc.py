import unittest
import os
import shutil
import scipy.ndimage

import os.path as osp
import numpy as np
import liblinearutil as ll


from PIL import Image
from time import time

import beijbom_vision_lib.misc.tools as bmt


class TestMiscTools(unittest.TestCase):
    workdir = 'workdir'

    def test_crop_center(self):
        I = np.ones([9, 9, 3])
        I[4, 4, 1] = 0

        # should grab the center pixel.
        self.assertEqual(np.sum(bmt.crop_center(I, 1)), 2)

        # test exceptions
        self.assertRaises(ValueError, bmt.crop_center, I, 0)
        self.assertRaises(TypeError, bmt.crop_center, I, 'sdf')

        # For an even size input image, grab the upper left of the four potential center-pixels
        I = np.ones([10, 10, 3])
        I[4, 4, 1] = 0        
        self.assertEqual(np.sum(bmt.crop_center(I, 1)), 2)

        # Now, try grabbing a 2 x 2 of an odd size image
        I = np.ones([9, 9, 3])
        I[3, 3, 1] = 0
        I[4, 4, 1] = 0

        # grabs the upper left 2x2 patch.
        self.assertEqual(np.sum(bmt.crop_center(I, 2)), 10)

        # Now, try grabbing a 2 x 2 of an even size image.
        I = np.ones([10, 10, 3])
        I[4, 4, 1] = 0
        I[5, 5, 1] = 0        

        # grabs the center 2x2 patch.
        self.assertEqual(np.sum(bmt.crop_center(I, 2)), 10)

        # Now, try grabbing a 3 x 3 of an even size image.
        I = np.ones([10, 10, 3])
        I[3, 3, 1] = 0

        # grabs the upper-left 3x3 patch.
        self.assertEqual(np.sum(bmt.crop_center(I, 3)), 26)

    def test_crop_center_rectangular(self):
        I = np.ones([9, 99, 3])
        I[4, 49, 1] = 0        
        self.assertEqual(np.sum(bmt.crop_center(I, 1)), 2)

        
    def test_crop_and_rotate(self):
        
        # basic test for the middle pixel
        I = np.ones([19, 19, 3], dtype = np.uint8)
        I[9, 9, 1] = 100
        P = bmt.crop_and_rotate(I, [9, 9], 1, 0)
        self.assertEqual(np.sum(P), 102)

        # try tiling
        P = bmt.crop_and_rotate(I, [9, 9], 1, 0, tile = True)
        self.assertEqual(np.sum(P), 102)

        # 3x3 patch with rotation (hard to test for all combinations of angles and patch-sizes due to interpolations)
        I = np.zeros([219, 219, 3], dtype = np.uint8)
        I[9, 9, 1] = 100

        for ang in [23, 30, 45, 90, 135]:
            P = bmt.crop_and_rotate(I, [9, 9], 3, ang)
            self.assertEqual(P[1, 1, 1], 100)

        # even patch without rotation
        I = np.zeros([219, 119, 3], dtype = np.uint8)
        I[8, 8, 1] = 100

        P = bmt.crop_and_rotate(I, [9, 9], 2, 0)
        self.assertEqual(np.sum(P), 100)


    def test_save_load(self):
        a = {'field':22, 'another_field':'value'}
        bmt.psave(a, osp.join(self.workdir, 'test.p'))
        b = bmt.pload(osp.join(self.workdir, 'test.p'))
        self.assertEqual(a, b)

    def test_int_to_rgb(self):

        # try on an actual image
        im = np.asarray(Image.open('static/bbc.jpg'))
        gray = np.squeeze(np.round(im[:,:,0]))
        out = bmt.int_to_rgb(gray)
        self.assertEqual(len(out.shape), 3)

        # try a sweep of nargs
        im = np.ones([19, 19], dtype = np.uint8)
        im[9, 9] = 2

        out = bmt.int_to_rgb(im)
        for bg in range(5):
            for ig in range(255):
                out = bmt.int_to_rgb(im, bg_color = bg, ignore = ig)
        
        self.assertRaises(ValueError, bmt.int_to_rgb, im, bg_color = bg, nmax = 1)
        self.assertRaises(ValueError, bmt.int_to_rgb, im, bg_color = bg, nmax = 2)

    def test_softmax(self):
        a = np.asarray([1, 2, 3, 4])
        b = bmt.softmax(a)
        self.assertEqual(np.sum(b), 1)

        a = np.ones([400, 300])
        b = bmt.softmax(a)
        for i in range(400):
            self.assertEqual(np.abs(np.sum(b[i,:]) - 1) < 0.0000000001, True)

    
    def test_get_good_colors(self):
        colors = bmt.get_good_colors(10)
        self.assertEqual(np.max(colors) <= 255, True)
        self.assertEqual(colors.shape[0], 10)
        self.assertEqual(colors.shape[1], 3)


    def test_hist_stretch(self):
        im = np.asarray(Image.open('static/bbc.jpg'))
        out = bmt.hist_stretch(im)
        self.assertEqual(im.shape, out.shape)    

    def test_acc(self):
        gt = [1, 2, 0]
        est = [1, 2]
        self.assertRaises(ValueError, bmt.acc, gt, est)

        gt = [1, 2, 1.2]
        est = [1, 2, 2]
        self.assertRaises(TypeError, bmt.acc, gt, est)

        gt = np.asarray([1, 2, 1.2])
        est = [1, 2, 2]
        self.assertRaises(TypeError, bmt.acc, gt, est)

        gt = [1, 2, 1.2]
        est = [1, 2, 2.2]
        self.assertRaises(TypeError, bmt.acc, gt, est)

        self.assertRaises(TypeError, bmt.acc, [], [ ])

        self.assertEqual(bmt.acc([2, 2], [1, 2]), 0.5)

    def setUp(self):
        os.makedirs(self.workdir)

    def tearDown(self):
        shutil.rmtree(self.workdir)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMiscTools)
    unittest.TextTestRunner(verbosity=2).run(suite)