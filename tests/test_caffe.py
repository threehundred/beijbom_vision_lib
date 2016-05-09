import os
import sys
import copy
import shutil
import caffe
import unittest

import os.path as osp
import numpy as np

from PIL import Image
from caffe import layers as L, params as P


import beijbom_vision_lib.caffe.tools as bct

from beijbom_vision_lib.caffe.nets.vgg import vgg_core

class TestCaffeTools(unittest.TestCase):

    workdir = 'workdir'
    basedir = ''

    def setUp(self):
        self.basedir = os.getcwd()
        os.chdir(self.basedir)
        os.makedirs(self.workdir)

    def tearDown(self):
        os.chdir(self.basedir)
        shutil.rmtree(self.workdir)

    def test_transformer(self):

        img = np.uint8(Image.open('static/bbc.jpg'))

        t = bct.Transformer()

        self.assertEqual(np.all(img == t.deprocess(t.preprocess(img))), True)

    def test_solver_copy(self):
        
        solver = bct.CaffeSolver()
        
        solver.write(osp.join(self.workdir, 'solver.prototxt'))
        solver2 = bct.CaffeSolver(empty = True)
        solver2.add_from_file(osp.join(self.workdir, 'solver.prototxt'))

        for k1, k2 in zip(sorted(solver.sp.keys()), sorted(solver2.sp.keys())):
            self.assertEqual(k1, k2)
            self.assertEqual(solver.sp[k1], solver2.sp[k2])

        # try again with the onlytrain mode
        solver = bct.CaffeSolver(onlytrain = True)
        
        solver.write(osp.join(self.workdir, 'solver.prototxt'))
        solver2 = bct.CaffeSolver(empty = True)
        solver2.add_from_file(osp.join(self.workdir, 'solver.prototxt'))

        for k1, k2 in zip(sorted(solver.sp.keys()), sorted(solver2.sp.keys())):
            self.assertEqual(k1, k2)
            self.assertEqual(solver.sp[k1], solver2.sp[k2])

    def test_solver_empty(self):

        solver = bct.CaffeSolver(empty = True)

        self.assertEqual(len(solver.sp.keys()), 0)

    def test_solver_onlytrain(self):

        solver = bct.CaffeSolver(onlytrain = True)
        for k in solver.sp.keys():
            self.assertEqual(k.find('test'), -1)

    def test_solver_debug(self):

        solver = bct.CaffeSolver(debug = True)
        self.assertEqual(solver.sp['test_iter'], '1')

        solver = bct.CaffeSolver(debug = True, onlytrain = True)
        self.assertFalse('test_iter' in solver.sp.keys())
        

    def test_find_latest_caffemodel(self):

        for iter_ in [1, 25, 121]:
            os.system('touch ./workdir/snapshot_iter_{}.caffemodel'.format(iter_))

        self.assertEqual(bct.find_latest_caffemodel(self.workdir)[0], 'snapshot_iter_121.caffemodel')
        self.assertEqual(bct.find_latest_caffemodel(self.workdir)[1], 121)
        self.assertEqual(bct.find_latest_caffemodel(self.workdir, prefix = 'not-snapshot')[0], None)
        self.assertEqual(bct.find_latest_caffemodel(self.workdir, suffix = 'not-caffemodel')[0], None)
        self.assertEqual(bct.find_latest_caffemodel(self.workdir, suffix = 'not-caffemodel')[1], 0)
        

    def test_find_latest_caffemodel_edge(self):
        self.assertEqual(bct.find_latest_caffemodel(self.workdir)[0], None)
        self.assertEqual(bct.find_latest_caffemodel(self.workdir)[1], 0)
     
    def test_workdir_setup(self):

        solver = bct.CaffeSolver(debug = True)
        
        solver.write(osp.join(self.workdir, 'solver.prototxt'))

        n = caffe.NetSpec()
        n.data, n.label = L.ImageData(transform_param = dict(crop_size=224, mean_value=128), source = '../static/imlist.txt', batch_size = 50, ntop=2)
        net = vgg_core(n, learn = True)

        net.score = L.InnerProduct(net.fc7, num_output=2, param=[dict(lr_mult=5, decay_mult=1), dict(lr_mult=10, decay_mult=0)])
        net.loss = L.SoftmaxWithLoss(net.score, n.label)

        with open(osp.join(self.workdir, 'trainnet.prototxt'), 'w') as w:
            w.write(str(net.to_proto()))

        with open(osp.join(self.workdir, 'testnet.prototxt'), 'w') as w:
            w.write(str(net.to_proto())) 

        caffefile = '/runs/templates/VGG_ILSVRC_16_layers_initial.caffemodel'
        if osp.isfile(caffefile):
            shutil.copyfile(caffefile, osp.join(self.workdir, 'initial.caffemodel'))       

        bct.run(self.workdir, nbr_iters = 3)

        self.assertTrue(osp.isfile(osp.join(self.workdir, 'train.log')))
        self.assertTrue(osp.isfile(osp.join(self.workdir, 'snapshot_iter_3.caffemodel')))

        caffemodel, iter_ = bct.find_latest_caffemodel(self.workdir)

        self.assertEqual(iter_, 3)
        net = bct.load_model(self.workdir, caffemodel, gpuid = 0, net_prototxt = 'testnet.prototxt', phase = caffe.TEST)
        estlist, scorelist = bct.classify_from_datalayer(net, n_testinstances = 3, batch_size = 50, scorelayer = 'score')

        self.assertEqual(len(scorelist), 3)
        self.assertEqual(len(estlist), 3)
        self.assertEqual(len(scorelist[0]), 2)

        img = np.asarray(Image.open('../static/bbc.jpg'))[:224, :224, :]
        imglist = []
        for itt in range(6):
            imglist.append(img)
        
        estlist, scorelist = bct.classify_from_imlist(imglist, net, bct.Transformer(), 4)

        self.assertEqual(len(scorelist), 6)
        self.assertEqual(len(estlist), 6)
        self.assertEqual(len(scorelist[0]), 2)


if __name__ == '__main__':
    if osp.isdir('workdir'):
        shutil.rmtree('workdir')
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCaffeTools)
    unittest.TextTestRunner(verbosity=2).run(suite)
