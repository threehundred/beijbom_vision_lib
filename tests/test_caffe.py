import os
import sys
import copy
import shutil
import caffe

import os.path as osp
import numpy as np

from PIL import Image
from caffe import layers as L, params as P


import beijbom_vision_lib.caffe.tools as bct

from beijbom_vision_lib.caffe.nets.vgg import vgg_core

workdir = 'workdir'

def run():
    
    if osp.isdir(workdir):
        shutil.rmtree(workdir)

    # run tests     
    transformer()   
    solver()
    workdir_setup()
    find_latest_caffemodel()


def transformer():

    img = np.uint8(Image.open('./static/bbc.jpg'))

    t = bct.Transformer()

    assert np.all(img == t.deprocess(t.preprocess(img)))

def solver():
    
    os.makedirs(workdir)

    solver = bct.CaffeSolver(debug = True)
    
    solver.write(osp.join(workdir, 'solver.prototxt'))

    solver_copy = copy.copy(solver)

    solver.add_from_file(osp.join(workdir, 'solver.prototxt'))

    for k1, k2 in zip(sorted(solver.sp.keys()), sorted(solver_copy.sp.keys())):
        assert k1 == k2
        assert solver.sp[k1] == solver.sp[k2]

    shutil.rmtree(workdir)


def find_latest_caffemodel():

    os.makedirs(workdir)
    for iter_ in [1, 25, 121]:
        os.system('touch ./workdir/snapshot_iter_{}.caffemodel'.format(iter_))

    assert bct.find_latest_caffemodel(workdir)[0] == 'snapshot_iter_121.caffemodel'
    assert bct.find_latest_caffemodel(workdir)[1] == 121
    assert bct.find_latest_caffemodel(workdir, prefix = 'not-snapshot')[0] == None
    assert bct.find_latest_caffemodel(workdir, suffix = 'not-caffemodel')[0] == None
    assert bct.find_latest_caffemodel(workdir, suffix = 'not-caffemodel')[1] == 0
    shutil.rmtree(workdir)

    os.makedirs(workdir)
    assert bct.find_latest_caffemodel(workdir)[0] == None
    assert bct.find_latest_caffemodel(workdir)[1] == 0
    shutil.rmtree(workdir)



def workdir_setup():

    os.makedirs(workdir)

    solver = bct.CaffeSolver(debug = True)
    
    solver.write(osp.join(workdir, 'solver.prototxt'))

    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(transform_param = dict(crop_size=224, mean_value=128), source = '../static/imlist.txt', batch_size = 50, ntop=2)
    net = vgg_core(n, learn = True)

    net.score = L.InnerProduct(net.fc7, num_output=2, param=[dict(lr_mult=5, decay_mult=1), dict(lr_mult=10, decay_mult=0)])
    net.loss = L.SoftmaxWithLoss(net.score, n.label)

    with open(osp.join(workdir, 'trainnet.prototxt'), 'w') as w:
        w.write(str(net.to_proto()))

    with open(osp.join(workdir, 'testnet.prototxt'), 'w') as w:
        w.write(str(net.to_proto())) 

    caffefile = '/runs/templates/VGG_ILSVRC_16_layers_initial.caffemodel'
    if osp.isfile(caffefile):
        shutil.copyfile(caffefile, osp.join(workdir, 'initial.caffemodel'))       


    bct.run(workdir, nbr_iters = 3)

    assert osp.isfile(osp.join(workdir, 'train.log'))
    assert osp.isfile(osp.join(workdir, 'snapshot_iter_3.caffemodel'))

    caffemodel, iter_ = bct.find_latest_caffemodel(workdir)

    assert iter_ == 3
    net = bct.load_model(workdir, caffemodel, gpuid = 0, net_prototxt = 'testnet.prototxt', phase = caffe.TEST)
    estlist, scorelist = bct.classify_from_datalayer(net, n_testinstances = 3, batch_size = 50, scorelayer = 'score')

    os.chdir('../')
    assert len(scorelist) == 3
    assert len(estlist) == 3
    assert len(scorelist[0]) == 2

    img = np.asarray(Image.open('static/bbc.jpg'))[:224, :224, :]
    imglist = []
    for itt in range(6):
        imglist.append(img)
    

    estlist, scorelist = bct.classify_from_imlist(imglist, net, bct.Transformer(), 4)

    assert len(scorelist) == 6
    assert len(estlist) == 6
    assert len(scorelist[0]) == 2

    shutil.rmtree(workdir)