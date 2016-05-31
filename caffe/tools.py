import glob
import os
import caffe
import sys
import json
import copy
import re
import tqdm

import os.path as osp
import numpy as np

from PIL import Image
from caffe import layers as L, params as P
from google.protobuf import text_format

import beijbom_vision_lib.misc.tools as bmt

"""
This module contains classes and wrappers for caffe.
"""

class Transformer:
    """
    Transformer is a class for preprocessing and deprocessing images according to the vgg16 pre-processing paradigm (scaling and mean subtraction.)
    """

    def __init__(self, mean = [0, 0, 0]):
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0

    def set_mean(self, mean):
        """
        Set the mean to subtract for centering the data.
        """
        self.mean = mean

    def set_scale(self, scale):
        """
        Set the data scaling.
        """
        self.scale = scale

    def preprocess(self, im):
        """
        preprocess() emulate the pre-processing occuring in the vgg16 caffe prototxt.
        """
    
        im = np.float32(im)
        im = im[:, :, ::-1] #change to BGR
        im -= self.mean
        im *= self.scale
        im = im.transpose((2, 0, 1))
        
        return im

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1] #change to RGB
        
        return np.uint8(im)


class CaffeSolver:
    """
    Caffesolver is a class for creating a solver.prototxt file. It sets default values and can export a solver parameter file.
    Note that all parameters are stored as strings. For technical reasons, the strings are stored as strings within strings.
    """

    def __init__(self, testnet_prototxt_path = "testnet.prototxt", trainnet_prototxt_path = "trainnet.prototxt", debug = False, empty = False, onlytrain = False):
       
        self.sp = {}

        # return empty solver.
        if empty:
            return

        # critical:
        self.sp['base_lr'] = '0.001'
        self.sp['momentum'] = '0.9'
        
        # speed:
        self.sp['test_iter'] = '100'
        self.sp['test_interval'] = '250'

        # looks:
        self.sp['display'] = '25'
        self.sp['snapshot'] = '2500'
        self.sp['snapshot_prefix'] = '"snapshot"' # string withing a string!
        
        # learning rate policy
        self.sp['lr_policy'] = '"fixed"'

        # important, but rare:
        self.sp['gamma'] = '0.1'
        self.sp['weight_decay'] = '0.0005'
        self.sp['train_net'] = '"' + trainnet_prototxt_path + '"'
        self.sp['test_net'] = '"' + testnet_prototxt_path + '"'

        # pretty much never change these.
        self.sp['max_iter'] = '1000000'
        self.sp['test_initialization'] = 'false'
        self.sp['average_loss'] = '25' # this has to do with the display.
        self.sp['iter_size'] = '1' #this is for accumulating gradients

        if debug:
            self.sp['max_iter'] = '12'
            self.sp['display'] = '1'
            
            self.sp['test_iter'] = '1'
            self.sp['test_interval'] = '4'

        if onlytrain:
            for f in ['test_net', 'test_iter', 'test_interval', 'test_initialization']:
                del self.sp[f]
            

    def add_from_file(self, filepath):
        """
        Reads a caffe solver prototxt file and updates the Caffesolver instance parameters.
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                splitLine = line.split(':')
                self.sp[splitLine[0].strip()] = splitLine[1].strip()
        return 1

    def write(self, filepath):
        """
        Export solver parameters to INPUT "filepath". Sorted alphabetically.
        """
        f = open(filepath, 'w')
        for key, value in sorted(self.sp.items()):
            if not(type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))
        return 1



def run(workdir = None, caffemodel = None, gpuid = 0, solverfile = 'solver.prototxt', log = 'train.log', snapshot_prefix = 'snapshot', caffepath = 'caffe', restart = False, nbr_iters = None):
    """
    run is a simple caffe wrapper for training nets. It basically does two things. (1) ensures that training continues from the most recent model, and (2) makes sure the output is captured in a log file.

    Takes
    workdir: directory where the net prototxt lives.
    caffemodel: name of a stored caffemodel.
    solverfile: name of solver.prototxt [this refers, in turn, to net.prototxt]
    log: name of log file
    snapshot_prefix: snapshot prefix. 
    caffepath: path the caffe binaries. This is required since we make a system call to caffe.
    restart: determines whether to restart even if there are snapshots in the directory.

    """

    # find initial caffe model
    if not caffemodel:
        caffemodel = glob.glob(os.path.join(workdir, "*initial.caffemodel"))
        if caffemodel:
            caffemodel = os.path.basename(caffemodel[0])

    # finds the latest snapshots
    latest_snapshot, max_iter = find_latest_caffemodel(workdir, prefix = snapshot_prefix, suffix = 'solverstate')

    # update solver with new max_iter parameter (if asked for)
    if not nbr_iters is None: 
        solver = CaffeSolver(empty = True)
        solver.add_from_file(os.path.join(workdir, solverfile))
        solver.sp['max_iter'] = str(max_iter + nbr_iters)
        solver.sp['snapshot'] = str(1000000) #disable this, don't need it.
        solver.write(os.path.join(workdir, solverfile))

    # by default, start from the most recent snapshot
    if latest_snapshot and not restart: 
        print "Running {} from iter {}.".format(workdir, max_iter)
        runstring  = 'cd {}; {} train -solver {} -snapshot {} -gpu {} 2>&1 | tee -a {}'.format(workdir, caffepath, solverfile, latest_snapshot, gpuid, log)

    # else, start from a pre-trained net defined in caffemodel
    elif caffemodel: 
        if(os.path.isfile(os.path.join(workdir, caffemodel))):
            print "Fine tuning {} from {}.".format(workdir, caffemodel)
            runstring  = 'cd {}; {} train -solver {} -weights {} -gpu {} 2>&1 | tee -a {}'.format(workdir, caffepath, solverfile, caffemodel, gpuid, log)

        else:
            raise IOError("Can't fine intial weight file: " + os.path.join(workdir, caffemodel))

    # Train from scratch. Not recommended for larger nets.
    else: 
        print "No caffemodel specified. Running {} from scratch!!".format(workdir)
        runstring  = 'cd {}; {} train -solver {} -gpu {} 2>&1 | tee -a {}'.format(workdir, caffepath, solverfile, gpuid, log)
    
    os.system(runstring)


def classify_from_datalayer(net, n_testinstances = 50, batch_size = 50, scorelayer = 'score'):
    """
    Runs the test-phase of a caffe net using the datalayer to load data. It runs through INPUT n_testinstances test instances in batches of INPUT batch_size.
    """
    scorelist = []
    for test_itt in tqdm.tqdm(range(n_testinstances//batch_size + 1)):
        scorelist.extend(list(copy.copy(net.blobs[scorelayer].data).astype(np.float)))
        net.forward()

    scorelist = scorelist[:n_testinstances]
    estlist = [np.argmax(s) for s in scorelist]  
    
    return estlist, scorelist

def cycle_runs(workdir, cycle_size = 1000, ncycles = 10, gpuid = 0, batch_size_test = 50, n_testinstances = 50, testnet_prototxt = 'testnet.prototxt', snapshot_prefix = 'snapshot', scorelayer = 'score'):
    """
    cycle_runs is a wrapper around run and classify_from_datalayer. After training the net for cycle_sizes iterations, it will run through the TEST net and store the results to disk.
    """
    for cycle in range(ncycles):
        run(workdir = workdir, nbr_iters = cycle_size, gpuid = gpuid)
        caffemodel = find_latest_caffemodel(workdir, snapshot_prefix = snapshot_prefix)
        net = load_model(workdir, caffemodel, gpuid = gpuid, net_prototxt = testnet_prototxt, phase = caffe.TEST)
        scorelist = classify_from_datalayer(net, n_testinstances = n_testinstances, batch_size = batch_size_test, scorelayer = scorelayer)
        bmt.psave(scorelist, osp.join(workdir, 'predictions_using_' + caffemodel +  '.p'))
        del net


def load_model(workdir, caffemodel, gpuid = 0, net_prototxt = 'net.prototxt', phase = caffe.TEST):
    """
    changes current directory to INPUT workdir and loads INPUT net_prototxt.
    """
    os.chdir(workdir)
    caffe.set_device(gpuid)
    caffe.set_mode_gpu()
    net = caffe.Net(net_prototxt, caffemodel, phase)
    net.forward() #one forward to initialize the net
    return net


def classify_from_imlist(im_list, net, transformer, batch_size, scorelayer = 'score', startlayer = 'conv1_1'):
    """
    classify_from_imlist classifies a list of images and returns estimated labels and scores. Only support classification nets (not FCNs).

    Takes
    im_list: list of images to classify (each stored as a numpy array).
    net: caffe net object
    transformer: transformer object as defined above.
    batch_size: batch size for the net.
    scorelayer: name of the score layer.
    startlayer: name of first convolutional layer.
    """

    scorelist = []
    for b in range(len(im_list) / batch_size + 1):
        for i in range(batch_size):
            pos = b * batch_size + i
            if pos < len(im_list):
                net.blobs['data'].data[i, :, :, :] = transformer.preprocess(im_list[pos])
        net.forward(start = startlayer)
        scorelist.extend(list(copy.copy(net.blobs[scorelayer].data).astype(np.float)))

    scorelist = scorelist[:len(im_list)]
    estlist = [np.argmax(s) for s in scorelist]  
    
    return estlist, scorelist


def find_latest_caffemodel(workdir, prefix = 'snapshot', suffix = 'caffemodel'):
    
    caffemodels = glob.glob(osp.join(workdir, '{}*.{}'.format(prefix, suffix)))
    if caffemodels:
        iter_ = [int(re.findall('.*_([0-9]*)[a-z]*', caffemodel)[0]) for caffemodel in caffemodels]
        return os.path.basename(caffemodels[np.argmax(iter_)]), np.max(iter_)
    else:
        return None, 0


def calculate_image_mean(imlist): 
    """
    Returns mean channel intensity across the images in imlist.
    NOTE: returns mean in in BGR order.
    """
    mean = np.zeros(3).astype(np.float32)
    for imname in imlist:
        im = np.asarray(Image.open(imname))
        if len(im.shape) == 2:
            im = np.mean(im, axis = 0)
            im = np.mean(im, axis = 0)
            mean = mean + [im, im, im]
        else:   
            im = im[:, :, ::-1] #change to BGR
            im = np.mean(im, axis = 0)
            im = np.mean(im, axis = 0)
            mean = mean + im
    mean /= len(imlist)
    print mean
    return mean


def pyparams_from_net(filepath):
    """
    read the python datalayer parameters from a net.prototxt
    """
    netobject = parse_net_prototxt(filepath)
    return eval(netobject.layer[0].python_param.param_str)


def parse_net_prototxt(filepath):
    """
    helper function to parse a net prototoxt
    """
    netobject = caffe.proto.caffe_pb2.NetParameter()
    with open(filepath, "r") as file:
        text_format.Merge(str(file.read()), netobject)
    return netobject


def find_best_iter(workdir, testtoken = 'predictions_using_snapshot_iter_*.caffemodel.p', accfcn = bmt.acc):
    
    iterlist = []
    bestacc = -1
    for testname in glob.glob(osp.join(workdir, testtoken)):
        
        [gtlist, estlist, scorelist] = bmt.pload(osp.join(workdir, testname))
        acc = accfcn(estlist, gtlist)
        iter_ = int(re.search('iter_([0-9]*).caffemodel.p', testname).group(1))

        iterlist.append((iter_, acc))
        if acc > bestacc:
            bestiter = iter_
            bestacc = acc

    return bestiter, iterlist


def clean_workdirs(workdirs):
    for workdir in workdirs:
        for file_ in glob.glob(os.path.join(workdir, 'snapshot*')):
            if os.path.isfile(file_):
                os.remove(file_)
        for file_ in glob.glob(os.path.join(workdir, 'predictions_*')):
            if os.path.isfile(file_):
                os.remove(file_)
        for file_ in glob.glob(os.path.join(workdir, '*.log')):
            if os.path.isfile(file_):
                os.remove(file_)
    for file_ in glob.glob(os.path.join(workdir, '*.testlog')):
            if os.path.isfile(file_):
                os.remove(file_)
