import glob, os, Image, math, colorsys, scipy, confmatrix, beijbom_pytools, caffe, re, lmdb, sys
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from copy import deepcopy, copy
import cPickle as pickle
from tqdm import tqdm

"""
beijbom caffe tools (bct) contains a ton of classes and wrappers for caffe.
"""


class Transformer:
    """
    Transformer is a class for preprocessing and deprocessing images according to the vgg16 pre-processing paradigm (scaling and mean subtraction.)
    """

    def __init__(self):
        self.mean = []
        self.scale = []

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
        for channel, channel_mean in list(enumerate(self.mean)):
            im[:, :, channel] = im[:, :, channel] - np.ones(im.shape[:2], dtype=np.uint8) * channel_mean
    
        im = im * self.scale
        im = im.transpose(2, 0, 1)
        
        return im

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im = im / self.scale

        for channel, channel_mean in list(enumerate(self.mean)):
            im[:, :, channel] = im[:, :, channel] + np.ones(im.shape[:2], dtype=np.uint8) * channel_mean
    
        return np.uint8(im)



class CaffeSolver:
    """
    Caffesolver is a class for creating a solver.prototxt file. It sets default values and can export a solver parameter file.
    """

    def __init__(self, net_prototxt_path = "net.prototxt", debug = False):
        
        self.sp = {}

        # critical:
        self.sp['base_lr'] = '1e-10'
        self.sp['momentum'] = '0.99'
        
        # speed:
        self.sp['test_iter'] = '100'
        self.sp['test_interval'] = '250'
        
        # looks:
        self.sp['display'] = '25'
        self.sp['snapshot'] = '2500'
        self.sp['snapshot_prefix'] = '"snapshot"'
        
        # learning rate policy
        self.sp['lr_policy'] = '"step"'
        self.sp['stepsize'] = '100000'

        # important, but rare:
        self.sp['gamma'] = '0.1'
        self.sp['weight_decay'] = '0.0005'
        self.sp['net'] = '"' + net_prototxt_path + '"'

        # pretty much never change these.
        self.sp['max_iter'] = '100000'
        self.sp['test_initialization'] = 'false'
        self.sp['average_loss'] = '1' # this has to do with the display.
        self.sp['iter_size'] = '1' #this is for accumulating gradients

        if (debug):
            self.sp['max_iter'] = '12'
            self.sp['test_iter'] = '1'
            self.sp['test_interval'] = '4'
            self.sp['display'] = '1'

    def add_from_file(self, filepath):
        """
        Reads a caffe solver prototxt file and updates the Caffesolver instance parameters.
        """
        with open(filepath, 'r') as f:
            for line in f:
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


def classify_list(im_list, net, transformer, batch_size, get_scores = False):
    nbatches = int(math.ceil(float(len(im_list)) / batch_size))
    pos = -1
    gt = []
    for b in range(nbatches):
        for i in range(batch_size):
            pos += 1
            if pos < len(im_list):
                net.blobs['data'].data[i, :, :, :] = transformer.preprocess(im_list[pos])
        net.forward(start = 'conv1_1')
        for i in range(batch_size):
            if get_scores:
                gt.append(net.blobs['fc8_mcr'].data[i,:].flatten())
            else:
                gt.append(np.argmax(net.blobs['fc8_mcr'].data[i,:].flatten()))

    return(gt[:len(im_list)])


def sac(im, net, transformer, target_size = [1024, 1024], padcolor = [126, 148, 137], get_scores = False):
    """
    sac (slice and classify) will slice the input image, feed each piece to the
    caffe net object, and then stitch the output back together to an output ground 
    truth image
    """
    input_size = im.shape[:2]
    (imlist, ncells) = beijbom_pytools.slice_image(im, target_size = target_size, padcolor = padcolor)
    imcounter = -1
    for row in range(ncells[0]):
        for col in range(ncells[1]):
            imcounter += 1
            net.blobs['data'].data[...] = transformer.preprocess(imlist[imcounter])
            net.forward(start = 'conv1_1')
            if get_scores:
                slice_gt = np.float32(np.squeeze(net.blobs['upscore'].data.transpose(2, 3, 1, 0)))
            else:    
                slice_gt = np.uint8(np.argmax(np.squeeze(net.blobs['upscore'].data.transpose(2, 3, 1, 0)), axis = 2))
            
            if col == 0:
                rowgt = deepcopy(slice_gt)
            else:
                rowgt = np.concatenate((rowgt, slice_gt), axis = 1) #build one row (along the columns)

        if row == 0:
            outgt = rowgt
        else:
            outgt = np.concatenate((outgt, rowgt), axis = 0) # concatenate the rows
    return outgt


def run(workdir = None, weights = 'weights.caffemodel', solver = 'solver.prototxt', log = 'train.log', snapshot_prefix = 'snapshot', caffepath = '/home/beijbom/cc/build/tools/caffe', restart = False):
    snapshots = glob.glob("/{}/{}*.solverstate".format(workdir, snapshot_prefix))
    if snapshots:
        _iter = [int(f[f.index('iter_')+5:f.index('.')]) for f in snapshots]
        latest_snapshot = snapshots[np.argmax(_iter)]
    if snapshots and not(restart):
        print "Running from iter {}.".format(np.max(_iter))
        runstring  = 'cd {}; {} train -solver {} -snapshot {} 2>&1 | tee -a {}'.format(workdir, caffepath, solver, latest_snapshot, log)
    elif(weights):
        if(os.path.isfile(os.path.join(workdir, weights))):
            print "Fine tuning from {}.".format(os.path.join(workdir, weights))
            runstring  = 'cd {}; {} train -solver {} -weights {} 2>&1 | tee -a {}'.format(workdir, caffepath, solver, weights, log)
        else:
            raise IOError("Can't fine intial weight file: " + os.path.join(workdir, weights))
    else:
        print "No weights specified. Running from scratch!!"
        runstring  = 'cd {}; {} train -solver {} 2>&1 | tee -a {}'.format(workdir, caffepath, solver, log)
    os.system(runstring)


def cycle_runs_debug(run_params, test_params):
    run_params = deepcopy(run_params)
    test_params = deepcopy(test_params)
    print 'Running tests...'
    for test_param in test_params:
        test_param['n_testinstances'] = 5
    cycle_sizes = np.ones(len(test_params), dtype = np.int) * 4 #4 iterations
    cycle_runs(run_params, test_params, cycle_sizes, 1)
    print 'Run test OK. Cleaning up.'
    for run_param in run_params:
        for file_ in glob.glob(os.path.join(run_param['workdir'], 'snapshot*')):
            os.remove(file_)
        for file_ in glob.glob(os.path.join(run_param['workdir'], 'predictions_on*')):
            os.remove(file_)
        os.remove(os.path.join(run_param['workdir'], 'train.log'))
    for test_param in test_params:
	test_param['n_testinstances'] = None

def cycle_runs(run_params, test_params, cycle_sizes, ncycles):
    """
    run_params is a list of dictionaries. Each dictionary must contain values for the following key:
    ['workdir'] : full path to the directory with the solver prototxt is located [str]
    """
    run_defaults = {'solver':'solver.prototxt','log':'train.log','snapshot_prefix':'snapshot','caffepath':'/home/beijbom/cc/build/tools/caffe', 'restart': False}
    test_defaults = {'caffemodel':None, 'snapshot_prefix':'snapshot', 'save':True, 'ignore_label':255, 'n_testinstances':None}
    for cycle in range(ncycles):
        for (cycle_size, params, tparams) in zip(cycle_sizes, run_params, test_params):
            # add defaults to run_parameter dict
            for key in list(set(run_defaults) - set(params)):
                params[key] = run_defaults[key]
            # find the current iteration
            snapshots = glob.glob(os.path.join(params['workdir'], "{}*.solverstate".format(params['snapshot_prefix'])))
            if snapshots:
                _iter = [int(f[f.index('iter_')+5:f.index('.')]) for f in snapshots]
                max_iter = np.max(_iter) + cycle_size - 1
            else:
                max_iter = cycle_size - 1

            # update solver with new max_iter parameter
            solver = CaffeSolver()
            solver.add_from_file(os.path.join(params['workdir'], params['solver']))
            solver.sp['max_iter'] = str(max_iter)
            solver.sp['snapshot'] = str(1000000) #disable this, don't need it.
            solver.write(os.path.join(params['workdir'], params['solver']))
            print "Running {} from {} to {} itts.".format(params['workdir'], max_iter-cycle_size+1, max_iter+1)
            run(**params)

            # classify all *net.prototxt in workdir
            testnets = glob.glob(os.path.join(params['workdir'], '*net.prototxt'))
            for testnet in testnets:
                
                # add params to tparams dict
                tparams['workdir'] = params['workdir'] #assuming the same workdir
                tparams['net_prototxt'] = testnet
                for key in list(set(test_defaults) - set(tparams)):
                    tparams[key] = test_defaults[key]

                classify(**tparams)


def load_model(rundir, snapshot):
    os.chdir(rundir)
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net('net.prototxt', snapshot, caffe.TEST)
    net.forward() 
    return 4


def nbr_lines(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def classify(workdir, scorelayer, caffemodel = None, snapshot_prefix = 'snapshot', net_prototxt = 'net.prototxt', save = False, ignore_label = 255, n_testinstances = None):
    """
    classify will run a trained net on a testset and return the ground truth, estimated labels and the score vectors.

    Takes
    workdir: directory where net_prototxt lives. All paths must be given relative to this directory
    scorelayer: name of layer to extract the scores from
    caffemodel: name of the stored caffemodel. If not given, the most recent model in workdir will be used.
    snapshot_prefix: snapshot prefix used by caffe. This is only required if caffemodel = None.
    net_prorotxt: name of the net prototxt to use. 
    save: wheather or not to save the output to disk or not.
    ignore_label: Ignores all labels where the gt = ignore_label. Used mostly for FCN models. 
    n_testinstances: Number of instances in the test list. If not given, this will be extracted automatically from the testlist or LMDB. Note that there is no support for 

    Gives
    (gt, est, scores): tuple with ground truth (as list), estimated labels (as list), scores as list of np arrays

    """

    os.chdir(workdir)
    if caffemodel is None:
        ## find and load latest model:
        caffemodels = glob.glob("{}*.caffemodel".format(snapshot_prefix))
        if caffemodels:
            _iter = [int(f[f.index('iter_')+5:f.index('.')]) for f in caffemodels]
            caffemodel = caffemodels[np.argmax(_iter)]
        else:
            print "Can't find a trained model in " + workdir + "."
            return None

    # find batch size from prototxt
    with open (net_prototxt, "r") as myfile:
        net_definition_str = myfile.read()
    batch_size = int(re.findall('(?<=batch_size: )[0-9]*', net_definition_str)[-1])

    # find the number of instances in test set:
    test_file = os.path.join('./../', re.findall("(?<=source: ../../)[a-z0-9]*.[a-z]*", net_definition_str)[-1])
    if n_testinstances is None:
        if test_file.find('lmdb') > -1:
            in_db = lmdb.open(test_file)
            n_testinstances = int(in_db.stat()['entries'])
        else: 
            n_testinstances = nbr_lines(test_file)

    print "Classifying " + test_file + " from "+ os.path.join(workdir, net_prototxt) + " using " + caffemodel + " with bs:" + str(batch_size) + ", and " + str(n_testinstances) + " total instances."
    sys.stdout.flush()

    # setup caffe and load model
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(net_prototxt, caffemodel, caffe.TEST)
    net.forward() #call once for allocation

    # classify. All the reshaping has to do with being able to handling both FCN and classification nets.
    gtlist = []
    scorelist = []
    for test_itt in tqdm(range(n_testinstances//batch_size + 1)):
        gt = copy(net.blobs['label'].data.transpose(0, 2, 3, 1)).astype(np.uint8)
        scores = copy(net.blobs[scorelayer].data.transpose(0, 2, 3, 1)).astype(np.float)
        nclasses = scores.shape[3]
        gt = np.repeat(gt, nclasses, axis = 3)
        keepind = gt != ignore_label
        scores = scores[keepind]
        scorelist.extend(list(np.reshape(scores, [scores.shape[0]/nclasses, nclasses])))
        gt = gt[keepind]
        gtlist.extend(list(np.reshape(gt, [gt.shape[0]/nclasses, nclasses])[:, 0]))
        net.forward()

    #If the net is not a FCN we need to cut of the lists (sicne the last iteration may be looping around)
    if net.blobs[scorelayer].data.shape[2] == 1: 
        gtlist = gtlist[:n_testinstances]
        scorelist = scorelist[:n_testinstances]

    # for convenience, include estimated labels
    estlist = [np.argmax(s) for s in scorelist]
    if (save):
        pickle.dump((gtlist, estlist, scorelist), open(os.path.join(workdir, 'predictions_on_' + test_file[5:] + '_using_' + caffemodel +  '.p'), 'wb'))

    return (gtlist, estlist, scorelist)
