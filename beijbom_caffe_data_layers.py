# ==============================================================================
# ==============================================================================
# =========================== IMAGENET LAYER ===================================
# ==============================================================================
# ==============================================================================

class ImageNetDataLayer(caffe.Layer):

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===
        params = eval(self.param_str)
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'imlistfile' in params.keys(), 'Params must include imlistfile.'
        assert 'imdictfile' in params.keys(), 'Params must include imdictfile.'
        assert 'crop_size' in params.keys(), 'Params must include crop_size.'
        assert 'im_mean' in params.keys(), 'Params must include im_mean.'
        
        self.batch_size = params['batch_size']

        # === set up thread and batch advancer ===
        self.thread_result = {}
        self.thread = None
        self.batch_advancer = ImageNetPatchBatchAdvancer(self.thread_result, params)
        self.dispatch_worker()

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, params['crop_size'], params['crop_size'])
        top[1].reshape(self.batch_size, 1)

    def reshape(self, bottom, top):
        """ happens during setup """
        pass

    def forward(self, bottom, top):
        if self.thread is not None:
            self.join_worker()

        for top_index, name in zip(range(len(top)), self.top_names):
            for i in range(self.batch_size):
                top[top_index].data[i, ...] = self.thread_result[name][i] 
        self.dispatch_worker()

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None


    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass


class ImageNetPatchBatchAdvancer():
    """
    The ImageNetPatchBatchAdvancer is a helper class to ImageNetDataLayer. It is called asychronosly and prepares the tops.
    """
    def __init__(self, result, params):
        self._cur = 0
        self.result = result
        self.params = params
        self.imlist = [line.rstrip('\n') for line in open(params['imlistfile'])]
        with open(params['imdictfile']) as f:
            self.imdict = json.load(f)
        self.transformer = TransformerWrapper(params['im_mean'])
        shuffle(self.imlist)

        print "DataLayer initialized with {} images".format(len(self.imlist))

    def __call__(self):
        self.result['data'] = []
        self.result['label'] = []

        if self._cur + self.params['batch_size'] >= len(self.imlist):
            self._cur = 0
            shuffle(self.imlist)
        
        # Loop over each image
        for imname in self.imlist[self._cur : self._cur + self.params['batch_size']]:
            self._cur += 1

            im = Image.open(imname) # Load image
            im = im.convert("RGB") # make sure it's 3 channels
            im = self.scale_augment(im) # scale augmentation

			# random crop
            (width, height) = im.size
            left = np.random.choice(width - 224)
            upper = np.random.choice(height - 224)
            im = im.crop((left, upper, left + 224, upper + 224))
            im = np.asarray(im)
           
			# random flip 
            flip = np.random.choice(2)*2-1
            im = im[:, ::flip, :]
                
            self.result['data'].append(self.transformer(im)            )
            self.result['label'].append(self.imdict[os.path.basename(imname)])

    def scale_augment(self, im):
        (width, height) = im.size
        width, height = float(width), float(height)
        if width <= height:
            wh_ratio = height / width
            new_width = int(np.random.choice(480-256) + 256)
            im = im.resize((new_width, int(new_width * wh_ratio)))
        else:
            hw_ratio = width / height
            new_height = int(np.random.choice(480-256) + 256)
            im = im.resize((int(new_height * hw_ratio), new_height))
        return im
    
class TransformerWrapper(Transformer):
    def __init__(self, mean):
        Transformer.__init__(self, mean)
    def __call__(self, im):
        return self.preprocess(im)


# ==============================================================================
# ==============================================================================
# ============================== REGRESSION LAYER ==============================
# ==============================================================================
# ==============================================================================

class RandomPointRegressionDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # === Read input parameters ===
        params = eval(self.param_str)
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'imlistfile' in params.keys(), 'Params must include imlistfile.'
        assert 'imdictfile' in params.keys(), 'Params must include imdictfile.'
        assert 'im_scale' in params.keys(), 'Params must include im_scale.'
        assert 'im_mean' in params.keys(), 'Params must include im_mean.'

        self.t0 = 0
        self.t1 = 0
        self.batch_size = params['batch_size']
        self.im_shape = params['im_shape']
        self.nclasses = params['nclasses']
        imlist = [line.rstrip('\n') for line in open(params['imlistfile'])]
        with open(params['imdictfile']) as f:
            imdict = json.load(f)

        transformer = TransformerWrapper()
        transformer.set_mean(params['im_mean'])
        transformer.set_scale(params['im_scale'])

        print "Setting up RandomPointRegressionDataLayer with batch size:{}".format(self.batch_size)

        # === set up thread and batch advancer ===
        self.thread_result = {}
        self.thread = None
        self.batch_advancer = RegressionBatchAdvancer(self.thread_result, self.batch_size, imlist, imdict, transformer, self.nclasses, self.im_shape)
        self.dispatch_worker()

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, self.im_shape[0], self.im_shape[1])
        top[1].reshape(self.batch_size, self.nclasses)

    def reshape(self, bottom, top):
        """ happens during setup """
        top[0].reshape(self.batch_size, 3, self.im_shape[0], self.im_shape[1])
        top[1].reshape(self.batch_size, self.nclasses)
        #pass

    def forward(self, bottom, top):
        # print time.clock() - self.t0, "seconds since last call to forward."
        if self.thread is not None:
            self.t1 = timer()
            self.join_worker()
            # print "Waited ", timer() - self.t1, "seconds for join."

        for top_index, name in zip(range(len(top)), self.top_names):
            for i in range(self.batch_size):
                top[top_index].data[i, ...] = self.thread_result[name][i] 
        self.t0 = time.clock()
        self.dispatch_worker()

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None


    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass


class RegressionBatchAdvancer():
    """
    The RegressionBatchAdvancer is a helper class to RandomPointRegressionDataLayer. It is called asychronosly and prepares the tops.
    """
    def __init__(self, result, batch_size, imlist, imdict, transformer, nclasses, im_shape):
        self.result = result
        self.batch_size = batch_size
        self.imlist = imlist
        self.imdict = imdict
        self.transformer = transformer
        self._cur = 0
        self.nclasses = nclasses
        self.im_shape = im_shape
        shuffle(self.imlist)

        print "RegressionBatchAdvancer is initialized with {} images".format(len(imlist))

    def __call__(self):
        
        t0 = timer()
        self.result['data'] = []
        self.result['label'] = []

        if self._cur == len(self.imlist):
            self._cur = 0
            shuffle(self.imlist)
        
        imname = self.imlist[self._cur]

        # Load image
        im = np.asarray(Image.open(imname))
        im = scipy.misc.imresize(im, self.im_shape)
        point_anns = self.imdict[os.path.basename(imname)][0]

        class_hist = np.zeros(self.nclasses).astype(np.float32)
        for (row, col, label) in point_anns:
            class_hist[label] += 1
        class_hist /= len(point_anns)

                
        self.result['data'].append(self.transformer.preprocess(im))
        self.result['label'].append(class_hist)
        self._cur += 1
        # print "loaded image {} in {} secs.".format(self._cur, timer() - t0)


# ==============================================================================
# ==============================================================================
# ============================== MULTILABEL LAYER ==============================
# ==============================================================================
# ==============================================================================

class RandomPointMultiLabelDataLayer(caffe.Layer):

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===
        params = eval(self.param_str)
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'imlistfile' in params.keys(), 'Params must include imlistfile.'
        assert 'imdictfile' in params.keys(), 'Params must include imdictfile.'
        assert 'im_scale' in params.keys(), 'Params must include im_scale.'
        assert 'im_mean' in params.keys(), 'Params must include im_mean.'

        self.t0 = 0
        self.t1 = 0
        self.batch_size = params['batch_size']
        self.nclasses = params['nclasses']
        self.im_shape = params['im_shape']
        imlist = [line.rstrip('\n') for line in open(params['imlistfile'])]
        with open(params['imdictfile']) as f:
            imdict = json.load(f)

        transformer = TransformerWrapper()
        transformer.set_mean(params['im_mean'])
        transformer.set_scale(params['im_scale'])

        print "Setting up RandomPointRegressionDataLayer with batch size:{}".format(self.batch_size)

        # === set up thread and batch advancer ===
        self.thread_result = {}
        self.thread = None
        self.batch_advancer = MultiLabelBatchAdvancer(self.thread_result, self.batch_size, imlist, imdict, transformer, self.nclasses, self.im_shape)
        self.dispatch_worker()

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, self.im_shape[0], self.im_shape[1])
        top[1].reshape(self.batch_size, self.nclasses)

    def reshape(self, bottom, top):
        """ happens during setup """
        pass

    def forward(self, bottom, top):
        # print time.clock() - self.t0, "seconds since last call to forward."
        if self.thread is not None:
            self.t1 = timer()
            self.join_worker()
            # print "Waited ", timer() - self.t1, "seconds for join."

        for top_index, name in zip(range(len(top)), self.top_names):
            for i in range(self.batch_size):
                top[top_index].data[i, ...] = self.thread_result[name][i] 
        self.t0 = time.clock()
        self.dispatch_worker()

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None


    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass


class MultiLabelBatchAdvancer():
    """
    The MultiLabelBatchAdvancer is a helper class to RandomPointRegressionDataLayer. It is called asychronosly and prepares the tops.
    """
    def __init__(self, result, batch_size, imlist, imdict, transformer, nclasses, im_shape):
        self.result = result
        self.batch_size = batch_size
        self.imlist = imlist
        self.imdict = imdict
        self.transformer = transformer
        self._cur = 0
        self.nclasses = nclasses
        self.im_shape = im_shape
        shuffle(self.imlist)

        print "MultiLabelBatchAdvancer is initialized with {} images".format(len(imlist))

    def __call__(self):
        
        t0 = timer()
        self.result['data'] = []
        self.result['label'] = []

        if self._cur == len(self.imlist):
            self._cur = 0
            shuffle(self.imlist)
        
        imname = self.imlist[self._cur]

        # Load image
        im = np.asarray(Image.open(imname))
        im = scipy.misc.imresize(im, self.im_shape)
        point_anns = self.imdict[os.path.basename(imname)][0]

        class_in_image = np.zeros(self.nclasses).astype(np.float32)
        for (row, col, label) in point_anns:
            class_in_image[label] = 1

                
        self.result['data'].append(self.transformer.preprocess(im))
        self.result['label'].append(class_in_image)
        self._cur += 1
        # print "loaded image {} in {} secs.".format(self._cur, timer() - t0)
