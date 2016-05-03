from caffe import layers as L, params as P

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1, learn=True):
    if learn:
        param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group, param=param)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, learn = True):
    
    if learn:
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)]

    fc = L.InnerProduct(bottom, num_output=nout, param=param)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def caffenet(n, nclasses, acclayer=False, softmax = False, learn=True):
    # first input 'n', must have layer member: n.data

    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, learn = learn)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, learn = learn)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, learn = learn)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, learn = learn)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, learn = learn)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, learn = learn)
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 4096, learn = learn)
    n.drop7 = L.Dropout(n.relu7, in_place=True)

    n.score = L.InnerProduct(n.drop7, num_output=nclasses, param=[dict(lr_mult=5, decay_mult=1), dict(lr_mult=10, decay_mult=0)])

    if softmax:
        n.loss = L.SoftmaxWithLoss(n.score, n.label)

    if acclayer:
        n.acc = L.Accuracy(n.score, n.label)
    return n