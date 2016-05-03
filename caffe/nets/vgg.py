import caffe
from caffe import layers as L, params as P

def vgg_core(n, learn = True):

    # first input 'n', must have layer member: n.data
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, learn = learn)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, learn = learn)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, learn = learn)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, learn = learn)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, learn = learn)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, learn = learn)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, learn = learn)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, learn = learn)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, learn = learn)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, learn = learn)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, learn = learn)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, learn = learn)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, learn = learn)
    n.pool5 = max_pool(n.relu5_3)

    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, learn = learn)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

    n.fc7, n.relu7 = fc_relu(n.fc6, 4096, learn = learn)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    return n



def fc_relu(bottom, nout, learn = True):
    
    if learn:
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param=[dict(lr_mult=0, decay_mult=1), dict(lr_mult=0, decay_mult=0)]

    fc = L.InnerProduct(bottom, num_output=nout, param=param)
    return fc, L.ReLU(fc, in_place=True)


def conv_relu(bottom, nout, ks=3, stride=1, pad=1, learn=True):
    if learn:
        param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]

    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, param=param)
    return conv, L.ReLU(conv, in_place=True)


def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)