
def conv_bn(bottom, nout, ks = 3, stride=1, pad = 0, learn = True):
    if learn:
        param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
    else:
        param = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
            num_output=nout, pad=pad, param = param, weight_filler=dict(type="msra"), bias_filler=dict(type="constant"))
    bn = L.BatchNorm(conv)
    lrn = L.LRN(bn)
    return conv, bn, lrn


def residual_standard_unit(n, nout, s, newdepth = False):
    """
    This creates the "standard unit" shown on the left side of Figure 5.
    """
    bottom = n.__dict__['tops'][n.__dict__['tops'].keys()[-1]] #find the last layer in netspec
    stride = 2 if newdepth else 1

    n[s + 'conv1'], n[s + 'bn1'], n[s + 'lrn1'] = conv_bn(bottom, ks = 3, stride = stride, nout = nout, pad = 1)
    n[s + 'relu1'] = L.ReLU(s + 'lrn1', in_place=True)
    n[s + 'conv2'], n[s + 'bn2'], n[s + 'lrn2'] = conv_bn(s + 'relu1', ks = 3, stride = 1, nout = nout, pad = 1)
   
    if newdepth: 
        n[s + 'conv_expand'], n[s + 'bn_expand'], n[s + 'lrn_expand'] = conv_bn(bottom, ks = 1, stride = 2, nout = nout, pad = 0)
        n[s + 'sum'] = L.Eltwise(s + 'lrn2', s + 'lrn_expand')
    else:
        n[s + 'sum'] = L.Eltwise(s + 'lrn2', bottom)

    n[s + 'relu2'] = L.ReLU(s + 'sum', in_place=True)
    

def residual_bottleneck_unit(n, nout, s, newdepth = False):
    """
    This creates the "standard unit" shown on the left side of Figure 5.
    """
    
    bottom = n.__dict__['tops'].keys()[-1] #find the last layer in netspec
    stride = 2 if newdepth and nout > 64 else 1

    n[s + 'conv1'], n[s + 'bn1'], n[s + 'lrn1'] = conv_bn(n[bottom], ks = 1, stride = stride, nout = nout, pad = 0)
    n[s + 'relu1'] = L.ReLU(n[s + 'lrn1'], in_place=True)
    n[s + 'conv2'], n[s + 'bn2'], n[s + 'lrn2'] = conv_bn(n[s + 'relu1'], ks = 3, stride = 1, nout = nout, pad = 1)
    n[s + 'relu2'] = L.ReLU(n[s + 'lrn2'], in_place=True)
    n[s + 'conv3'], n[s + 'bn3'], n[s + 'lrn3'] = conv_bn(n[s + 'relu2'], ks = 1, stride = 1, nout = nout * 4, pad = 0)
   
    if newdepth: 
        n[s + 'conv_expand'], n[s + 'bn_expand'], n[s + 'lrn_expand'] = conv_bn(n[bottom], ks = 1, stride = stride, nout = nout * 4, pad = 0)
        n[s + 'sum'] = L.Eltwise(n[s + 'lrn3'], n[s + 'lrn_expand'])
    else:
        n[s + 'sum'] = L.Eltwise(n[s + 'lrn3'], n[bottom])

    n[s + 'relu3'] = L.ReLU(n[s + 'sum'], in_place=True)

def residual_net(total_depth, data_layer_params, num_classes = 1000, acclayer = True):
    """
    Generates nets from "Deep Residual Learning for Image Recognition". Nets follow architectures outlined in Table 1. 
    """
    # figure out network structure
    net_defs = {
        18:([2, 2, 2, 2], "standard"),
        34:([3, 4, 6, 3], "standard"),
        50:([3, 4, 6, 3], "bottleneck"),
        101:([3, 4, 23, 3], "bottleneck"),
        152:([3, 8, 36, 3], "bottleneck"),
    }
    assert total_depth in net_defs.keys(), "net of depth:{} not defined".format(total_depth)

    nunits_list, unit_type = net_defs[total_depth] # nunits_list a list of integers indicating the number of layers in each depth.
    nouts = [64, 128, 256, 512] # same for all nets

    # setup the first couple of layers
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = 'beijbom_caffe_data_layers', layer = 'ImageNetDataLayer',
                ntop = 2, param_str=str(data_layer_params))
    n.conv1, n.bn1, n.lrn1 = conv_bn(n.data, ks = 7, stride = 2, nout = 64, pad = 3)
    n.relu1 = L.ReLU(n.lrn1, in_place=True)
    n.pool1 = L.Pooling(n.relu1, stride = 2, kernel_size = 3)
    
    # make the convolutional body
    for nout, nunits in zip(nouts, nunits_list): # for each depth and nunits
        for unit in range(1, nunits + 1): # for each unit. Enumerate from 1.
            s = str(nout) + '_' + str(unit) + '_' # layer name prefix
            if unit_type == "standard":
                residual_standard_unit(n, nout, s, newdepth = unit is 1 and nout > 64)
            else:
                residual_bottleneck_unit(n, nout, s, newdepth = unit is 1)
                
    # add the end layers                    
    n.global_pool = L.Pooling(n.__dict__['tops'][n.__dict__['tops'].keys()[-1]], pooling_param = dict(pool = 1, global_pooling = True))
    n.score = L.InnerProduct(n.global_pool, num_output = num_classes,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    if acclayer:
        n.accuracy = L.Accuracy(n.score, n.label)

    return n