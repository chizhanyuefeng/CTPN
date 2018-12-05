import tensorflow.contrib.slim as slim

# class VggModel(object):

def VggModel(inputs):
    net = slim.conv2d(inputs, 64, [3, 3], scope='conv1_1')
    net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
    net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool1')

    net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
    net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
    net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool2')

    net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
    net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
    net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
    net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool3')

    net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
    net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
    net = slim.conv2d(net, 512, [3, 3], scope='conv4_3')
    net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool4')

    net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
    net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
    net = slim.conv2d(net, 512, [3, 3], scope='conv5_3')

    return net
