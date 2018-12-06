import tensorflow.contrib.slim as slim
from lib.utils.config import cfg

def vgg_base(inputs, scope=None):
    featuremap_scale = 1

    net = slim.conv2d(inputs, 64, [3, 3], scope='conv1_1')
    net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
    net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool1')
    featuremap_scale *= 2

    net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
    net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
    net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool2')
    featuremap_scale *= 2

    net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
    net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
    net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
    net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool3')
    featuremap_scale *= 2

    net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
    net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
    net = slim.conv2d(net, 512, [3, 3], scope='conv4_3')

    if featuremap_scale != cfg["ANCHOR_WIDTH"]:
        net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool4')
        featuremap_scale *= 2

        net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_3')

    return net, featuremap_scale
