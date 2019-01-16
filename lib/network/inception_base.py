import tensorflow as tf
import tensorflow.contrib.slim as slim

from lib.utils.config import cfg

def inception_base(inputs, scope=None):
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
    # net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool3')

    net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
    net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
    net = slim.conv2d(net, 512, [3, 3], scope='conv4_3')

    if featuremap_scale*2 != cfg["ANCHOR_WIDTH"]:
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_3')
        net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool5')
        featuremap_scale *= 2

    net = _inception_module(net)
    net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool6')
    featuremap_scale *= 2

    return net, featuremap_scale

def _inception_module(net, module_index=None):
    """
    inception_module
    :param net:
    :param module_index:
    :return:
    """

    # Inception Module 0
    scope = "inception_module_0"
    with tf.variable_scope(scope):
        with tf.variable_scope(scope):
            branch_0 = slim.conv2d(net, 128, [1, 1], scope=scope + '/conv2d_b0_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 128, [1, 1], scope=scope + '/conv2d_b1_1x1')
            branch_1 = slim.conv2d(branch_1, 128, [5, 5], scope=scope + '/conv2d_b1_5x5')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 128, [1, 1], scope=scope + '/conv2d_b2_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope=scope + '/conv2d_b2_0_3x3')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope=scope + '/conv2d_b2_1_3x3')
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(net, [3, 3], stride=1, padding='SAME', scope=scope + '/avgPool_b3_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope=scope + '/conv2d_b3_1x1')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3, name=scope + '/concat_m1')

    # Inception Module 1
    scope = "inception_module_1"
    with tf.variable_scope(scope):
        with tf.variable_scope(scope):
            branch_0 = slim.conv2d(net, 128, [1, 1], scope=scope + '/conv2d_b0_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 128, [1, 1], scope=scope + '/conv2d_b1_1x1')
            branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope=scope + '/conv2d_b1_1x3')
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 128, [1, 1], scope=scope + '/conv2d_b2_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope=scope + '/conv2d_b2_0_3x3')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope=scope + '/conv2d_b2_1_3x3')
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(net, [3, 3], stride=1, padding='SAME', scope=scope + '/avgPool_b3_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [5, 5], scope=scope + '/conv2d_b3_1x5')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3, name=scope + '/concat_m2')

    return net