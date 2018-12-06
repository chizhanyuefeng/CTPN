import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow

from lib.utils.config import cfg
from lib.network.inception_model import inception_base
from lib.network.vgg_model import vgg_base

class CTPN(object):

    def __init__(self):
        pass

    def inference(self):

        inputs_img_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3])

        if cfg["BACKBONE"] == "InceptionNet":
            features = inception_base(inputs_img_tensor)
        elif cfg["BACKBONE"] == "VggNet":
            features = inception_base(inputs_img_tensor)
        else:
            assert 0, "error: backbone {} is not support!".format(cfg["BACKBONE"])

        features = slim.conv2d(features, 512, [3, 3], )



    def _proposal_layer(self):
        pass

    def _anchor_layer(self):
        pass

    def loss(self):
        pass

if __name__ == "__main__":
    pretrain_model_path = './models/pretrain_model/inception_v4.ckpt'
    reader = pywrap_tensorflow.NewCheckpointReader(pretrain_model_path)
    keys = reader.get_variable_to_shape_map().keys()
    print(keys)