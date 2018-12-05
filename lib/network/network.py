import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow

# from lib.utils.config import cfg

class CTPN(object):

    def __init__(self):
        pass

    def _backbone(self):
        pass

    def inference(self):
        pass

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