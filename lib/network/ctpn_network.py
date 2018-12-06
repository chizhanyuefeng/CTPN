import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow

from lib.utils.config import cfg
from lib.network.inception_base import inception_base
from lib.network.vgg_base import vgg_base

class CTPN(object):

    def __init__(self):
        pass

    def inference(self):

        proposal_predicted, proposal_cls_score = self.__ctpn_base()

        proposal_cls_score_shape = tf.shape(proposal_cls_score)
        proposal_cls_score = tf.reshape(proposal_predicted, [-1, cfg["CLASSES_NUM"]])
        proposal_cls_prob = tf.reshape(tf.nn.softmax(proposal_cls_score), proposal_cls_score_shape)


    def __proposal_layer(self):
        pass

    def __anchor_layer(self):
        pass

    def __ctpn_base(self):
        stddev = 0.01
        weight_decay = cfg["TRAIN"]["WEIGHT_DECAY"]

        with tf.variable_scope("CTPN_Network"):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(0.0, stddev=stddev),
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                activation_fn=tf.nn.relu):
                inputs_img_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3])

                if cfg["BACKBONE"] == "InceptionNet":
                    features = inception_base(inputs_img_tensor)
                elif cfg["BACKBONE"] == "VggNet":
                    features = inception_base(inputs_img_tensor)
                else:
                    assert 0, "error: backbone {} is not support!".format(cfg["BACKBONE"])

                features = slim.conv2d(features, 512, [3, 3], scope='rpn_conv_3x3')
                features_channel = tf.shape(features)[-1]
                features = self.__Bilstm(features, features_channel, 128, features_channel)

                # proposal_predicted shape = [1, h, w, A*cfg["CLASSES_NUM"]]
                proposal_predicted = slim.conv2d(features, len(cfg["ANCHOR_HEIGHT"]) * cfg["CLASSES_NUM"], [1, 1], scope='bbox_conv_1x1')
                # proposal_cls_score shape = [1, h, w, A*cfg["CLASSES_NUM"]]
                proposal_cls_score = slim.conv2d(features, len(cfg["ANCHOR_HEIGHT"]) * cfg["CLASSES_NUM"], [1, 1], scope='bbox_conv_1x1')

        return proposal_predicted, proposal_cls_score

    def __Bilstm(self, input, d_i, d_h, d_o, name="Bilstm"):
        """
        双向rnn
        :param input:
        :param d_i: 512 每个timestep 携带信息
        :param d_h: 128 一层rnn
        :param d_o: 512 最后rrn层输出
        :param name:
        :param trainable:
        :return:
        """

        with tf.variable_scope(name):
            shape = tf.shape(input)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            img = tf.reshape(input, [N * H, W, C])
            # print('dididididi',d_i)
            img.set_shape([None, None, d_i])

            lstm_fw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)

            lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, img, dtype=tf.float32)
            lstm_out = tf.concat(lstm_out, axis=-1)

            lstm_out = tf.reshape(lstm_out, [N * H * W, 2*d_h])

            outputs = slim.fully_connected(lstm_out, d_o)
            outputs = tf.reshape(outputs, [N, H, W, d_o])

            return outputs


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