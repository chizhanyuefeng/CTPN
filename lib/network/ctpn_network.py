import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow

from lib.utils.config import cfg
from lib.network.inception_base import inception_base
from lib.network.vgg_base import vgg_base
from lib.rpn_layer.generate_anchors import generate_anchors
from lib.rpn_layer.anchor_target_layer_tf import anchor_target_layer
from lib.rpn_layer.proposal_layer_tf import proposal_layer

class CTPN(object):

    def __init__(self):
        pass

    def inference(self):
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.proposal_predicted, proposal_cls_score = self.__ctpn_base()

        proposal_cls_score_shape = tf.shape(proposal_cls_score)
        self.proposal_cls_score = tf.reshape(proposal_cls_score, [-1, cfg["CLASSES_NUM"]])
        self.proposal_cls_prob = tf.reshape(tf.nn.softmax(self.proposal_cls_score), proposal_cls_score_shape)

    def __proposal_layer(self):
        """
        回归proposal框
        :param input: shape = ['rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info']
        :param _feat_stride: [16, ]
        :param anchor_scales: [16]
        :param cfg_key: "TEST"
        :param name:
        :return:
        """

        # input[0] shape is (1, H, W, Ax2)
        # rpn_rois <- (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        with tf.variable_scope("proposal_layer"):
            blob, bbox_delta = tf.py_func(proposal_layer,
                                          [self.proposal_cls_prob, self.proposal_predicted, self.im_info, "TEST", [cfg["ANCHOR_WIDTH"], ], [cfg["ANCHOR_WIDTH"]]],
                                          [tf.float32, tf.float32])

            rpn_rois = tf.reshape(blob, [-1, 5], name='rpn_rois')  # shape is (1 x H x W x A, 2)
            rpn_targets = tf.convert_to_tensor(bbox_delta, name='rpn_targets')  # shape is (1 x H x W x A, 4)
            return rpn_rois, rpn_targets


    def bbox_transform_inv(self):
        pass

    def clip_boxes(self):
        pass


    def __anchor_layer(self):
        with tf.variable_scope(name) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
                tf.py_func(anchor_target_layer,
                           [self.proposal_cls_score, self.proposal_predicted, self.im_info, [cfg["ANCHOR_WIDTH"], ], [cfg["ANCHOR_WIDTH"]]],
                           [tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),
                                              name='rpn_labels')  # shape is (1 x H x W x A, 2)
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets,
                                                    name='rpn_bbox_targets')  # shape is (1 x H x W x A, 4)
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights,
                                                           name='rpn_bbox_inside_weights')  # shape is (1 x H x W x A, 4)
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights,
                                                            name='rpn_bbox_outside_weights')  # shape is (1 x H x W x A, 4)

            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

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
                    features = vgg_base(inputs_img_tensor)
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
    # pretrain_model_path = './models/pretrain_model/inception_v4.ckpt'
    # reader = pywrap_tensorflow.NewCheckpointReader(pretrain_model_path)
    # keys = reader.get_variable_to_shape_map().keys()
    # print(keys)

    a = np.zeros([1,2])
    b = np.ones([4,2])
    print(np.vstack((a,b)))

    feat_stride = [16]
    shift_x = tf.range(0, 4) * feat_stride
    shift_y = tf.range(0, 5) * feat_stride

    # shift_x shape = [height, width]
    # 生成同样维度的两个矩阵
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

    # shifts shape = [height*width,4]
    shifts = tf.transpose(tf.stack((tf.reshape(shift_x,[-1]), tf.reshape(shift_y,[-1]),
                          tf.reshape(shift_x,[-1]), tf.reshape(shift_y,[-1]))))

    sess = tf.Session()
    print(sess.run(shifts))
