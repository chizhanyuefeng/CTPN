import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow

from lib.utils.config import cfg
from lib.network.inception_base import inception_base
from lib.network.vgg_base import vgg_base
from lib.rpn_layer.anchor_target_layer_tf import anchor_target_layer
from lib.rpn_layer.proposal_layer_tf import proposal_layer

class CTPN(object):

    def __init__(self, is_train=False):
        self.img_input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="img_input")
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name="img_info")

    def inference(self):

        proposal_predicted, proposal_cls_score, proposal_cls_prob = self.__ctpn_base()
        rpn_rois, rpn_targets = self.__proposal_layer(proposal_cls_prob, proposal_predicted)

        return rpn_rois, rpn_targets

    def build_loss(self):

        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')

        proposal_predicted, proposal_cls_score, _ = self.__ctpn_base()

        rpn_labels, \
        rpn_bbox_targets, \
        rpn_bbox_inside_weights, \
        rpn_bbox_outside_weights = self.__anchor_layer(proposal_cls_score)

        # classification loss
        # (1, H, W, A x d) -> (1, H, WxA, d)
        cls_pred_shape = tf.shape(proposal_cls_score)
        cls_pred_reshape = tf.reshape(proposal_cls_score, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])
        rpn_cls_score = tf.reshape(cls_pred_reshape, [-1, 2])

        rpn_label = tf.reshape(rpn_labels, [-1])  # shape (HxWxA)

        fg_keep = tf.equal(rpn_label, 1)
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))

        rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep)  # shape (N, 2)
        rpn_label = tf.gather(rpn_label, rpn_keep)
        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score)

        # box loss TODO:回归2个值
        rpn_bbox_pred = proposal_predicted  # shape (1, H, W, Ax4)
        rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep)  # shape (N, 4)
        rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep)
        rpn_bbox_inside_weights = tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep)
        rpn_bbox_outside_weights = tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep)

        rpn_loss_box_n = tf.reduce_sum(rpn_bbox_outside_weights * self.__smooth_l1_dist(
            rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), reduction_indices=[1])

        rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
        rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)

        model_loss = rpn_cross_entropy + rpn_loss_box

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(regularization_losses) + model_loss

        return total_loss, model_loss, rpn_cross_entropy, rpn_loss_box

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path, encoding='latin1').item()
        for key in data_dict:
            with tf.variable_scope("CTPN_Network/" + key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print("assign pretrain model "+subkey+ " to "+key)
                    except ValueError:
                        print("ignore "+key)
                        if not ignore_missing:
                            raise

    def __ctpn_base(self):
        """
        特征提取层 feature extract layer
        :return: proposal_predicted : shape = [1, h, w, A*4
                 proposal_cls_score: shape = [1, h, w, A*cfg["CLASSES_NUM"]]
                 proposal_cls_prob: shape = [1, h, w, A*cfg["CLASSES_NUM"]]
        """
        stddev = 0.01
        weight_decay = cfg["TRAIN"]["WEIGHT_DECAY"]

        assert cfg["ANCHOR_WIDTH"] == 8 or cfg["ANCHOR_WIDTH"] == 16, \
            'Anchor must be 8 or 16!Not be {}.'.format(cfg["ANCHOR_WIDTH"])

        with tf.variable_scope("CTPN_Network"):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(0.0, stddev=stddev),
                                weights_regularizer=slim.l2_regularizer(weight_decay)
                                ):

                if cfg["BACKBONE"] == "InceptionNet":
                    features, featuremap_scale = inception_base(self.img_input)
                elif cfg["BACKBONE"] == "VggNet":
                    features, featuremap_scale = vgg_base(self.img_input)
                else:
                    assert 0, "error: backbone {} is not support!".format(cfg["BACKBONE"])

            print('featuremap_scale is {}, anchor width is {}'.format(featuremap_scale, cfg['ANCHOR_WIDTH']))
            assert featuremap_scale == cfg['ANCHOR_WIDTH']

            print("using {} backbone...".format(cfg["BACKBONE"]))

            features = slim.conv2d(features, 512, [3, 3], scope='rpn_conv_3x3')

            if cfg["USE_LSTM"]:
                features = self.__bilstm(features, 512, 128, 512)
            else:
                features = self.__semantic_info_extract_layer(features)
            print('Lstm is using?', cfg["USE_LSTM"])

            proposal_predicted = self._lstm_fc(features, 512, 10 * 4, scope_name="bbox_pred")
            proposal_cls_score = self._lstm_fc(features, 512, 10 * 2, scope_name="cls_pred")

            proposal_cls_score_shape = tf.shape(proposal_cls_score)
            # proposal_cls_score_reshape shape = [h*w*A, cfg["CLASSES_NUM"]]
            proposal_cls_score_reshape = tf.reshape(proposal_cls_score, [proposal_cls_score_shape[0],
                                                                         proposal_cls_score_shape[1],
                                                                         -1,
                                                                         cfg["CLASSES_NUM"]])
            proposal_cls_score_reshape_shape = tf.shape(proposal_cls_score_reshape)
            proposal_cls_score_reshape = tf.reshape(proposal_cls_score_reshape, [-1, proposal_cls_score_reshape_shape[3]])
            # proposal_cls_prob shape = [1, h, w, A*cfg["CLASSES_NUM"]]
            proposal_cls_prob = tf.reshape(tf.nn.softmax(proposal_cls_score_reshape),
                                           [-1,
                                            proposal_cls_score_reshape_shape[1],
                                            proposal_cls_score_reshape_shape[2],
                                            proposal_cls_score_reshape_shape[3]]
                                           )

        return proposal_predicted, proposal_cls_score, proposal_cls_prob

    def __proposal_layer(self, proposal_cls_prob, proposal_predicted):
        """
        回归proposal框
        :param proposal_cls_prob: shape = [1, h, w, Axclass_num]
        :param proposal_predicted: shape = [1, h, w, Ax4] TODO:回归2个值
        :return rpn_rois : shape = [1 x H x W x A, 5]
                rpn_targets : shape = [1 x H x W x A, 2]
        """
        with tf.variable_scope("proposal_layer"):
            blob, bbox_delta = tf.py_func(proposal_layer,
                                          [proposal_cls_prob, proposal_predicted, self.im_info, [cfg["ANCHOR_WIDTH"], ]],
                                          [tf.float32, tf.float32])

            rpn_rois = tf.reshape(blob, [-1, 5], name='rpn_rois')
            rpn_targets = tf.convert_to_tensor(bbox_delta, name='rpn_targets')
            return rpn_rois, rpn_targets

    def __anchor_layer(self, proposal_cls_score):
        with tf.variable_scope("anchor_layer"):
            # 'rpn_cls_score', 'gt_boxes', 'im_info'
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
                tf.py_func(anchor_target_layer,
                           [proposal_cls_score, self.gt_boxes, self.im_info, self.img_input],
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

    def __bilstm(self, input, input_channel, hidden_unit_num, output_channel, name="Bilstm"):
        """
        双向rnn
        :param input:
        :param input_channel: 512 每个timestep 携带信息
        :param hidden_unit_num: 128 一层rnn
        :param output_channel: 512 最后rrn层输出
        :param name:
        :param trainable:
        :return:
        """
        with tf.variable_scope(name):
            shape = tf.shape(input)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            img = tf.reshape(input, [N * H, W, C])
            img.set_shape([None, None, input_channel])

            lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)

            lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, img, dtype=tf.float32)
            lstm_out = tf.concat(lstm_out, axis=-1)

            lstm_out = tf.reshape(lstm_out, [N * H * W, 2 * hidden_unit_num])
            outputs = slim.fully_connected(lstm_out, output_channel, activation_fn=None, weights_regularizer=None)
            outputs = tf.reshape(outputs, [N, H, W, output_channel])

            return outputs

    def __semantic_info_extract_layer(self, input, name="semantic_extract_layer"):
        """
        使用inception 模块来代替 lstm 提取语义信息
        :param input:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(input, 128, [1, 1], scope='Conv2d_b0_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(input, 128, [1, 1], scope='Conv2d_b1_1x1')
                branch_1 = slim.conv2d(branch_1, 128, [1, 3], scope='Conv2d_b1_5x5')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(input, 128, [1, 1], scope='Conv2d_b2_1x1')
                branch_2 = slim.conv2d(branch_2, 128, [1, 5], scope='Conv2d_b2_0_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(input, [3, 3], stride=1, padding='SAME', scope='AvgPool_b3_3x3')
                branch_3 = slim.conv2d(branch_3, 128, [1, 7], scope='Conv2d_b3_1x1')
            outputs = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3, name='concat')

        return outputs

    def __smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name):
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                        (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)

    def _lstm_fc(self, net, input_channel, output_channel, scope_name):
        with tf.variable_scope(scope_name):
            shape = tf.shape(net)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            net = tf.reshape(net, [N * H * W, C])

            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            init_biases = tf.constant_initializer(0.0)
            weights = tf.get_variable('weights', [input_channel, output_channel], initializer=init_weights)
            biases = tf.get_variable('biases', [output_channel], initializer=init_biases)

            output = tf.matmul(net, weights) + biases
            output = tf.reshape(output, [N, H, W, output_channel])
        return output


if __name__ == "__main__":
    pass
    # pretrain_model_path = './models/pretrain_model/inception_v4.ckpt'
    # reader = pywrap_tensorflow.NewCheckpointReader(pretrain_model_path)
    # keys = reader.get_variable_to_shape_map().keys()
    # print(keys)


