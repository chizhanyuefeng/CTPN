import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow

from lib.utils.config import cfg
from lib.network.inception_base import inception_base
from lib.network.vgg_base import vgg_base
from lib.rpn_layer.generate_anchors import generate_anchors

class CTPN(object):

    def __init__(self):
        pass

    def inference(self):
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        proposal_predicted, proposal_cls_score = self.__ctpn_base()


        proposal_cls_score_shape = tf.shape(proposal_cls_score)
        self.proposal_cls_score = tf.reshape(proposal_predicted, [-1, cfg["CLASSES_NUM"]])
        self.proposal_cls_prob = tf.reshape(tf.nn.softmax(proposal_cls_score), proposal_cls_score_shape)


    def __proposal_layer(self):
        anchors = generate_anchors()
        anchor_num = anchors.shape[0]

        # 原始图像的高宽、缩放尺度
        img_info = self.im_info[0]
        pre_nms_topN = cfg["TEST"]["RPN_PRE_NMS_TOP_N"]
        post_nms_topN = cfg["TEST"]["RPN_POST_NMS_TOP_N"]
        nms_thresh = cfg["TEST"]["RPN_NMS_THRESH"]
        min_size = cfg["TEST"]["RPN_MIN_SIZE"]

        # feature-map的高宽
        height, width = tf.shape(self.proposal_cls_prob.shape[1:3])

        # 获取第一个分类结果
        scores = tf.reshape(tf.reshape(self.proposal_cls_prob, [1, height, width, anchor_num, cfg["CLASSES_NUM"]])[:, :, :, :, 1],
                            [1, height, width, anchor_num])

        bbox_deltas = self.proposal_cls_prob

        # 同anchor-target-layer-tf这个文件一样，生成anchor的shift，进一步得到整张图像上的所有anchor
        feat_stride = [cfg["ANCHOR_WIDTH"]]
        shift_x = tf.range(0, width) * feat_stride
        shift_y = tf.range(0, height) * feat_stride

        # shift_x shape = [height, width]
        # 生成同样维度的两个矩阵
        shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
        # shifts shape = [height*width,4]
        shifts = tf.transpose(tf.stack((tf.reshape(shift_x, [-1]), tf.reshape(shift_y, [-1]),
                                        tf.reshape(shift_x, [-1]), tf.reshape(shift_y, [-1]))))

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = anchor_num  # 10
        K = shifts.shape[0]  # height*width,[height*width,4]
        anchors = anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))

        anchors = anchors.reshape((K * A, 4))  # 这里得到的anchor就是整张图像上的所有anchor

        # (HxWxA, 2)
        bbox_deltas = tf.reshape(bbox_deltas, (-1, 2))

        # Same story for the scores:
        scores = tf.reshape(scores, (-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)  # 做逆变换，得到box在图像上的真实坐标

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])  # 将所有的proposal修建一下，超出图像范围的将会被修剪掉


    def bbox_transform_inv(self):
        pass

    def clip_boxes(self):
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
