import os
import numpy as np
import tensorflow as tf

from lib.utils.config import cfg
from lib.network.ctpn_network import CTPN
from lib.dataset.img_utils import resize_img


class CtpnDetector(object):
    def __init__(self, sess):
        self.sess = sess
        self.ctpn_network = CTPN()
        self.rpn_rois = self.ctpn_network.inference()
        # feed_dict = {net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}
        # rois = sess.run([net.get_output('rois')[0], net.get_output('rpn_targets')], feed_dict=feed_dict)
        # rois = rois[0]

    def detect(self, img):
        img, ratio = resize_img(img)

        img_info = img.shape
        img_input = np.reshape(img, [1, img_info[0], img_info[1], img_info[2]])
        feed_dict = {self.ctpn_network.img_input: img_input, self.ctpn_network.im_info: [img_info]}
        rois = self.sess.run(self.rpn_rois[0], feed_dict=feed_dict)

        scores = rois[:, 0]
        boxes = rois[:, 1:5] / ratio
        return scores, boxes