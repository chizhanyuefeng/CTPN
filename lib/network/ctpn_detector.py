import os
import time
import numpy as np
import tensorflow as tf

from lib.utils.config import cfg
from lib.network.ctpn_network import CTPN
from lib.dataset.img_utils import resize_img, img_normailize
from lib.text_connect.text_connector import TextConnector


class CtpnDetector(object):
    def __init__(self, sess):
        self.ctpn_network = CTPN()
        self.rpn_rois, self.rpn_targets = self.ctpn_network.inference()
        #
        # config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allocator_type = 'BFC'
        # config.gpu_options.per_process_gpu_memory_fraction = 0.75
        # with tf.Session(config=config) as sess:
        self.sess = sess

        saver = tf.train.Saver()
        try:
            ckpt = tf.train.get_checkpoint_state(cfg["TEST"]["CHECKPOINTS_PATH"])
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('done')
        except:
            assert 0, 'Check your model {}'.format(ckpt.model_checkpoint_path)

    def detect(self, img):
        img, scale = resize_img(img)
        img = img_normailize(img)
        h, w, c = img.shape
        img_input = np.reshape(img, [1, h, w, c])
        img_info = [h, w, 1]
        s = time.time()
        scores, pp_boxes = self._get_net_output(img_input, img_info)
        print('net:', time.time()-s)

        s = time.time()
        text_connector = TextConnector()
        # 得到是resize图像后的bbox
        # print(img_info)
        # print('boxes, scores[:, np.newaxis]',boxes.shape, scores[:, np.newaxis].shape,scores.shape)
        text_proposals, scores, boxes = text_connector.detect(pp_boxes, scores[:, np.newaxis], img_info[:2])
        print('merge:', time.time() - s)
        # 原图像的绝对bbox位置
        original_bbox, scores = self._resize_bbox(boxes, scale)

        return pp_boxes, original_bbox, scores

    def _get_net_output(self, img_input, img_info):
        feed_dict = {self.ctpn_network.img_input: img_input, self.ctpn_network.im_info: [img_info]}
        res_list = self.sess.run([self.rpn_rois, self.rpn_targets], feed_dict=feed_dict)
        # print(rois)
        # print(rois[0])
        rois = res_list[0]
        print(rois.shape)
        scores = rois[:, 0]
        boxes = rois[:, 1:5]
        return scores, boxes

    def _resize_bbox(self, boxes, scale):
        resized_bbox = []
        scores = []
        for box in boxes:
            bbox = []
            bbox.append(min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale)))
            bbox.append(min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale)))
            bbox.append(max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale)))
            bbox.append(max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale)))
            scores.append(box[8])
            resized_bbox.append(bbox)

        return resized_bbox, scores



