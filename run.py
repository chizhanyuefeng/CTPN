import numpy as np
import cv2
import time
import os
import tensorflow as tf
# import tensorflow.contrib.slim as slim
# from lib.dataset.dataload import Dataload
from lib.network.solver_wrapper import SloverWrapper
# from lib.utils.config import cfg
from lib.network.ctpn_detector import CtpnDetector


train = True

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    if train:
        s = SloverWrapper(sess)
        s.train_model()
    else:
        ctpn = CtpnDetector(sess)
        img = cv2.imread("./data/image/0001.jpg")
        img_dir = './data/image/'
        img_list = os.listdir(img_dir)
        for img_name in img_list:
            img = cv2.imread(os.path.join(img_dir, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print('*********************************************')
            s = time.time()
            a = ctpn.detect(img)
            print('time:', time.time()-s)
            b_img = img.copy()
            for b in a[1]:
                cv2.rectangle(b_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))
            cv2.imwrite(os.path.join('./result', img_name), b_img)
            p_img = img.copy()
            for b in a[0]:
                cv2.rectangle(p_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))
            cv2.imwrite(os.path.join('./proposal', img_name), p_img)

