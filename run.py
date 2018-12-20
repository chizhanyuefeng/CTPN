import numpy as np
import cv2
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.dataset.dataload import Dataload
from lib.network.solver_wrapper import SloverWrapper
from lib.utils.config import cfg
from lib.network.ctpn_detector import CtpnDetector


train = False

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.75
with tf.Session(config=config) as sess:
    if train:
        s = SloverWrapper(sess)
        s.train_model()
    else:
        ctpn = CtpnDetector(sess)
        img = cv2.imread("./test/8.png")
        a = ctpn.detect(img)
        s = time.time()
        # for i in range(20):
        #     img = cv2.imread("./test/1.png")
        #     a = ctpn.detect(img)
        # print((time.time()-s)/20)

        for b in a[1]:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))

        cv2.imshow('d', img)
        cv2.waitKey()

# a = np.arange(0, 20).reshape([2, 10])
# print(a[0])
# print(a[0, 0::1])