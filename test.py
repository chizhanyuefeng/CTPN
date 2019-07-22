from tensorflow.python import pywrap_tensorflow
import numpy as np
import cv2
import os
from lib.dataset.dataload import Dataload
from lib.utils.config import cfg

if __name__ == "__main__":
    # pretrain_model_path = './models/pretrain_model/inception_v4.ckpt'
    # reader = pywrap_tensorflow.NewCheckpointReader(pretrain_model_path)
    # keys = reader.get_variable_to_shape_map().keys()
    # print(len(keys))
    # print(keys)

    train_data_load = Dataload(cfg["TRAIN"]["TRAIN_IMG_DIR"], cfg["TRAIN"]["TRAIN_LABEL_DIR"])
    for i in range(10):
        img_input, labels, img_info = train_data_load.getbatch()
        print(img_info)

        # txt_dir = "/home/tony/ocr/CTPN/train_label"
        # img_dir = "/home/tony/ocr/CTPN/train_image"
        # img_list = os.listdir(img_dir)
        img = img_input[0]
        for label in labels:

            if label[4] == 1:
                cv2.rectangle(img, (label[0], label[1]), (label[2], label[3]), (255, 0, 0))
            else:
                cv2.rectangle(img, (label[0], label[1]), (label[2], label[3]), (0, 255, 0))

        cv2.imshow("dw", img)
        cv2.waitKey()

    #
    # for img_name in img_list:
    #     img_basename = img_name.split(".")[0]
    #     txt_name = img_basename + ".txt"
    #     # print(img_name)
    #     img = cv2.imread(os.path.join(img_dir, img_name))
    #     with open(os.path.join(txt_dir, txt_name)) as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             _, x1, y1, x2, y2 = line.split('\t')
    #
    #             cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0))
    #
    #         cv2.imshow("dw", img)
    #         cv2.waitKey()




