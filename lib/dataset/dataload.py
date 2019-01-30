import os
import random
import numpy as np
import cv2

from lib.utils.config import cfg
from lib.dataset.img_utils import img_normailize, resize_img

class Dataload(object):

    epoch = 0
    img_path_list = None
    current_index = 0

    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_path_list = os.listdir(img_dir)
        label_path_list = os.listdir(label_dir)

        assert len(self.img_path_list) == len(label_path_list), \
            "img num={} is not equal with label num={}".format(len(self.img_path_list), len(label_path_list))

        self.class_dict = {name: i for i, name in enumerate(cfg["CLASSES_NAME"])}

    def getbatch(self):
        img = cv2.imread(os.path.join(self.img_dir, self.img_path_list[self.current_index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _ = resize_img(img)
        img = img_normailize(img)
        h, w, c = img.shape
        # print(img.shape)
        img_data = np.reshape(img, [1, h, w, c])

        label_name = self.img_path_list[self.current_index].split('.')[0] + '.txt'
        assert os.path.exists(os.path.join(self.label_dir, label_name)), \
            "{} is not exist".format(os.path.join(self.label_dir, label_name))

        labels_data = []
        with open(os.path.join(self.label_dir, label_name)) as f:
            lines = f.readlines()
            for line in lines:
                encode_data = line.split('\t')
                label = []
                label.append(float(encode_data[1]))
                label.append(float(encode_data[2]))
                label.append(float(encode_data[3]))
                label.append(float(encode_data[4]))
                label.append(int(self.class_dict[encode_data[0]]))
                labels_data.append(label)

        if self.current_index + 1 + 1 <= len(self.img_path_list):
            self.current_index += 1
        elif self.current_index + 1 + 1 > len(self.img_path_list):
            self.epoch += 1
            self.current_index = 0
            random.shuffle(self.img_path_list)

        return img_data, np.array(labels_data, np.int), np.array(img.shape).reshape([1, 3])


if __name__ == "__main__":
    d = Dataload(cfg["TRAIN"]["TRAIN_IMG_DIR"], cfg["TRAIN"]["TRAIN_LABEL_DIR"])

    img_input, labels, img_info = d.getbatch()
    img = img_input[0]
    print(labels)
    for bbox in labels:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0))

    cv2.imshow('d', img)
    cv2.waitKey()


