import os
import cv2
import numpy as np

def draw_img(img, bbox_list, cls_list):
    def ploy_pots(bbox):
        ploy_bbox = []
        ploy_bbox.append([bbox[0], bbox[1]])
        ploy_bbox.append([bbox[2], bbox[3]])
        ploy_bbox.append([bbox[4], bbox[5]])
        ploy_bbox.append([bbox[6], bbox[7]])

        return ploy_bbox


    print_color = (255, 0, 0)
    hand_color = (0, 255, 0)
    for i, bbox in enumerate(bbox_list):
        if cls_list[i] == 'print':
            cv2.polylines(img, [np.array(ploy_pots(bbox))], True, print_color)
        elif cls_list[i] == 'handwritten':
            cv2.polylines(img, [np.array(ploy_pots(bbox))], True, hand_color)

    return img


img_dir = '/home/zzh/ocr/dataset/ikkyyu_data/v8/img'
label_dir = '/home/zzh/ocr/dataset/ikkyyu_data/v8/label'

img_list = os.listdir(img_dir)
for img_name in img_list:
    img = cv2.imread(os.path.join(img_dir, img_name))
    img_base_name = img_name.split('.')[0]
    label_name = img_base_name + '.txt'

    with open(os.path.join(label_dir, label_name), 'r') as rf:
        lines = rf.readlines()
        bbox_list = []
        cls_list = []

        for line in lines:
            x1, y1, x2, y2, x3, y3, x4, y4, cls = line.split(',')
            cls = cls.replace('\n', '')
            cls_list.append(cls)
            box = []
            box.append(int(x1))
            box.append(int(y1))
            box.append(int(x2))
            box.append(int(y2))
            box.append(int(x3))
            box.append(int(y3))
            box.append(int(x4))
            box.append(int(y4))
            bbox_list.append(box)

        img = draw_img(img, bbox_list, cls_list)
        cv2.imshow('dwa', img)
        cv2.waitKey()




