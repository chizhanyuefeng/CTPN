import os
import numpy as np
import math
import cv2 as cv

# from lib.utils.config import cfg
from lib.dataset.img_utils import resize_img

train_img_dir = "/home/tony/ocr/ocr_dataset/tal_ocr_data_v7/img"
train_xml_dir = "/home/tony/ocr/ocr_dataset/tal_ocr_data_v7/txt"

val_img_dir = "/home/tony/ocr/ocr_dataset/ctpn/val_data/img"
val_xml_dir = "/home/tony/ocr/ocr_dataset/ctpn/val_data/xml"

img_dir = train_img_dir
xml_dir = train_xml_dir


label_temp_dir = 'train_label_tmp'
out_path = 'train_img_tmp'

proposal_width = 8  # float(cfg["ANCHOR_WIDTH"])

class_name = ["handwritten", "print"]


def parse_txt(txt_path):
    class_list = []
    bbox_list = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split('\t')
            if int(data[0]) > 1:
                continue
            class_list.append(class_name[int(data[0])])
            bbox = []
            bbox.append(int(data[1]))
            bbox.append(int(data[2]))
            bbox.append(int(data[3]))
            bbox.append(int(data[4]))
            bbox_list.append(bbox)

    return class_list, bbox_list



if not os.path.exists(out_path):
    os.makedirs(out_path)
files = os.listdir(img_dir)
files.sort()

for file in files:
    _, basename = os.path.split(file)
    if basename.lower().split('.')[-1] not in ['jpg', 'png', 'JPG', 'JPEG', 'jpeg', 'PNG']:
       print(basename.lower().split('.')[-1])
       continue
    stem, ext = os.path.splitext(basename)
    txt_file = os.path.join(xml_dir, stem + '.txt')
    img_path = os.path.join(img_dir, file)
    # print(img_path)

    img = cv.imread(img_path)
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    re_im, im_scale = resize_img(img)

    re_size = re_im.shape
    cv.imwrite(os.path.join(out_path, stem) + '.jpg', re_im)

    class_list, bbox_list = parse_txt(txt_file)

    assert len(class_list) == len(bbox_list), 'bbox和label不对应'

    assert len(class_list) > 0, 'xml文件有问题{}'.format(txt_file)

    for bbox_index in range(len(bbox_list)):
        # if class_list[bbox_index] == 2:
        #     continue

        if len(bbox_list[bbox_index]) == 8:
            xmin = int(np.floor(float(min(bbox_list[bbox_index][0], bbox_list[bbox_index][2], bbox_list[bbox_index][4], bbox_list[bbox_index][6])) / img_size[0] * re_size[0]))
            ymin = int(np.floor(float(min(bbox_list[bbox_index][1], bbox_list[bbox_index][3], bbox_list[bbox_index][5], bbox_list[bbox_index][7])) / img_size[1] * re_size[1]))
            xmax = int(np.ceil(float(max(bbox_list[bbox_index][0], bbox_list[bbox_index][2], bbox_list[bbox_index][4], bbox_list[bbox_index][6])) / img_size[0] * re_size[0]))
            ymax = int(np.ceil(float(max(bbox_list[bbox_index][1], bbox_list[bbox_index][3], bbox_list[bbox_index][5], bbox_list[bbox_index][7])) / img_size[1] * re_size[1]))
        elif len(bbox_list[bbox_index])==4:
            xmin = int(np.floor(float(bbox_list[bbox_index][0])/img_size[0] * re_size[0]))
            ymin = int(np.floor(float(bbox_list[bbox_index][1])/img_size[1] * re_size[1]))
            xmax = int(np.ceil(float(bbox_list[bbox_index][2])/img_size[0] * re_size[0]))
            ymax = int(np.ceil(float(bbox_list[bbox_index][3])/img_size[1] * re_size[1]))
        else:
            print(txt_file)
            assert 0, "{}bbox error".format(txt_file)

        if xmin < 0:
            xmin = 0
        if xmax > re_size[1] - 1:
            xmax = re_size[1] - 1
        if ymin < 0:
            ymin = 0
        if ymax > re_size[0] - 1:
            ymax = re_size[0] - 1

        width = xmax - xmin + 1
        height = ymax - ymin + 1

        step = proposal_width
        x_left = []
        x_right = []
        x_left.append(xmin)
        x_left_start = int(math.ceil(xmin / proposal_width) * proposal_width)
        if x_left_start == xmin:
            x_left_start = xmin + proposal_width
        for i in np.arange(x_left_start, xmax, proposal_width):
            x_left.append(i)
        x_left = np.array(x_left)

        x_right.append(x_left_start - 1)
        for i in range(1, len(x_left) - 1):
            x_right.append(x_left[i] + proposal_width-1)
        x_right.append(xmax)
        x_right = np.array(x_right)

        idx = np.where(x_left == x_right)
        x_left = np.delete(x_left, idx, axis=0)
        x_right = np.delete(x_right, idx, axis=0)

        if not os.path.exists(label_temp_dir):
            os.makedirs(label_temp_dir)

        # if class_list[bbox_index] <= 1:
        #     current_class = class_name[class_list[bbox_index] + 1]
        # else:
        #     assert 0, '不该出现其他类型的class:{}'.format(class_list[bbox_index])

        with open(os.path.join(label_temp_dir, stem) + '.txt', 'a+') as f:
            for i in range(len(x_left)):
                f.writelines(class_list[bbox_index])
                f.writelines("\t")
                f.writelines(str(x_left[i]))
                f.writelines("\t")
                f.writelines(str(ymin))
                f.writelines("\t")
                f.writelines(str(x_right[i]))
                f.writelines("\t")
                f.writelines(str(ymax))
                f.writelines("\n")
