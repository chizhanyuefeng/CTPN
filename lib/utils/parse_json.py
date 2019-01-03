import os
import json
import tqdm
import cv2
import numpy as np
from urllib import request

label = {"手写框": 0, "打印框": 1, "整体": 2}


def parse_json(path, save_dir):
    img_path = os.path.join(save_dir, 'img')
    gt_file = os.path.join(save_dir, 'gt_label')
    gt_res_path = os.path.join(save_dir, 'gt_res')

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if not os.path.exists(gt_file):
        os.makedirs(gt_file)

    r = open(path, 'r')
    str_data = r.read()
    data = json.loads(str_data)
    img_dict_list = data["image_datas"]
    for img_dict in tqdm.tqdm(img_dict_list):
        img_url = img_dict["pic_url"]
        img_name = img_dict["pic_name"].split("/")[-1]
        img_save_path = os.path.join(img_path, img_name)
        request.urlretrieve(img_url, img_save_path)

        img = cv2.imread(img_save_path)
        h, w, _ = img.shape
        img_bboxes_info = []
        txt_name = img_name.split('.')[0] + '.txt'
        wf = open(os.path.join(gt_file, txt_name), 'w')

        for bbox_dict in img_dict["mark_datas"]:
            bbox_info = []
            bbox_info.append(label[bbox_dict["markd_label"]])
            bbox_list = bbox_dict["marked_path"].split(' ')
            bbox_info.append(int(float(bbox_list[0].replace('M', '')) * w))
            bbox_info.append(int(float(bbox_list[1]) * h))
            bbox_info.append(int(float(bbox_list[2].replace('L', '')) * w))
            bbox_info.append(int(float(bbox_list[3]) * h))
            bbox_info.append(int(float(bbox_list[4].replace('L', '')) * w))
            bbox_info.append(int(float(bbox_list[5]) * h))
            bbox_info.append(int(float(bbox_list[6].replace('L', '')) * w))
            bbox_info.append(int(float(bbox_list[7]) * h))
            img_bboxes_info.append(img_bboxes_info)

            if bbox_info[0] == 0:
                color = (255, 0, 0)
            elif bbox_info[0] == 1:
                color = (0, 255, 0)
            elif bbox_info[0] == 2:
                color = (0, 0, 255)

            pts = np.array([[bbox_info[1], bbox_info[2]],
                            [bbox_info[3], bbox_info[4]],
                            [bbox_info[5], bbox_info[6]],
                            [bbox_info[7], bbox_info[8]]], np.int32)

            cv2.polylines(img, [pts], True, color)
            # if bbox_info[0]>1:
            #     continue

            if bbox_info[0]==0:
                wf.writelines("handwritten")
            elif bbox_info[0]==1:
                wf.writelines("print")
            elif bbox_info[0]==2:
                wf.writelines("all")
            wf.writelines('\t')
            wf.writelines(str(min(bbox_info[1], bbox_info[3], bbox_info[5], bbox_info[7])))
            wf.writelines('\t')
            wf.writelines(str(min(bbox_info[2], bbox_info[4], bbox_info[6], bbox_info[8])))
            wf.writelines('\t')
            wf.writelines(str(max(bbox_info[1], bbox_info[3], bbox_info[5], bbox_info[7])))
            wf.writelines('\t')
            wf.writelines(str(max(bbox_info[2], bbox_info[4], bbox_info[6], bbox_info[8])))
            wf.writelines('\n')
        wf.close()
        cv2.imwrite(os.path.join(gt_res_path, img_name), img)