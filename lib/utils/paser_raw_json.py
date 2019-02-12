import os
import tqdm
import json
import cv2
import numpy as np
from urllib import request

img_path = "/home/zzh/ocr/dataset/ikkyyu_data/v8/img"
res_path = "/home/zzh/ocr/dataset/ikkyyu_data/v8/vis"
txt_path = "/home/zzh/ocr/dataset/ikkyyu_data/v8/label"
label = {"手写框": 0, "打印框": 1, "整体": 2}


def parse_json(path):
    r = open(path, 'r')
    str_data = r.read()
    # print(str_data)
    data = json.loads(str_data)
    img_dict_list = data["image_datas"]
    for img_dict in tqdm.tqdm(img_dict_list):
        img_url = img_dict["pic_url"]
        # print(img_dict["pic_name"])
        img_name = img_dict["pic_name"].split("/")[-1]
        img_save_path = os.path.join(img_path, img_name)
        request.urlretrieve(img_url, img_save_path)

        img = cv2.imread(img_save_path)
        h, w, _ = img.shape
        img_bboxes_info = []
        txt_name = img_name.split('.')[0] + '.txt'
        wf = open(os.path.join(txt_path, txt_name), 'w')
        if len(img_dict["mark_datas"]) == 1:
            # 无需标注的图片
            continue
        for bbox_dict in img_dict["mark_datas"]:
            try:
                bbox_info = []
                bbox_info.append(label[bbox_dict["markd_label"]])

                bbox_list = bbox_dict["marked_path"].split(' ')
                if len(bbox_list) != 9:
                    continue
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

                if bbox_info[0]==2:
                    continue

                pts = np.array([[bbox_info[1], bbox_info[2]],
                                [bbox_info[3], bbox_info[4]],
                                [bbox_info[5], bbox_info[6]],
                                [bbox_info[7], bbox_info[8]]], np.int32)
                # print(pts)
                cv2.polylines(img, [pts], True, color, 2)

                wf.writelines(str(bbox_info[1]))
                wf.writelines(',')
                wf.writelines(str(bbox_info[2]))
                wf.writelines(',')
                wf.writelines(str(bbox_info[3]))
                wf.writelines(',')
                wf.writelines(str(bbox_info[4]))
                wf.writelines(',')
                wf.writelines(str(bbox_info[5]))
                wf.writelines(',')
                wf.writelines(str(bbox_info[6]))
                wf.writelines(',')
                wf.writelines(str(bbox_info[7]))
                wf.writelines(',')
                wf.writelines(str(bbox_info[8]))
                wf.writelines(',')
                if bbox_info[0]==0:
                    wf.writelines("handwritten")
                elif bbox_info[0]==1:
                    wf.writelines("print")
                wf.writelines('\n')
            except ValueError:
                print(bbox_dict["marked_text"])
                print(bbox_dict["index_num"])
                assert 0
        wf.close()
        cv2.imwrite(os.path.join(res_path, img_name), img)

def analysis_json(path):
    r = open(path, 'r')
    str_data = r.read()
    data = json.loads(str_data)
    img_dict_list = data["image_datas"]
    label = []
    for img_dict in tqdm.tqdm(img_dict_list):
        for bbox_dict in img_dict["mark_datas"]:
            if bbox_dict["markd_label"] not in label:
                label.append(bbox_dict["markd_label"])
    print(label)


json_dir = '/home/zzh/v8/json'
json_list = os.listdir(json_dir)
for json_name in json_list:
    print(json_name)
    parse_json(os.path.join(json_dir, json_name))
