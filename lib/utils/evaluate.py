import os
import numpy as np
from lib.utils.polygon_utils import polygon_iou, polygon_riou

from lib.utils.parse_json import parse_json

label = {"handwritten": 0, "print": 1, "all": 2}
workspace_dir = './'


def evaluate(gt_boxes_list, pred_boxes_list, overlap=0.5, r_overlap=0.9):
    """
    评价网络模型
    :param gt_boxes_list: [[x1,y1,x2,y2,x3,y3,x4,y4,cls],...]
    :param pred_boxes_list: [[x1,y1,x2,y2,x3,y3,x4,y4,cls,score],...]
    :param overlap:
    :param r_overlap:
    :return:
    """
    # 根据score对预测box进行排序
    pred_boxes_list.sort(key=lambda x: x[9], reverse=True)

    TP = 0
    pred_flag = [0] * len(pred_boxes_list)
    gt_flag = [0] * len(gt_boxes_list)
    for i, pred_box_info in enumerate(pred_boxes_list):
        pred_box = [[pred_box_info[0], pred_box_info[1]],
                    [pred_box_info[2], pred_box_info[3]],
                    [pred_box_info[4], pred_box_info[5]],
                    [pred_box_info[6], pred_box_info[7]]]
        pred_cls = pred_box_info[8]
        for j, gt_box_info in enumerate(gt_boxes_list):
            gt_cls = gt_box_info[8]
            if gt_flag[j] == 1:
                continue
            if pred_cls != gt_cls:
                continue
            gt_box = [[gt_box_info[0], gt_box_info[1]],
                      [gt_box_info[2], gt_box_info[3]],
                      [gt_box_info[4], gt_box_info[5]],
                      [gt_box_info[6], gt_box_info[7]]]
            iou = polygon_iou(pred_box, gt_box)
            riou = polygon_riou(pred_box, gt_box)

            if iou > overlap and riou > r_overlap:
                TP += 1
                pred_flag[i] = 1
                gt_flag = 1
                break

    precision = TP/float(len(pred_boxes_list))
    recall = TP/float(len(gt_boxes_list))
    F1_score = 2*(precision*recall)/(precision+recall)

    return TP, precision, recall, F1_score


def read_label_file(file_path, is_gt):
    """
    读取label文件，并返回手写，打印的bbox信息
    :param file_path:
    :param is_gt:
    :return:
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        print_boxes_info_list = []
        handwritten_boxes_info_list = []
        for line in lines:
            info = line.split()
            bbox_info = []
            # bbox pts
            bbox_info.append(int(info[0]))
            bbox_info.append(int(info[1]))
            bbox_info.append(int(info[2]))
            bbox_info.append(int(info[3]))
            bbox_info.append(int(info[4]))
            bbox_info.append(int(info[5]))
            bbox_info.append(int(info[6]))
            bbox_info.append(int(info[7]))
            # bbox cls
            bbox_info.append(int(label[info[8]]))
            if not is_gt:
                bbox_info.append(float(info[9]))

            if info[8] == 'handwritten':
                handwritten_boxes_info_list.append(bbox_info)
            elif info[8] == 'print':
                print_boxes_info_list.append(bbox_info)

    return print_boxes_info_list, handwritten_boxes_info_list


def evaluate_all(workspace_dir):
    gt_file_dir = os.path.join(workspace_dir, 'gt_label')
    gt_file_list = os.listdir(gt_file_dir)

    pred_file_dir = os.path.join(workspace_dir, 'pred_label')
    pred_file_list = os.listdir(pred_file_dir)

    assert len(pred_file_list) == len(gt_file_list), '{}和{}中的文件数目不一致'.format(gt_file_dir, pred_file_dir)

    # 0: 手写, 1: 打印
    all_TP = [0, 0]
    all_pred_num = [0, 0]
    all_gt_num = [0, 0]
    per_file_res = []

    for pred_file in pred_file_list:
        if pred_file not in gt_file_list:
            assert 0, '{}预测文件没有在{}找到应gt文件'.format(pred_file, gt_file_dir)
        gt_bboxes_info_list = read_label_file(os.path.join(gt_file_dir, pred_file), True)
        pred_bboxes_info_list= read_label_file(os.path.join(pred_file_dir, pred_file), False)
        res = {}
        res['file'] = pred_file
        for i in range(2):
            TP, precision, recall, F1_score = evaluate(gt_bboxes_info_list[i],
                                                       pred_bboxes_info_list[i],
                                                       overlap=0.5,
                                                       r_overlap=0.9)
            all_TP[i] += TP
            all_gt_num[i] += len(gt_bboxes_info_list[i])
            all_pred_num[i] += len(pred_bboxes_info_list[i])
            res[i] = [precision, recall, F1_score]
        per_file_res.append(res)


if __name__ == "__main__":
    l = [[1,2,3],[2,3,6],[3,4,0],[4,5,9],[5,6,1],[6,7,2]]
    l.sort(key=lambda x:x[2], reverse=True)
    # a = np.array(l)
    # b = a[:, 2]
    # b = np.reshape(b, [-1, 1])
    # c = np.sort(b)
    # print(b)
    # print(c)
    print(l)

"""
原定本周工作计划：
1.清洗整理下周返回的标注数据，更新新的标注需求，继续迭代检测部分。
2.继续完善proposal合并策略。
3.针对小印章项目需求，制定接口和清洗数据。

实际本周工作计划：
1.商讨测试检测算法的量化方案。
2.重新制定OCR判题标注需求（将版面分析和百桥需求剔除，并根据以后需求进行修改）。
3.写测试脚本提供测试人员。
4.针对小印章项目需求，制定接口并验收清洗数据。

下周工作计划：
1.验收清洗1000张2C检测数据。
2.针对清洗完的数据进行模型迭代。
3.交付一版小印章的baseline。

"""