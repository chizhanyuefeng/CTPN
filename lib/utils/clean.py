import os
import cv2
import tqdm
import shapely

from shapely.geometry import Polygon, MultiPoint
import numpy as np


def find_big_box(boxes_list, threhold):
    # boxes_np = [None, 4]
    boxes_np = np.array(boxes_list)

    boxes_height = boxes_np[:, 3] - boxes_np[:, 1]

    boxes_avg_height = np.sum(boxes_height) / np.shape(boxes_height)[0]

    height_filter_value = boxes_avg_height * threhold

    height_filter = np.where(boxes_height >= height_filter_value)
    filted_boxes = boxes_np[height_filter]

    height_filter = np.where(boxes_height < height_filter_value)
    need_boxes = boxes_np[height_filter]

    return need_boxes, filted_boxes


def read_file(txt_file):
    with open(txt_file, 'r') as rf:
        lines = rf.readlines()
        print_boxes = []
        handwritten_boxes = []
        if len(lines)== 0:
           return None, None, None, None
        for line in lines:
            cls, x1, y1, x2, y2 = line.split('\t')
            if cls == 'all':
                continue
            elif cls == 'handwritten':
                box = []
                box.append(int(x1))
                box.append(int(y1))
                box.append(int(x2))
                box.append(int(y2))
                handwritten_boxes.append(box)
            elif cls == 'print':
                box = []
                box.append(int(x1))
                box.append(int(y1))
                box.append(int(x2))
                box.append(int(y2))
                print_boxes.append(box)
            else:
                assert 0, 'error label: {}'.format(cls)
        if len(print_boxes) == 0:
           return handwritten_boxes, [], [], [],
        need_print_boxes, filted_print_boxes = find_big_box(print_boxes, 1.3)
        need_print_boxes = need_print_boxes.tolist()
        filted_handwritten_boxes = []
        new_filted_print_boxes = []
        all_temp = handwritten_boxes.copy()
        need_print_boxes_temp = need_print_boxes.copy()
        all_temp.extend(need_print_boxes_temp)

        for filted_print_box in filted_print_boxes:
            for box in all_temp:
                iou = rectangle_riou(filted_print_box, box)
                if iou > 0.2:
                    # print(handwritten_boxes)
                    
                    if box in handwritten_boxes:
                        handwritten_boxes.remove(box)
                        filted_handwritten_boxes.append(box)
                    elif box in need_print_boxes:
                        need_print_boxes.remove(box)
                        new_filted_print_boxes.append(box)
        new_filted_print_boxes.extend(filted_print_boxes)
        return handwritten_boxes, need_print_boxes, filted_handwritten_boxes, new_filted_print_boxes

def rectangle_riou(pred_box, gt_box):
    """
    计算预测和gt的iou
    :param pred_box: list [x1, y1, x2, y2]
    :param gt_box: list [x1, y1, x2, y2]
    :return:
    """
    def rectangle2polygon(box):
        return [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]

    pred_polygon_points = np.array(rectangle2polygon(pred_box)).reshape(4, 2)
    pred_poly = Polygon(pred_polygon_points).convex_hull
    gt_polygon_points = np.array(rectangle2polygon(gt_box)).reshape(4, 2)
    gt_poly = Polygon(gt_polygon_points).convex_hull

    if not pred_poly.intersects(gt_poly):
        iou = 0
    else:
        try:
            inter_area = pred_poly.intersection(gt_poly).area
            # union_area = gt_box.area
            union_area = gt_poly.area
            if union_area == 0:
                return 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


txt_dir = '/home/zzh/v8/txt/'
new_txt_dir = '/home/zzh/v8/new_txt/'
img_dir = '/home/zzh/v8/img/'
filted_dir = '/home/zzh/v8/box_filter_mask/'

img_name_list = os.listdir(img_dir)

for img_name in tqdm.tqdm(img_name_list):
    img_basename = img_name.split('.')[0]
    print(img_basename)
    handwritten_boxes, \
    need_print_boxes, \
    filted_handwritten_boxes, \
    filted_print_boxes = read_file(os.path.join(txt_dir, img_basename + '.txt'))
    if handwritten_boxes == None:
       continue
    all_filted = filted_handwritten_boxes
    all_filted.extend(filted_print_boxes)
    img = cv2.imread(os.path.join(img_dir, img_name))

    for boxes in all_filted:
        # cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 0, 255), 2)
        img[boxes[1]:boxes[3], boxes[0]:boxes[2], :] = (255, 255, 255)
    # handwritten_boxes.extend(need_print_boxes)
    with open(os.path.join(new_txt_dir, img_basename+'.txt'), 'w') as wf:
        for boxes in handwritten_boxes:
            wf.writelines('handwritten')
            wf.writelines('\t')
            wf.writelines(str(boxes[0]))
            wf.writelines('\t')
            wf.writelines(str(boxes[1]))
            wf.writelines('\t')
            wf.writelines(str(boxes[2]))
            wf.writelines('\t')
            wf.writelines(str(boxes[3]))
            wf.writelines('\n')

        for boxes in need_print_boxes:
            wf.writelines('print')
            wf.writelines('\t')
            wf.writelines(str(boxes[0]))
            wf.writelines('\t')
            wf.writelines(str(boxes[1]))
            wf.writelines('\t')
            wf.writelines(str(boxes[2]))
            wf.writelines('\t')
            wf.writelines(str(boxes[3]))
            wf.writelines('\n')
        #     cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0), 2)
        # for boxes in need_print_boxes:
        #     cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (255, 0, 0), 2)

    cv2.imwrite(os.path.join(filted_dir, img_name), img)

# a = [[1,2],[3,4],[5,6]]
# b = [1,2].copy()
# print(b in a)
