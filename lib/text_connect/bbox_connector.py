import numpy as np
import cv2
import os

# 高度上下浮动像素值
HEIGHT_FLOAT_PIXEL = 25
# left bbox x2 与right bbox x1 差值的绝对值
WIDTH_FLOAT_PIXEL = 20
# left bbox x1 与right bbox x1 差值的绝对值
LEFT_RIGHT_X1_FLOAT_PIXEL = 20

class BboxConnector(object):
    def __init__(self, bbox_list):

        self.bboxes = np.array(bbox_list,dtype=np.int)
        # self.img = img
        # self.img_width = img.shape[1]
        # self.img_height = img.shape[0]
        self.center_height = (self.bboxes[:, 1] + self.bboxes[:, 3])/2
        self.center_width = (self.bboxes[:, 0] + self.bboxes[:, 2])/2
        self.bboxes_num = len(bbox_list)
        self.bbox_is_conected = np.array([False] * self.bboxes_num)
        self.result_bbox_list = []

    def _find_vertical_groups(self):
        """
        vertical_groups中存储的[bbox_num, n] n个bbox的index
        :return:
        """
        self.vertical_groups = []
        for i in range(self.bboxes_num):
            group = []
            group.append(i)
            for j in range(self.bboxes_num):
                if i!=j:
                    if abs(self.center_height[i] - self.center_height[j])<=HEIGHT_FLOAT_PIXEL:
                        group.append(j)
            self.vertical_groups.append(group)

    def start(self):
        self._find_vertical_groups()
        for group in self.vertical_groups:
            filter = []
            for i in range(len(group)):
                if self.bbox_is_conected[group[i]]==True:
                    filter.append(i)
            for i in range(len(filter)):
                group.pop(filter[i]-i)

            if len(group)<=1:
                continue
            self.sort_group_by_x1(group)
            self.connect_group_bboxes(group)

        not_connect_bbox = np.asarray(self.bboxes[np.where(self.bbox_is_conected==False)])
        self.result_bbox_list.extend(not_connect_bbox)
        return self.result_bbox_list

    def connect_group_bboxes(self, group):
        group_len = len(group)
        start = 0
        while start < group_len-1:
            left_bbox_height = self.bboxes[group[start]][3] - self.bboxes[group[start]][1]
            right_bbox_height = self.bboxes[group[start+1]][3] - self.bboxes[group[start+1]][1]
            if (left_bbox_height <= right_bbox_height) and \
                    (self.bboxes[group[start+1]][0] - self.bboxes[group[start]][2])<=WIDTH_FLOAT_PIXEL and \
                    abs(self.bboxes[group[start+1]][0] - self.bboxes[group[start]][0])>=LEFT_RIGHT_X1_FLOAT_PIXEL:
                #进行融合
                bbox = []
                bbox.append(min(self.bboxes[group[start]][0], self.bboxes[group[start + 1]][0]))
                bbox.append(min(self.bboxes[group[start]][1], self.bboxes[group[start + 1]][1]))
                bbox.append(max(self.bboxes[group[start]][2], self.bboxes[group[start + 1]][2]))
                bbox.append(max(self.bboxes[group[start]][3], self.bboxes[group[start + 1]][3]))
                self.result_bbox_list.append(bbox)
                self.bbox_is_conected[group[start]] = True
                self.bbox_is_conected[group[start+1]] = True

                start += 2
            else:
                start += 1


    def sort_group_by_x1(self, group):
        for i in range(len(group)-1):
            for j in range(len(group)-i-1):
                if self.bboxes[group[j]][0] > self.bboxes[group[j + 1]][0]:
                    group[j], group[j + 1] = group[j + 1], group[j]



if __name__ == "__main__":
    img_dir = '/home/tony/ocr/Arithmetic_Func_detection_for_CTPN_v1/data/demo/'
    bbox_dir = '/home/tony/ocr/Arithmetic_Func_detection_for_CTPN_v1/data/results/'
    img_list = os.listdir(img_dir)

    for img_name in img_list:
        img = cv2.imread(os.path.join(img_dir, img_name))
        bbox = []
        with open(os.path.join(bbox_dir, img_name.split('.')[0]+'.txt')) as f:
            lines = f.readlines()
            for line in lines:
                x1,y1,x2,y2 = line.split(',')
                box = []
                box.append(float(x1))
                box.append(float(y1))
                box.append(float(x2))
                box.append(float(y2))
                bbox.append(box)
                #cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))

        c = BboxConnector(bbox)
        connected_bbox_list = c.start()
        for i in connected_bbox_list:
            cv2.rectangle(img, (i[0],i[1]),(i[2],i[3]),(255,0,0),2)

        cv2.imwrite(img_name, img)


    # img = cv2.imread('/home/tony/ocr/ocr_dataset/tal_detec_data_v2/img/hs (1).jpg')
    #
    # p = ParseXml('/home/tony/ocr/ocr_dataset/tal_detec_data_v2/xml/hs (1).xml',True)
    # img_name, classes, bbox = p.get_bbox_class()
    # # print(bbox)
    # c = BboxConnector(bbox, img)
    #
    # bbox = c.start()
    #
    # for i in bbox:
    #     cv2.rectangle(img, (i[0],i[1]),(i[2],i[3]),(255,0,0))
    #
    # cv2.imwrite('123.jpg',img)






