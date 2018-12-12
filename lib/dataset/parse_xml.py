import os
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import tqdm
import logging
import time

img_path = '/home/tony/Downloads/第五批测试集/原始图片'
xml_path = '/home/tony/Downloads/第五批测试集/生成的xml文件'
res_path = '/home/tony/Downloads/第五批测试集/res'
# log_path = '/home/tony/ocr/ocr_dataset/souti1/log'
#
#
#
# logger = logging.getLogger('xml')
# logger.setLevel(logging.DEBUG)
#
# # 添加文件输出
# log_file = log_path + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.logs'
# file_handler = logging.FileHandler(log_file, mode='w')
# file_handler.setLevel(logging.DEBUG)
# file_formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# file_handler.setFormatter(file_formatter)
# logger.addHandler(file_handler)
#
# # 添加控制台输出
# consol_handler = logging.StreamHandler()
# consol_handler.setLevel(logging.DEBUG)
# consol_formatter = logging.Formatter('%(message)s')
# consol_handler.setFormatter(consol_formatter)
# logger.addHandler(consol_handler)


class ParseXml(object):

    def __init__(self, xml_path):
        self.classes = []
        self.bbox = []
        self.img_name = xml_path.split('/')[-1].replace('.xml', '')
        # print(xml_path)
        self.res = self._read_xml(xml_path)

    def get_bbox_class(self):

        if self.res is True:
            return self.img_name, self.classes, self.bbox
        else:
            return self.img_name, None, None

    def _read_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        itmes = root.findall("outputs/object/item")

        for i in itmes:
            res = self._parse_item(i)
            if res is False:
                return False
        return True

    def _parse_item(self, item):
        class_elem = item.find('name')

        if item.find('bndbox'):
            bbox = []
            bndbox = item.find('bndbox')
            if (bndbox.find('xmin') is None) or (bndbox.find('ymin') is None) or \
                (bndbox.find('xmax') is None) or (bndbox.find('ymax') is None):
                logger.info('%s.xml文件中bbox有问题' % (self.img_name))
                return True

            bbox.append(int(bndbox.find('xmin').text))
            bbox.append(int(bndbox.find('ymin').text))
            bbox.append(int(bndbox.find('xmax').text))
            bbox.append(int(bndbox.find('ymax').text))
            self.bbox.append(bbox)
            self.classes.append(int(float(class_elem.text)))
            return True
        elif item.find('polygon'):
            pos = []
            polygon = item.find('polygon')
            pos.append(int(polygon.find('x1').text))
            pos.append(int(polygon.find('y1').text))

            if polygon.find('x2') is not None:
                pos.append(int(polygon.find('x2').text))
                pos.append(int(polygon.find('y2').text))
            else:
                logger.info('%s.xml文件中多边形框选有问题,少点' % (self.img_name))
                return False
            if polygon.find('x3') is not None:
                pos.append(int(polygon.find('x3').text))
                pos.append(int(polygon.find('y3').text))
            else:
                logger.info('%s.xml文件中多边形框选有问题,少点' % (self.img_name))
                return False
            if polygon.find('y4') is not None:
                pos.append(int(polygon.find('x4').text))
                pos.append(int(polygon.find('y4').text))
                self.bbox.append(pos)
                self.classes.append(int(class_elem.text))
            else:
                logger.info('%s.xml文件中多边形框选有问题,少点' % (self.img_name))
                return False

            if polygon.find('x5'):
                logger.info('%s.xml文件中多边形框选有问题,少点' % (self.img_name))
                return False

            return True
        else:
            logger.info('%s.xml文件中含有其他类型未知框' % (self.img_name))
            return False

def draw_bbox(img_name, class_list, bbox_list):
    debug = 0

    # if 'hs (3)' in img_name:
    #     debug = 1

    img_name_list = glob.glob(os.path.join(img_path, img_name+'.*'))
    if len(img_name_list) != 1:
        assert 0, 'img_name_list:{}'.format(img_name_list)


    if os.name == 'posix':
        img_read_name = img_name_list[0].split('/')[-1]
    else:
        img_read_name = img_name_list[0].split('\\')[-1]
    # print(img_read_name)

    img = cv2.imread(os.path.join(img_path, img_read_name))
    # print(bbox_list)
    # cv2.imshow('2', img)
    # cv2.waitKey()


    print_color = (0, 0, 255)
    hand_color = (255, 0, 0)
    merage_color = (0, 255, 0)
    # print(class_list)
    for i in range(len(class_list)):
        if class_list[i] == 0:
            color = print_color
        elif class_list[i] == 1:
            color = hand_color
        elif class_list[i] == 2:
            color = merage_color
        else:
            assert 0,'出现错误class编号:{}'.format(class_list[i])

        if len(bbox_list[i]) == 4:
            cv2.rectangle(img, (bbox_list[i][0], bbox_list[i][1]), (bbox_list[i][2], bbox_list[i][3]), color, 2)
        else:
            cv2.line(img, (bbox_list[i][0], bbox_list[i][1]),
                     (bbox_list[i][2], bbox_list[i][3]), color, 2)
            cv2.line(img, (bbox_list[i][2], bbox_list[i][3]),
                     (bbox_list[i][4], bbox_list[i][5]), color, 2)
            cv2.line(img, (bbox_list[i][4], bbox_list[i][5]),
                     (bbox_list[i][6], bbox_list[i][7]), color, 2)
            cv2.line(img, (bbox_list[i][6], bbox_list[i][7]),
                     (bbox_list[i][0], bbox_list[i][1]), color, 2)
        if debug:
            cv2.imshow('w', img)
            cv2.waitKey()
    cv2.imwrite(os.path.join(res_path, img_read_name), img)



if __name__ =="__main__":

    # xml_name = os.listdir(xml_path)
    # for i in tqdm.tqdm(range(len(xml_name))):
    #     name = os.path.join(xml_path, xml_name[i])
    #     p = ParseXml(name)
    #     img_name, class_list, bbox_list = p.get_bbox_class()
    #     if class_list is not None:
    #         # print(len(bbox_list), len(class_list))
    #         draw_bbox(img_name, class_list, bbox_list)

    cv2.minAreaRect('')


    # random_scale_inds = np.random.randint(0, high=len([600]),
    #                                 size=100)
    # print(random_scale_inds)