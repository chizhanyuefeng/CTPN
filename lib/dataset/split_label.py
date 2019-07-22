import os
import sys

import cv2 as cv
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())
from lib.dataset.utils import orderConvex, shrink_poly

DATA_FOLDER = "./data"
DATA_IMAGE_DIR = os.path.join(DATA_FOLDER, 'image')
DATA_LABEL_DIR = os.path.join(DATA_FOLDER, 'label')


MAX_LEN = 1200
MIN_LEN = 600
WIDTH = 8

im_fns = os.listdir(os.path.join(DATA_FOLDER, "image"))
im_fns.sort()

OUTPUT = "./data/ctpn_train"

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)
if not os.path.exists(os.path.join(OUTPUT, "image")):
    os.makedirs(os.path.join(OUTPUT, "image"))
if not os.path.exists(os.path.join(OUTPUT, "label")):
    os.makedirs(os.path.join(OUTPUT, "label"))

for im_fn in tqdm(im_fns):
    try:
        _, fn = os.path.split(im_fn)
        bfn, ext = os.path.splitext(fn)
        if ext.lower() not in ['.jpg', '.png', '.jpeg']:
            continue

        gt_path = os.path.join(DATA_FOLDER, "label", bfn + '.txt')
        img_path = os.path.join(DATA_FOLDER, "img", im_fn)

        img = cv.imread(img_path)
        img_size = img.shape
        im_size_min = np.min(img_size[0:2])
        im_size_max = np.max(img_size[0:2])

        im_scale = float(600) / float(im_size_min)
        if np.round(im_scale * im_size_max) > 1200:
            im_scale = float(1200) / float(im_size_max)
        new_h = int(img_size[0] * im_scale)
        new_w = int(img_size[1] * im_scale)
        #
        # new_h = new_h if new_h // WIDTH == 0 else (new_h // WIDTH + 1) * WIDTH
        # new_w = new_w if new_w // WIDTH == 0 else (new_w // WIDTH + 1) * WIDTH

        re_im = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        re_size = re_im.shape

        polys = []
        cls_list = []
        with open(gt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            x1, y1, x2, y2, x3, y3, x4, y4, cls = line.split(',')
            cls = cls.replace('\n', '')
            poly = np.array([float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)]).reshape([4, 2])
            poly[:, 0] = poly[:, 0] / img_size[1] * re_size[1]
            poly[:, 1] = poly[:, 1] / img_size[0] * re_size[0]
            poly = orderConvex(poly)
            polys.append(poly)
            cls_list.append(cls)

        res_polys = []
        res_cls_list = []
        for i, poly in enumerate(polys):
            # delete polys with width less than 10 pixel
            if np.linalg.norm(poly[0] - poly[1]) < 5 or np.linalg.norm(poly[3] - poly[0]) < 5:
                continue

            res = shrink_poly(poly, r=WIDTH)

            res = res.reshape([-1, 4, 2])
            res_num = res.shape[0]
            for r in res:
                x_min = np.min(r[:, 0])
                y_min = np.min(r[:, 1])
                x_max = np.max(r[:, 0])
                y_max = np.max(r[:, 1])

                res_polys.append([x_min, y_min, x_max, y_max])

            res_cls_list.extend([cls_list[i]]*res_num)

        cv.imwrite(os.path.join(OUTPUT, "image", fn), re_im)
        with open(os.path.join(OUTPUT, "label", bfn) + ".txt", "w") as f:
            for i, p in enumerate(res_polys):
                line = "\t".join(str(p[i]) for i in range(4))
                line = res_cls_list[i] + '\t' + line
                f.writelines(line + "\r\n")

            for i, p in enumerate(res_polys):
                if res_cls_list[i] == 'print':
                    color = (255, 0, 0)
                elif res_cls_list[i] == 'handwritten':
                    color = (0, 255, 0)
                else:
                    assert 0, 'error cls'
                cv.rectangle(re_im,(p[0],p[1]),(p[2],p[3]),color=color,thickness=1)

    except:
        print("Error processing {}".format(im_fn))
