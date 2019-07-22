import cv2
import numpy as np

def resize_img(img, scale=600, max_scale=1200):
    ratio = float(scale) / min(img.shape[0], img.shape[1])
    if max_scale != None and ratio * max(img.shape[0], img.shape[1]) > max_scale:
        ratio = float(max_scale) / max(img.shape[0], img.shape[1])

    iiimg = cv2.resize(img, None, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
    return iiimg, ratio

def img_normailize(img, method="PIXEL_MEANS"):
    img = img.astype(np.float32, copy=True)
    if method == "PIXEL_MEANS":
        means = np.array([[[102.9801, 115.9465, 122.7717]]])
        img -= means
    else:
        img = img / 255.0 * 2 - 1
    return img
