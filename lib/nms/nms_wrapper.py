import numpy as np
from lib.utils.config import cfg
pure_python_nms = True

try:
    #from lib.nms.gpu_nms import gpu_nms
    from lib.nms.cython_nms import nms as cython_nms
    from lib.nms.cpu_nms import cpu_nms
except ImportError:
    pure_python_nms = True


#from lib.nms.gpu_nms import gpu_nms

# from lib.nms.cpu_nms import cpu_nms

# def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):
#
#     keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
#                         np.float32(sigma), np.float32(Nt),
#                         np.float32(threshold),
#                         np.uint8(method))
#     return keep
#

def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    if pure_python_nms:
        # print("Fall back to pure python nms")
        return py_cpu_nms(dets, thresh)
    if cfg["USE_GPU_NMS"]:
        return gpu_nms(dets, thresh, device_id=0)
    else:
        return cython_nms(dets, thresh)

def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep
