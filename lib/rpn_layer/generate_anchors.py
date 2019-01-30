import numpy as np
from lib.utils.config import cfg

def generate_basic_anchors(sizes):
    base_size = cfg["ANCHOR_WIDTH"]
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
    # anchors shape =[10,4]
    anchors = np.zeros((len(sizes), 4), np.int32)
    index = 0
    for h, w in sizes:
        anchors[index] = scale_anchor(base_anchor, h, w)
        index += 1
    return anchors


def scale_anchor(anchor, h, w):
    x_ctr = (anchor[0] + anchor[2]) * 0.5
    y_ctr = (anchor[1] + anchor[3]) * 0.5
    scaled_anchor = anchor.copy()
    scaled_anchor[0] = x_ctr - w / 2  # xmin
    scaled_anchor[2] = x_ctr + w / 2  # xmax
    scaled_anchor[1] = y_ctr - h / 2  # ymin
    scaled_anchor[3] = y_ctr + h / 2  # ymax
    return scaled_anchor


def generate_anchors():
    heights = cfg["ANCHOR_HEIGHT"]
    width = cfg["ANCHOR_WIDTH"]

    sizes = []
    for h in heights:
        sizes.append((h, width))
    return generate_basic_anchors(sizes)

if __name__ == '__main__':
    a = generate_anchors()
    print(a)
