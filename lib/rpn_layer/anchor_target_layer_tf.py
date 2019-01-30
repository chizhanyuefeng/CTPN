# -*- coding:utf-8 -*-
import cv2
import numpy as np
import numpy.random as npr
from lib.rpn_layer.generate_anchors import generate_anchors
from lib.bbox_utils.bbox import bbox_overlaps
from lib.utils.config import cfg
from lib.bbox_utils.bbox_transform import bbox_transform

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, img=None):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    Parameters
    ----------
    rpn_cls_score: (1, H, W, Ax2) bg/fg scores of previous conv layer
    gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
    im_info: a list of [image_height, image_width, scale_ratios]
    _feat_stride: the downsampling ratio of feature map to the original input image
    ----------
    Returns
    ----------
    rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
    rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
                            that are the regression objectives
    rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
    rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
                            beacuse the numbers of bgs and fgs mays significiantly different
    """
    # print(gt_boxes)
    _feat_stride = [cfg["ANCHOR_WIDTH"], ]
    _anchors = generate_anchors()  # 生成基本的anchor,一共10个
    _num_anchors = _anchors.shape[0]  # 10个anchor

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    im_info = im_info[0]  # 图像的高宽及通道数

    # 在feature-map上定位anchor，并加上delta，得到在实际图像中anchor的真实坐标
    # Algorithm:
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]  # feature-map的高宽
    # print('feature shape:', rpn_cls_score.shape)
    # print('img info', im_info)

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y) # in W H order
    # K is H x W
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()  # 生成feature-map和真实image上anchor之间的偏移量
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors  # 10个anchor
    K = shifts.shape[0]  # 50*37，feature-map的宽乘高的大小
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))  # 相当于复制宽高的维度，然后相加
    all_anchors = all_anchors.reshape((K * A, 4))
    # print(all_anchors)
    total_anchors = int(K * A)

    # only keep anchors inside the image
    # 仅保留那些还在图像内部的anchor，超出图像的都删掉
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]
    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]  # 保留那些在图像内的anchor

    # 至此，anchor准备好了
    # --------------------------------------------------------------
    # label: 1 is positive, 0 is negative,
    # (A)
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)  # 初始化label，均为-1

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt), shape is A x G
    # 计算anchor和gt-box的overlap，用来给anchor上标签
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))  # 假设anchors有x个，gt_boxes有y个，返回的是一个（x,y）的数组
    # 存放每一个anchor和每一个gtbox之间的overlap
    argmax_overlaps = overlaps.argmax(axis=1)  # 找到和每一个gtbox，overlap最大的那个anchor
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)  # 找到每个位置上10个anchor中与gtbox，overlap最大的那个
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg["TRAIN"]["RPN_CLOBBER_POSITIVES"]:
        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < cfg["TRAIN"]["RPN_NEGATIVE_OVERLAP"]] = 0  # 先给背景上标签，小于0.3overlap的

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1  # 每个位置上的10个anchor中overlap最大的认为是前景
    # fg label: above threshold IOU
    labels[max_overlaps >= cfg["TRAIN"]["RPN_POSITIVE_OVERLAP"]] = 1  # overlap大于0.7的认为是前景

    if cfg["TRAIN"]["RPN_CLOBBER_POSITIVES"]:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg["TRAIN"]["RPN_NEGATIVE_OVERLAP"]] = 0

    # subsample positive labels if we have too many
    # 对正样本进行采样，如果正样本的数量太多的话
    # 限制正样本的数量不超过128个
    # TODO: 这个后期可能还需要修改，毕竟如果使用的是字符的片段，那个正样本的数量是很多的。
    num_fg = int(cfg["TRAIN"]["RPN_FG_FRACTION"] * cfg["TRAIN"]["RPN_BATCHSIZE"])
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)  # 随机去除掉一些正样本
        labels[disable_inds] = -1  # 变为-1

    # subsample negative labels if we have too many
    # 对负样本进行采样，如果负样本的数量太多的话
    # 正负样本总数是300，限制正样本数目最多150，
    # 如果正样本数量小于150，差的那些就用负样本补上，凑齐300个样本
    num_bg = cfg["TRAIN"]["RPN_BATCHSIZE"] - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    # 至此， 上好标签，开始计算rpn-box的真值
    # --------------------------------------------------------------
    # 根据anchor和gtbox计算得真值（anchor和gtbox之间的偏差）
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # 内部权重，前景就给1，其他是0
    bbox_inside_weights[labels == 1, :] = np.array(cfg["TRAIN"]["RPN_BBOX_INSIDE_WEIGHTS"])

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg["TRAIN"]["RPN_POSITIVE_WEIGHT"] < 0:  # 暂时使用uniform 权重，也就是正样本是1，负样本是0
        # uniform weighting of examples (given non-uniform sampling)
        positive_weights = np.ones((1, 4))
        negative_weights = np.zeros((1, 4))
    else:
        assert ((cfg["TRAIN"]["RPN_POSITIVE_WEIGHT"] > 0) &
                (cfg["TRAIN"]["RPN_POSITIVE_WEIGHT"] < 1))
        positive_weights = (cfg["TRAIN"]["RPN_POSITIVE_WEIGHT"] /
                            (np.sum(labels == 1)) + 1)
        negative_weights = ((1.0 - cfg["TRAIN"]["RPN_POSITIVE_WEIGHT"]) /
                            (np.sum(labels == 0)) + 1)
    # 外部权重，前景是1，背景是0
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    # 一开始是将超出图像范围的anchor直接丢掉的，现在在加回来
    # 这些anchor的label是-1，也即dontcare
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    # 这些anchor的真值是0，也即没有值
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    # 内部权重以0填充
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    # 外部权重以0填充
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # reshap一下label
    labels = labels.reshape((1, height, width, A))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))
    rpn_bbox_targets = bbox_targets

    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))
    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))
    rpn_bbox_outside_weights = bbox_outside_weights

    # print("rpn_labels", np.where(rpn_labels==1))
    # print("rpn_bbox_targets", rpn_bbox_targets[0])
    # print("rpn_bbox_inside_weights", rpn_bbox_inside_weights[0])
    # print("rpn_bbox_outside_weights", rpn_bbox_outside_weights[0])

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights



def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)


if __name__ == "__main__":
    img = cv2.imread('/home/zzh/ocr/CTPN/CTPN-tf/10pic/train_image/IMG_2355.jpg')

    im_info = [800, 600, 1]
    _feat_stride = [16, ]

    _anchors = generate_anchors()  # 生成基本的10个anchor
    _num_anchors = _anchors.shape[0]  # 10个anchor

    im_info = im_info[0]  # 原始图像的高宽、缩放尺度

    rpn_cls_prob_shape = [1, 50, 370, 2]

    pre_nms_topN = cfg["TEST"]["RPN_PRE_NMS_TOP_N"]  # 12000,在做nms之前，最多保留的候选box数目
    post_nms_topN = cfg["TEST"]["RPN_POST_NMS_TOP_N"]  # 2000，做完nms之后，最多保留的box的数目
    nms_thresh = cfg["TEST"]["RPN_NMS_THRESH"]  # nms用参数，阈值是0.7
    min_size = cfg["TEST"]["RPN_MIN_SIZE"]  # 候选box的最小尺寸，目前是16，高宽均要大于16

    height, width = rpn_cls_prob_shape[1:3]  # feature-map的高宽
    width = width // 10

    # Enumerate all shifts
    # 同anchor-target-layer-tf这个文件一样，生成anchor的shift，进一步得到整张图像上的所有anchor
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride

    # shift_x shape = [height, width]
    # 生成同样维度的两个矩阵
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # print("shift_x", shift_x.shape)
    # print("shift_y", shift_y.shape)
    # shifts shape = [height*width,4]
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors  # 10
    K = shifts.shape[0]  # height*width,[height*width,4]
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))  # 这里得到的anchor就是整张图像上的所有anchor
    print(anchors)

    for box in anchors:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))

    cv2.imshow('d', img)
    cv2.waitKey()