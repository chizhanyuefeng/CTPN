import cv2
import shapely
import numpy as np

from shapely.geometry import Polygon, MultiPoint


def polygon_from_list(line):
    """
    给定点的list坐标，返回多边形polygon对象
    :param line:
    :return:
    """
    # polygon_points = [float(o) for o in line.split(',')[:8]]
    polygon_points = np.array(line).reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon


def polygon_iou(pred_box, gt_box):
    """
    计算预测和gt的iou
    :param pred_box: list [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :param gt_box: list [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :return:
    """
    pred_polygon_points = np.array(pred_box).reshape(4, 2)
    pred_poly = Polygon(pred_polygon_points).convex_hull
    gt_polygon_points = np.array(gt_box).reshape(4, 2)
    gt_poly = Polygon(gt_polygon_points).convex_hull
    union_poly = np.concatenate((pred_polygon_points, gt_polygon_points))

    if not pred_poly.intersects(gt_poly):
        iou = 0
    else:
        try:
            inter_area = pred_poly.intersection(gt_poly).area
            # union_area = pred_box.area + gt_box.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def polygon_riou(pred_box, gt_box):
    """
    计算预测和gt的riou
    :param pred_box: list [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :param gt_box: list [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :return:
    """
    pred_polygon_points = np.array(pred_box).reshape(4, 2)
    pred_poly = Polygon(pred_polygon_points).convex_hull
    gt_polygon_points = np.array(gt_box).reshape(4, 2)
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


def polygon_area(pts_list):
    polygon_points = np.array(pts_list).reshape(4, 2)
    poly = Polygon(polygon_points).convex_hull
    return poly.area


if __name__ == "__main__":
    box1 = [[50, 40], [120, 20], [160, 80], [40, 120]]
    print(polygon_area(box1))
    box2 = [[110, 60], [180, 60], [180, 160], [110, 160]]
    print(polygon_area(box2))

    print(polygon_iou(box1, box2))

    img = np.zeros([200, 200, 3]) + 255

    cv2.polylines(img, [np.array(box1)], True, (255, 0, 0))
    cv2.polylines(img, [np.array(box2)], True, (0, 255, 0))

    union = [np.array([[110, 96], [160, 80], [146, 60], [110, 60]])]

    cv2.polylines(img, union, True, (0, 0, 255))

    cv2.imshow('d', img)
    cv2.waitKey()