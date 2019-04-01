# -*- coding: utf-8 -*-
# @Author: Hobo
# @Date:   2018-11-26 16:39:47
# @Last Modified by:   Hobo
# @Last Modified time: 2019-03-21 09:56:06

import cv2
import numpy as np


def non_max_suppression(bboxes, threshold=0.5, mode='union'):
    '''Non max suppression.
    Args:
      bboxes: (tensor) bounding boxes and scores sized [N, 5].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.
    Returns:
      Bboxes after nms.
      Picked indices.
    Ref:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]

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
        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return bboxes[keep], keep


def bbox_regression(bboxes):
    '''Bounding box regression.
    Args:
      bboxes: (tensor) bounding boxes sized [N,9], containing:
        x1, y1, x2, y2, score, regy1, regx1, regy2, regx2.
    Return:
      Regressed bounding boxes sized [N,5].
    '''
    bbw = bboxes[:, 2] - bboxes[:, 0] + 1
    bbh = bboxes[:, 3] - bboxes[:, 1] + 1

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    scores = bboxes[:, 4]

    # Note the sequence.
    rgy1 = bboxes[:, 5]
    rgx1 = bboxes[:, 6]
    rgy2 = bboxes[:, 7]
    rgx2 = bboxes[:, 8]

    ret = np.vstack([x1 + rgx1 * bbw,
                     y1 + rgy1 * bbh,
                     x2 + rgx2 * bbw,
                     y2 + rgy2 * bbh,
                     scores])
    return ret.T


def padding(bboxes, im_width, im_height):
    '''如果填充图像的边缘太大，则填充图像边缘.'''
    bboxes[:, 0] = np.maximum(0, bboxes[:, 0])
    bboxes[:, 1] = np.maximum(0, bboxes[:, 1])
    bboxes[:, 2] = np.minimum(im_width - 1, bboxes[:, 2])
    bboxes[:, 3] = np.minimum(im_height - 1, bboxes[:, 3])
    return bboxes


def bbox_to_square(bboxes):
    '''使边框成方形.'''
    square_bbox = bboxes.copy()

    w = bboxes[:, 2] - bboxes[:, 0] + 1
    h = bboxes[:, 3] - bboxes[:, 1] + 1
    max_side = np.maximum(h, w)

    square_bbox[:, 0] = bboxes[:, 0] + (w - max_side) * 0.5
    square_bbox[:, 1] = bboxes[:, 1] + (h - max_side) * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1

    return square_bbox


def draw_and_show(imgMat, bboxes, points=[]):
    '''Draw bboxes and points on image, and show.
    Args:
      img: image to draw on.
      bboxes: (tensor) bouding boxes sized [N,4].
      points: (tensor) landmark points sized [N,10],
        coordinates arranged as [x,x,x,x,x,y,y,y,y,y].
    '''
    print('Drawing..')

    num_boxes = bboxes.shape[0]
    for i in range(num_boxes):
        box = bboxes[i]
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        print('Rect:', y1, x1, y2, x2)
        # 由于im是旋转的，所以需要交换x和y.
        cv2.rectangle(imgMat, (y1, x1), (y2, x2), (0, 255, 255), 2)

        if len(points):
            p = points[i]
            for i in range(5):
                x = int(p[i])
                y = int(p[i + 5])
                # 再次，交换x和y。
                cv2.circle(imgMat, (y, x), 1, (0, 0, 255), 2)

    cv2.imshow('result', imgMat)
    cv2.waitKey(0)

def process_bbox(imgMat, bboxes,sizeWH):
    imCoreList = []
    for i in range(bboxes.shape[0]):
        x1 = int(bboxes[i, 0])
        y1 = int(bboxes[i, 1])
        x2 = int(bboxes[i, 2])
        y2 = int(bboxes[i, 3])
        im_crop = imgMat[y1:y2 + 1, x1:x2 + 1, :]
        imCoreList.append(cv2.resize(im_crop, (sizeWH, sizeWH)))
    return imCoreList