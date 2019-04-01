# -*- coding: utf-8 -*-
# @Author: Hobo
# @Date:   2018-11-26 09:28:52
# @Last Modified by:   Hobo
# @Last Modified time: 2018-11-26 09:49:17
import cv2
import utlis
import numpy as np

IMG_MEAN = 127.5
IMG_INV_STDDEV = 1.0 / 128.0

INPUT_DATA_WH = 48


class Onet(object):
    def __init__(self, protoText, caffeModel, threshold):
        self._ONet = cv2.dnn.readNetFromCaffe(protoText, caffeModel)
        if not self._ONet:
            print("无效的protoText或caffeModel")
        self._threshold = threshold

    def run(self, imgMat, bboxes):
        rows = imgMat.shape[0]
        cols = imgMat.shape[1]
        reImCoreList = utlis.process_bbox(imgMat, bboxes, INPUT_DATA_WH)
        # 1.参数为图像数据列表，2.为缩放因子，3.为缩放后图像大小，4.为均值，5.为是否交换颜色通道标志，6.为是否进行裁剪操作
        pBlob = cv2.dnn.blobFromImages(reImCoreList, IMG_INV_STDDEV, None, [
            IMG_MEAN, IMG_MEAN, IMG_MEAN], False)

        self._ONet.setInput(pBlob)
        outputBlobs = self._ONet.forward(["conv6-2", 'conv6-3', "prob1"])

        regression = outputBlobs[0]
        landMark = outputBlobs[1]
        confidence = outputBlobs[2][:, 1]

        indices = np.where(confidence > self._threshold)

        rects = bboxes[indices][:, 0:4]
        scores = confidence[indices]
        scores = scores.reshape(-1, 1)
        regs = regression[indices]

        points = landMark[indices]  # Note `y` is in the front.
        points_y = points[:, 0:5]  # [N,5]
        points_x = points[:, 5:10]  # [N,5]

        w = rects[:, 2] - rects[:, 0] + 1
        h = rects[:, 3] - rects[:, 1] + 1

        x1 = rects[:, 0]
        y1 = rects[:, 1]

        points_x = points_x * w.reshape(-1, 1) + x1.reshape(-1, 1)
        points_y = points_y * h.reshape(-1, 1) + y1.reshape(-1, 1)

        # We move `x` ahead, points=[x,x,x,x,x,y,y,y,y,y].
        _bboxes, points = np.hstack([rects, scores, regs]), np.hstack([points_x, points_y])

        bboxes = utlis.bbox_regression(_bboxes)

        bboxes, picked_indices = utlis.non_max_suppression(bboxes, 0.7, 'min')
        points = points[picked_indices]
        bboxes = utlis.padding(bboxes, cols, rows)  # h,w

        print('After ONet bboxes shape: ', bboxes.shape, '\n')
        if bboxes.shape[0] == 0:
            return
        return bboxes, points
