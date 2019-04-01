# -*- coding: utf-8 -*-
# @Author: Hobo
# @Date:   2018-11-26 09:28:41
# @Last Modified by:   Hobo
# @Last Modified time: 2018-11-26 09:49:20
import cv2
import utlis
import numpy as np

IMG_MEAN = 127.5
IMG_INV_STDDEV = 1.0 / 128.0

INPUT_DATA_WH = 24

class Rnet(object):
    def __init__(self, protoText, caffeModel, threshold):
        self._RNet = cv2.dnn.readNetFromCaffe(protoText, caffeModel)
        if not self._RNet:
            print("无效的protoText或caffeModel")
        self._threshold = threshold

    def run(self, imgMat, bboxes):
        rows = imgMat.shape[0]
        cols = imgMat.shape[1]
        reImCoreList = utlis.process_bbox(imgMat, bboxes, INPUT_DATA_WH)
        # 1.参数为图像数据列表，2.为缩放因子，3.为缩放后图像大小，4.为均值，5.为是否交换颜色通道标志，6.为是否进行裁剪操作
        pBlob = cv2.dnn.blobFromImages(reImCoreList, IMG_INV_STDDEV, None, [
            IMG_MEAN, IMG_MEAN, IMG_MEAN], False)

        self._RNet.setInput(pBlob)
        outputBlobs = self._RNet.forward(["conv5-2", "prob1"])

        confidence = outputBlobs[1][:,1]
        regression = outputBlobs[0]

        indices = np.where(confidence > self._threshold)
        rects = bboxes[indices][:, 0:4]  # [N,4]
        scores = confidence[indices]  # [N,]
        scores = scores.reshape(-1, 1)  # [N,1]
        regs = regression[indices]  # [N,4]

        opt_bboxes = np.hstack([rects, scores, regs])  # [N,9]

        bboxes, _ = utlis.non_max_suppression(opt_bboxes, 0.7)
        bboxes = utlis.bbox_regression(bboxes)
        bboxes = utlis.bbox_to_square(bboxes)
        bboxes = utlis.padding(bboxes, cols, rows)

        print('After RNet bboxes shape: ', bboxes.shape)
        if bboxes.shape[0] == 0:
            return
        return bboxes

