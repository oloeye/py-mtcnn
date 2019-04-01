# -*- coding: utf-8 -*-
# @Author: Hobo
# @Date:   2018-11-26 09:28:24
# @Last Modified by:   Hobo
# @Last Modified time: 2019-03-21 15:31:10

import os
import cv2
import math
import utlis
import numpy as np

np.set_printoptions(threshold=np.inf)

# 缩放尺寸最小不能小于12，也就是缩放到12为止
P_NET_WINDOW_SIZE = 12.0
P_NET_STRIDE = 2

IMG_MEAN = 127.5
IMG_INV_STDDEV = 1.0 / 128.0


class Pnet(object):

    def __init__(self, protoText, caffeModel, threshold):
        self._PNet = cv2.dnn.readNetFromCaffe(protoText, caffeModel)
        if not self._PNet:
            print("无效的protoText或caffeModel")
        self._threshold = threshold

    def getPnetBoxes(self, confidence, regression, scaleFactor, threshold):
        # np.where(condition,x,y) 只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标
        # (等价于numpy.nonzero)。这里的坐标以tuple的形式给出，通常原数组有多少维，
        # 输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标

        # 满足条件(condition)，输出x，不满足输出y。 np.where(condition, x, y)
        # np.transpose在不指定参数是默认是矩阵转置

        y, x = np.where(confidence > threshold)

        scores = confidence[y, x]

        reg_y1, reg_x1, reg_y2, reg_x2 = [
            regression[i, y, x] for i in range(4)]
        reg = np.array([reg_x1, reg_y1, reg_x2, reg_y2])

        # Get face rects. np.round #四舍五入
        x1 = np.round((P_NET_STRIDE * x + 1) / scaleFactor)
        y1 = np.round((P_NET_STRIDE * y + 1) / scaleFactor)
        x2 = np.round(
            (P_NET_STRIDE * x + 1 + P_NET_WINDOW_SIZE - 1) / scaleFactor)
        y2 = np.round(
            (P_NET_STRIDE * y + 1 + P_NET_WINDOW_SIZE - 1) / scaleFactor)

        rect = np.array([x1, y1, x2, y2])

        bbox = np.vstack([rect, scores, reg])  # [9,N]

        return bbox.T  # [N,9]

    def run(self, imgMat, minFaceSize, scaleFactor):
        finalFaces = []  # 所有尺度的边界框 Bounding boxes of all scales
        # 获取图片像素的行数和列数
        rows = imgMat.shape[0]
        cols = imgMat.shape[1]
        maxFaceSize = min(rows, cols)

        faceSize = minFaceSize

        while faceSize <= maxFaceSize:
            currentScale = P_NET_WINDOW_SIZE / faceSize
            imgH = math.ceil(rows * currentScale)  # 向上取整 貌似有问题哟
            imgW = math.ceil(cols * currentScale)

            resizeImg = cv2.resize(
                imgMat, (imgW, imgH), interpolation=cv2.INTER_AREA)

            # cv2.imwrite(str(faceSize) + '.png', resizeImg)

            # 1.参数为图像数据，2.为缩放因子，3.为缩放后图像大小，4.为均值，5.为是否交换颜色通道标志，6.为是否进行裁剪操作
            pBlob = cv2.dnn.blobFromImage(resizeImg, IMG_INV_STDDEV, (imgW, imgH), [
                IMG_MEAN, IMG_MEAN, IMG_MEAN], False)

            self._PNet.setInput(pBlob)
            outputBlobs = self._PNet.forward(["conv4-2", "prob1"])

            confidence = outputBlobs[1][0][1]  # 置信度
            regression = outputBlobs[0][0]  # 回归

            # cv2.imshow(str(faceSize) + '.png', resizeImg)
            # cv2.waitKey()

            faces = self.getPnetBoxes(
                confidence, regression, currentScale, self._threshold)

            bboxes, _ = utlis.non_max_suppression(faces, 0.5)
            finalFaces.append(bboxes)

            faceSize /= scaleFactor  # 从12变到图像的 最小长或宽

        if finalFaces:
            totalBoxes = np.vstack(finalFaces)
            bboxes, _ = utlis.non_max_suppression(totalBoxes, 0.7)
            bboxes = utlis.bbox_regression(bboxes)

            bboxes = utlis.bbox_to_square(bboxes)
            bboxes = utlis.padding(bboxes, cols, rows)
            if bboxes.shape[0] == 0:
                return
            return bboxes


if __name__ == '__main__':
    presentPath = os.path.abspath(os.path.dirname(__file__))
    modelFile = presentPath + "/data/models/" + \
        "det1.caffemodel"
    configFile = presentPath + "/data/models/" + "det1.prototxt"

    testImgpath = presentPath + "/data/" + "download.png"

    img = cv2.imread(testImgpath)
    img_bk = img.copy()
    if (img.shape[2] == 3):
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape[2])
    elif (img.shape[2] == 4):
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        print(img.shape[2])

    rgbImg = rgbImg.astype(np.float32)
    # to meet the Caffe needs. w,h,c -> h,w,c
    rgbImg = np.transpose(rgbImg, (1, 0, 2))
    # THRESHOLD = [0.6, 0.7, 0.7]
    facer = Pnet(configFile, modelFile, 0.6)
    bboxes = facer.run(rgbImg, 16, 0.709)

    utlis.draw_and_show(img_bk, bboxes)
