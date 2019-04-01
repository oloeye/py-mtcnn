![](https://pic1.zhimg.com/80/v2-b0df6adfef0889fb613bf51c6a3e409c_hd.jpg)

O-Net的训练数据生成类似于RNet，包括面部轮廓关键点数据和通过P-Net和R-Net后对每个图片检测出来的bounding boxes。O-Net生成大小为2的回归框分类特征；大小为4的回归框位置的回归特征；大小为10的人脸轮廓位置回归特征。

分数超过阈值的候选框对应的Bbox回归数据以及landmark数据进行保存。

将Bbox回归数据以及landmark数据映射到原始图像坐标上。

再次实施nms对人脸框进行合并。

经过这层层筛选合并后，最终剩下的Bbox以及其对应的landmark就是我们苦苦追求的结果了。

```python
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
```

