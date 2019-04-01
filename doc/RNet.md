![](https://img-blog.csdn.net/20170614104132727?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZnV3ZW55YW4=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

如果你解决了`P-Net`网络，那么`R-Net`网络理解起来就比较简单了。`R-Net`仍然只做检测和人脸框回归两个任务。忽略下图中的`Facial landmark`。只是对`P-Net`生成的候选框再次筛选，获取更好的人脸候选框。

**大楷步骤**：把P-Net生成的候选框，resize()到24x24x3大小图片后输入R-Net网络，通过R-Net网络最后输出BBox候选框。

```python
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
```

回归处理大概如下

![](https://pic2.zhimg.com/80/v2-7acab54b76a6982f34c919f13335e7a1_hd.jpg)

