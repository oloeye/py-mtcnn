![](https://upload-images.jianshu.io/upload_images/3403753-1429dd971042cbc0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/877/format/webp)

## PNet网络

MTCNN主要就`P-Net`、`R-Net`、`O-Net`网络，这篇文章主要详细讲解`P-Net`网络和BoundingBox的处理。

看上面的网络结构图，发现输入的是一个`12x12x3`大小图片，这只是一个标记，其实可以输入任意大小的图片，这是因为`P-Net`是一个**全卷积网络（FCN）**，卷积、池化、非线性激活都是一些可以接受任意尺度矩阵的运算，但全连接运算是需要规定输入。如果网络中有全连接层，则输入的图片尺度(一般)需固定；如果没有全连接层，图片尺度可以是任意的。同时使用卷积运算代替了滑动窗口运算，大幅提高了效率。但是传入任意大小的图片后，输出就不再是`1x1`大小的特征图了，而是一个`WxH`的特征图。每个特征图上就变成看（`WxHx2`的分类，`WxHx4`的回归，`WxHx10`轮廓点），这样就不用先从resize的图上截取各种`12x12x3`的图再送入网络了，而是一次性送入，再根据结果回推每个结果对应的`12x12`的图在输入图片的什么位置，那么怎么解决检测不同大小的人脸了，接下来就引入**图像金字塔**。

### 图像金字塔

上面的模型结构可以看出，输入最小就是`12x12`大小的彩色图片了，所以在图片缩放过程中，最小就`12x12`。MTCNN作者设置的缩放比例因子为`0.709`。只要我们的图片缩放小于等于`12x12`后就停止缩放。从而得到一系列大小不同的图片，此时，12×12的PNet就可以检测大小不同的人脸了。因为检测框大小不变，但是输入图像的尺寸发生了变化。下面是使用python 的opencv库操作Caffe模型（因为这样安装和使用比较简单）

```python
# 获取图片像素的行数和列数
rows = imgMat.shape[0]
cols = imgMat.shape[1]
maxFaceSize = min(rows, cols)

faceSize = minFaceSize   # 可以检测最小人脸

while faceSize <= maxFaceSize:
    currentScale = P_NET_WINDOW_SIZE / faceSize
    imgH = math.ceil(rows * currentScale)  # 向上取整 貌似有问题哟
    imgW = math.ceil(cols * currentScale)

    resizeImg = cv2.resize(
        imgMat, (imgW, imgH), interpolation=cv2.INTER_AREA)

    # 1.参数为图像数据，2.为缩放因子，3.为缩放后图像大小，4.为均值，5.为是否交换颜色通道标志，6.为是否进行裁剪操作
    pBlob = cv2.dnn.blobFromImage(resizeImg, IMG_INV_STDDEV, (imgW, imgH), [
        IMG_MEAN, IMG_MEAN, IMG_MEAN], False)

    self._PNet.setInput(pBlob)
    outputBlobs = self._PNet.forward(["conv4-2", "prob1"])

    confidence = outputBlobs[1][0][1]  # 是人脸置信度
    regression = outputBlobs[0][0]  # 回归坐标

    # cv2.imshow(str(faceSize) + '.png', resizeImg)
    # cv2.waitKey()

    faces = self.getPnetBoxes(
        confidence, regression, currentScale, self._threshold)

    bboxes, _ = utlis.non_max_suppression(faces, 0.5)
    finalFaces.append(bboxes)

    faceSize /= scaleFactor  # 从12变到图像的 最小长或宽 scaleFactor: 缩放比，这里设为0.709

```

可以看出上面的代码是使用循环进行图片缩放的，缩放后的图片放入caffe的`P-Net`模型中得出二分类信息，边框坐标（是相对`12x12`的坐标，后续是需要映射到原图像）。

通过`self._PNet.forward(["conv4-2", "prob1"])`产生结果后，

```python
# 这是regression 的缩减数据格式，所以在看代码时候一定需要根据输出结合查看
# regression = outputBlobs[0][0] 得到回归坐标，
# confidence = outputBlobs[1][0][1]  得到是人脸置信度
[array([[[[ 8.76786932e-03,  9.55079123e-03,  8.35606456e-03,
           2.79193372e-03,  6.37449324e-04,  7.98143446e-04,
           3.49151529e-03, -1.56425871e-03, -2.14402936e-03,
          -6.99994154e-03, -9.76257026e-04, -3.01474705e-04,
           1.70527361e-02,  1.57605242e-02,  1.48732215e-02],
         [ 1.32623129e-03,  6.34383224e-03,  1.06124571e-02,
           8.86401813e-03,  7.76462816e-03,  2.22903304e-03,
          -3.01218405e-03, -5.05260937e-03, -8.17736611e-04,
          -4.48854640e-04,  3.22072953e-03, -2.58858129e-03,
          -4.48854640e-04,  3.22072953e-03, -2.58858129e-03,]]]],
       dtype=float32),
 array([[[[9.95795488e-01, 9.79869843e-01, 9.93828833e-01,
          9.94909346e-01, 9.82185900e-01, 9.88265753e-01,
          9.69525695e-01, 9.78296399e-01, 9.90925670e-01,
          9.97975290e-01, 9.94776726e-01, 9.87867117e-01,
          9.73019779e-01, 9.95782495e-01, 9.98129189e-01]],

        [[6.37987687e-04, 8.52419180e-04, 7.51846412e-04,
          8.31131591e-04, 6.86273037e-04, 6.52176444e-04,
          5.32965409e-04, 4.46803781e-04, 4.76430519e-04,
          6.13360316e-04, 4.53693123e-04, 4.64633864e-04,
          7.88545003e-04, 5.84406429e-04, 4.65819263e-04]]]],
       dtype=float32), 
]
```

需要做的是人脸置信度判断和产生边界框（Bounding Box）。这里的边界框是一个比较坑的地方，考虑到P-Net的输入实在太小，因此在训练的时候很难截取到完全合适的人脸，因此训练边界框的生成时广泛采用了部分样本。因此，P-Net直接输出的边界框并不是传统回归中的边界坐标，而是预测人脸位置相对于输入图片的位置差。所以，需要专门的算法将位置差转换为真实位置。

![](https://pic4.zhimg.com/80/v2-d404ced92b6178f7f988d06ce9a5672b_hd.jpg)

```python
def getPnetBoxes(self, confidence, regression, scaleFactor, threshold):
    y, x = np.where(confidence > threshold)
    scores = confidence[y, x]

    reg_y1, reg_x1, reg_y2, reg_x2 = [
        regression[i, y, x] for i in range(4)]
    reg = np.array([reg_x1, reg_y1, reg_x2, reg_y2])

    x1 = np.round((P_NET_STRIDE * x + 1) / scaleFactor)
    y1 = np.round((P_NET_STRIDE * y + 1) / scaleFactor)
    x2 = np.round(
        (P_NET_STRIDE * x + 1 + P_NET_WINDOW_SIZE - 1) / scaleFactor)
    y2 = np.round(
        (P_NET_STRIDE * y + 1 + P_NET_WINDOW_SIZE - 1) / scaleFactor)

    rect = np.array([x1, y1, x2, y2])
    bbox = np.vstack([rect, scores, reg])  # [9,N]
    return bbox.T  # [N,9]
```

confidence是每个候选框的置信度。由于图像是二维的，所以confidence也是二维的。因此，y和x就是置信度大于阈值的候选框的坐标。那么，bbox中的前两项就是每个候选框在原始图像中的像素坐标（乘上2是因为搜索的步长为2）。因此，bbox的内容分别为：**bbox所在的候选框的像素坐标、候选框的置信度以及候选框自身的坐标位置差。**

上面已经可以得到很多候选框，在这些候选框中很多都是无效的，那么加需要使用**非极大值抑制方法（NMS）**，该算法的主要思想是：**将所有框的得分排序，选中最高分及其对应的框；遍历其余的框，如果和当前最高分框的重叠面积(IOU)大于一定阈值，我们就将框删除；从未处理的框中继续选一个得分最高的**。后续会纤细介绍NMS的。

```python
def non_max_suppression(bboxes, threshold=0.5, mode='union'):
    '''Non max suppression.
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
```

通过上面的NMS算法后，返回输入bboxes中选中的目标,在通过`bbox_to_square(bboxes)`返回正方形的候选框，因为后面的`R-Net`网络需要输入正方形图像。

```python
if finalFaces:
    totalBoxes = np.vstack(finalFaces)
    bboxes, _ = utlis.non_max_suppression(totalBoxes, 0.7)
    bboxes = utlis.bbox_regression(bboxes)

    bboxes = utlis.bbox_to_square(bboxes)
    bboxes = utlis.padding(bboxes, cols, rows)
    if bboxes.shape[0] == 0:
        return
    return bboxes

```

**注意：** 在这个过程中没有使用特征点信息，忽略下图中的Facial landmark。其实我以不知道为什么不使用。由于模型在训练过程中，**对x坐标和y坐标的判定方式和OpenCV相反，因此代码中存在多个转置操作，**本质上是为了适应模型的处理。还需要注意的是在输入图像要**转换为rgb三通道图像，转换为float32，需要进行转置**

```python
    img = cv2.imread(testImgpath)
    img_bk = img.copy()
    if (img.shape[2] == 3):
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape[2])
    elif (img.shape[2] == 4):
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        print(img.shape[2])

    rgbImg = rgbImg.astype(np.float32)
    rgbImg = np.transpose(rgbImg, (1, 0, 2))
```





