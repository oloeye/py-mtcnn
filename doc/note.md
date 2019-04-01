## mtcnn

人脸识别的首要任务就是检测人脸，在现在的开源方案中只要`mtcnn`是比较优秀，速度快，准确度高。

mtcnn 是一种Multi-task的人脸检测框架，使用3个cnn级联的方式同时检测人脸和人脸特征点检测同时进行。

![](https://img-blog.csdn.net/20161121135107750?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGlueXpoYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**MTCNN的算法流程(pipeline)：**

- 通过`P-Net`**全卷积神经网络**（FCN），输入图像金字塔图片，然后生成后候选框和边框回归向量(bounding box regression vectors)，在使用。Bounding box regression的方法来校正这些候选窗，使用非极大值抑制（NMS）合并重叠的候选框。
- 在通过`N-Net`完善候选框，把`P-Net`的候选框输入`N-Net`网络，去掉大部分失败框，继续使用Bounding box regression和NMS（非抑制极大值）合并。
- 最后使用`O-Net`输出最终人脸框和特征点位置，

**CNN的结构图如下：**

![](https://img-blog.csdn.net/20161121141606734?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGlueXpoYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



**具体流程：**

- 对需要检测的图片进行`scale`操作，得到一个图像金字塔，这样可以检测到不同大小的脸。
- 把不同大小的图片输入到`P-Net`网络，在`P-Net`中是`12x12x3`输入，由于是全卷积可以输入任意大小的图片，只是输入`12x12x3`后得到`1x1x2`和`1x1x4`、`1x1x10`。但是输入的是输入`m*nx3`后得到`m*nx2`和`m*nx4`。输出的矩阵上每点对应原图像上`12x12`区域。
- 设定一个阈值，来判断哪些目标框是人脸
- 然后使用`nms`算法对目标框进行筛选，并根据`P-Net`输出的值进行`bounding box regression`，并还原到原图像上。
- 将`P-Net`输出的候选框缩放到`24x24`的大小，并进入到`R-Net`中，接下来就和上面一样的筛选。
- 最后使用`O-Net`输出`Facial landmark localization`



**需要注意的是：**

- **face classification**判断是不是人脸使用的是softmax，因此输出是2维的，一个代表是人脸，一个代表不是人脸
- **bounding box regression**回归出的是bounding box左上角和右下角的偏移\(dx1, dy1, dx2, dy2\)，因此是4维的
- **facial landmark localization**回归出的是左眼、右眼、鼻子、左嘴角、右嘴角共5个点的位置，因此是10维的
- 在**训练阶段**，3个网络都会将关键点位置作为监督信号来引导网络的学习， 但在**预测阶段**，P-Net和R-Net仅做人脸检测，不输出关键点位置（因为这时人脸检测都是不准的），关键点位置仅在O-Net中输出。
- **Bounding box**和**关键点**输出均为**归一化后的相对坐标**，Bounding Box是相对待检测区域（R-Net和O-Net是相对输入图像），归一化是相对坐标除以检测区域的宽高，关键点坐标是相对Bounding box的坐标，归一化是相对坐标除以Bounding box的宽高，这里先建立起初步的印象，具体可以参看后面准备训练数据部分和预测部分的代码细节。

