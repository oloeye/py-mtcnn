# -*- coding: utf-8 -*-
# @Author: Hobo
# @Date:   2018-11-26 09:48:49
# @Last Modified by:   Hobo
# @Last Modified time: 2018-11-26 17:08:21
import time
from oNet import *
from pNet import *
from rNet import *

np.set_printoptions(threshold=np.inf)

if __name__ == '__main__':
    presentPath = os.path.abspath(os.path.dirname(__file__))

    testImgpath = presentPath + "/data/" + "2007_007763.jpg"

    img = cv2.imread(testImgpath)

    # 满足Caffe的需求. w,h,c -> h,w,c
    if (img.shape[2] == 3):
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape[2])
    elif (img.shape[2] == 4):
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        print(img.shape[2])
    rgbImg = rgbImg.astype(np.float32)
    rgbImg = np.transpose(rgbImg, (1, 0, 2))

    THRESHOLD = [0.6, 0.7, 0.7]
    start = time.time()
    # -----------First stage.---------------------------------
    p_modelFile = presentPath + "/data/models/" + "det1.caffemodel"
    p_configFile = presentPath + "/data/models/" + "det1.prototxt"

    p_net = Pnet(p_configFile, p_modelFile, THRESHOLD[0])
    bboxes = p_net.run(rgbImg, 24, 0.709)
    # p_img_bk = img.copy()
    # utlis.draw_and_show(p_img_bk, bboxes)

    # -----------Second stage.---------------------------------
    r_modelFile = presentPath + "/data/models/" + "det2.caffemodel"
    r_configFile = presentPath + "/data/models/" + "det2.prototxt"


    # 宽和高相同，不需要进行转换了
    r_net = Rnet(r_configFile, r_modelFile, THRESHOLD[1])
    bboxes = r_net.run(rgbImg, bboxes)
    # r_img_bk = img.copy()
    # utlis.draw_and_show(r_img_bk, bboxes)

    # -----------Third  stage.---------------------------------
    o_modelFile = presentPath + "/data/models/" + "det3.caffemodel"
    o_configFile = presentPath + "/data/models/" + "det3.prototxt"


    # 宽和高相同，不需要进行转换了
    o_net = Onet(o_configFile, o_modelFile, THRESHOLD[2])
    bboxes, landMark = o_net.run(rgbImg, bboxes)
    print(time.time()-start)
    o_img_bk = img.copy()
    utlis.draw_and_show(o_img_bk, bboxes, points=landMark)
