# coding = utf-8

import os
import sys
import numpy as np
import cv2
from urllib import request
import time

# URL到图片mat
def url2imgcv2(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    start_time = time.time()
    # resp = request.urlopen(url, timeout=5)
    resp = request.urlopen(url, timeout=3)
    try:
        # bytearray将数据转换成（返回）一个新的字节数组
        # asarray 复制数据，将结构化数据转换成ndarray
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        # cv2.imdecode()函数将数据解码成Opencv图像格式
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        # return the imagea
        # logger.info("url2imgcv2, time: %fs, img_shape: %d, img_url: %s" % ( time.time()-start_time, image.shape[0], url) )
    except:
        print ('url2imgcv2 拉取图片超时!')
        image = []
    h, w, c = image.shape
    if c==4:
        image = image[:,:,:3]
    return image
