#!/usr/bin/python
#-*- encoding: utf8 -*-

import cv2
import math
import numpy as np

# 判断是否检测到目标
def detectTarget(image, color='red'):
    if color == 'red':
        color_range = [(0, 43, 46), (6, 255, 255)]

    if image is None:
        return False
    image_copy = image.copy()
    height = image_copy.shape[0]
    width = image_copy.shape[1]

    frame = cv2.resize(image_copy, (width, height), interpolation=cv2.INTER_CUBIC)  # 将图片缩放
    frame = cv2.GaussianBlur(frame, (3, 3), 0)  # 高斯模糊
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
    h, s, v = cv2.split(frame)  # 分离出各个HSV通道
    v = cv2.equalizeHist(v)  # 直方图化
    frame = cv2.merge((h, s, v))  # 合并三个通道

    frame = cv2.inRange(frame, color_range[0], color_range[1])  # 对原图像和掩模进行位运算
    opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算
    (image, contours, hierarchy) = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓

    # 在contours中找出最大轮廓
    contour_area_max = 0
    area_max_contour = None
    for c in contours:  # 遍历所有轮廓
        contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            area_max_contour = c

    if area_max_contour is not None:
        if contour_area_max > 50:
            return color
    return 'None'