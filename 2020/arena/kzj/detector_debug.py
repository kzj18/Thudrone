#!/usr/bin/python
#-*- encoding: utf8 -*-

import os
import cv2
import math
import numpy as np
import time

test_mode = True
img_file = '//home//kzj18//Pictures//data'

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

    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGRA2GRAY)
    gray_image = cv2.bitwise_and(gray_image, gray_image, mask=closed)
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        1,
        100,
        param1=100,
        param2=10,
        minRadius=1,
        maxRadius=100
        )
    
    if test_mode:
        savepic(img_file, 'original', image_copy)
        savepic(img_file, 'frame', frame)
        savepic(img_file, 'opened', opened)
        savepic(img_file, 'closed', closed)
        savepic(img_file, 'gray', gray_image)

    # 在contours中找出最大轮廓
    contour_area_max = 0
    area_max_contour = None
    for c in contours:  # 遍历所有轮廓
        contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            area_max_contour = c

    if not circles is None:
        circle_pic = image_copy.copy()
        '''
        circle_r_max = 0
        r_max_circle = None
        for c in circles[0]:
            if c[2] > circle_r_max:
                circle_r_max = c[2]
                r_max_circle = c
        cv2.circle(circle_pic, (r_max_circle[0], r_max_circle[1]), r_max_circle[2], (0, 0, 255), -1)
        '''
        for c in circles[0]:
            cv2.circle(circle_pic, (c[0], c[1]), c[2], (0, 255, 0), 1)
        if test_mode:
            savepic(img_file, 'circle', circle_pic)
    else:
        print('None')

    if area_max_contour is not None:
        if contour_area_max > 50:
            if test_mode:
                countour_pic = image_copy.copy()
                cv2.drawContours(countour_pic, [area_max_contour], 0, (0, 255, 0))
                savepic(img_file, 'contour', countour_pic)
            return color
    return 'None'

def savepic(folder_name, file_name, pic):
    current = time.strftime('%b_%d_%Y_%H_%M_%S')
    folder_name += '//' + current
    file_name = folder_name + '//' + file_name + '.png'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    if not os.path.exists(file_name):
        cv2.imwrite(file_name, pic)
    return

if __name__ == '__main__':
    img = cv2.imread('//home//kzj18//Pictures//ball_env.jpeg', cv2.IMREAD_UNCHANGED)
    #img = cv2.imread('//home//kzj18//Pictures//three_balls.jpg', cv2.IMREAD_UNCHANGED)
    result = detectTarget(img)
    print(result)
    