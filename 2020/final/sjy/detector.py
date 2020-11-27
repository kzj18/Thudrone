#!/usr/bin/python
#-*- encoding: utf8 -*-

import os
import cv2
import math
import numpy as np
import time

test_mode = False
#img_file = '//home//kzj18//Pictures//data'
img_file = os.path.abspath('').replace('/', '//') + '//data'
#record_data = '//home//kzj18//Pictures//data//record'
record_data = os.path.abspath('').replace('/', '//') + '//data//record'
COLOR_RANGE = {
    'r': [(0, 100, 72), (10, 255, 255)],
    'y': [(26, 43, 46), (34, 255, 255)],
    'b': [(100, 43, 46), (124, 255, 255)]
}

# 判断是否检测到目标
def detectFire(image, color='r', record_mode = False):
    if image is None:
        return 'No Picture'
    width = image.shape[1]
    image_copy = image.copy()
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGRA2GRAY)
    opened = get_mask(image_copy, COLOR_RANGE, color, 'fire')
    gray_image = cv2.bitwise_and(gray_image, gray_image, mask=opened)
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        1,
        100,
        param1=100,
        param2=10,
        minRadius=30,
        maxRadius=100
        )
    
    if not circles is None:
        circle_r_max = 0
        r_max_circle = None
        for c in circles[0]:
            if c[2] > circle_r_max:
                circle_r_max = c[2]
                r_max_circle = c
        
        if test_mode:
            circle_pic = image_copy.copy()
            cv2.circle(circle_pic, (r_max_circle[0], r_max_circle[1]), r_max_circle[2], (0, 255, 0), 1)
            savepic(img_file, 'circle', circle_pic)
            savepic(img_file, 'gray', gray_image)
        if record_mode:
            circle_pic = image_copy.copy()
            recordpic(record_data, 'original_success', circle_pic)
            cv2.circle(circle_pic, (r_max_circle[0], r_max_circle[1]), r_max_circle[2], (0, 255, 0), 1)
            recordpic(record_data, 'circle_success', circle_pic)
            recordpic(record_data, 'gray_success', gray_image)
        return r_max_circle[0:2]
    elif record_mode:
        circle_pic = image_copy.copy()
        recordpic(record_data, 'original_unsuccess', circle_pic)
        recordpic(record_data, 'gray_unsuccess', gray_image)
    return None

def detectBall(image, record_mode = False):
    if image is None:
        return ['No Picture', 0]
    area = {
        'r': 0,
        'y': 0,
        'b': 0
    }
    contour = {
        'r': None,
        'y': None,
        'b': None
    }
    image_copy = image.copy()
    for color in ['r', 'y', 'b']:
        closed = get_mask(image_copy, COLOR_RANGE, color, 'ball')
        (image_contours, contours, hierarchy) = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找出轮廓

        # 在contours中找出最大轮廓
        contour_area_max = 0
        area_max_contour = None
        for c in contours:  # 遍历所有轮廓
            contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
            if contour_area_temp > contour_area_max:
                contour_area_max = contour_area_temp
                area_max_contour = c
        area[color] = contour_area_max
        contour[color] = area_max_contour

    color = max(area, key=area.get)

    if contour[color] is not None:
        if record_mode:
            contour_pic = image_copy.copy()
            recordpic(record_data, 'original_success', contour_pic)
            cv2.drawContours(contour_pic, [contour[color]], 0, (0, 255, 0))
            recordpic(record_data, 'contour_' + color + '_%d'%area[color], contour_pic)
        if area[color] > 150:
            if test_mode:
                contour_pic = image_copy.copy()
                cv2.drawContours(contour_pic, [contour[color]], 0, (0, 255, 0))
                savepic(img_file, 'contour', contour_pic)
            return [color, area[color]]
    elif record_mode:
        contour_pic = image_copy.copy()
        recordpic(record_data, 'original_unsuccess', contour_pic)

    return ['e', 0]

# 判断是否检测到目标
def detectTarget(img):
    if img is None:
        return False
    image_copy = img.copy()
    height = image_copy.shape[0]
    width = image_copy.shape[1]

    frame = cv2.resize(image_copy, (width, height), interpolation=cv2.INTER_CUBIC)  # 将图片缩放
    frame = cv2.GaussianBlur(frame, (3, 3), 0)  # 高斯模糊
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间
    h, s, v = cv2.split(frame)  # 分离出各个HSV通道
    v = cv2.equalizeHist(v)  # 直方图化
    frame = cv2.merge((h, s, v))  # 合并三个通道

    frame = cv2.inRange(frame, self.red_color_range_[0], self.red_color_range_[1])  # 对原图像和掩模进行位运算
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
            return True
    return False
    
def get_mask(image, color_range, color, task):
    name = task + '_' + color + '_'
    height = image.shape[0]
    width = image.shape[1]

    frame = image.copy()

    frame = cv2.resize(frame, (width, height), cv2.INTER_LINEAR)  # 将图片缩放
    frame = cv2.GaussianBlur(frame, (7, 7), 0)  # 高斯模糊
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间

    frame = cv2.inRange(frame, color_range[color][0], color_range[color][1])  # 对原图像和掩模进行位运算
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)  # 闭运算
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)  # 开运算
    if test_mode:
        savepic(img_file, name + 'original', image)
        savepic(img_file, name + 'frame', frame)
        savepic(img_file, name + 'opened', opened)
        savepic(img_file, name + 'closed', closed)
    return opened

def savepic(folder_name, file_name, pic):
    current = time.strftime('%b_%d_%Y_%H_%M_%S')
    folder_name += '//' + current
    file_name = folder_name + '//' + file_name + '.png'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if not os.path.exists(file_name):
        cv2.imwrite(file_name, pic)
    return

def recordpic(folder_name, file_name, pic):
    current = time.strftime('%b_%d_%Y_%H_%M_%S_')
    file_name = folder_name + '//' + current + file_name + '.png'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if not os.path.exists(file_name):
        cv2.imwrite(file_name, pic)
    return

def info():
    print(img_file)
    print(record_data)

if __name__ == '__main__':
    '''
    test_mode = bool(input('mode:'))
    ball = cv2.imread('//home//kzj18//Pictures//ball_env.jpeg', cv2.IMREAD_UNCHANGED)
    result = detectFire(ball)
    print(result)
    result = detectBall(ball)
    print(result)
    '''
    name = input('index:')
    name = '//home//kzj18//divide_color//data//' + str(name) + '.png'
    fire = cv2.imread(name)
    detectFire(fire)
    cv2.destroyAllWindows()
