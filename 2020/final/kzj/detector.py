#!/usr/bin/python
#-*- encoding: utf8 -*-

import os
import cv2
import math
import numpy as np
import time
from sys import platform
import copy
import math as m
from yolo import YOLO
from PIL import Image

test_mode = False
#img_file = '//home//kzj18//Pictures//data'
#record_data = '//home//kzj18//Pictures//data//record'
python_file = os.path.dirname(__file__)
img_file = python_file + '/data'
record_data = python_file + '/data/record'
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
    
def cv22Image(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = np.array(img)
    return Image.fromarray(np.uint8(img))

def detectBall(img, counter):
    weight = python_file + '/weights/ball.pth'
    classes = python_file + '/weights/ball.txt'
    yolo = YOLO(weight=weight, classes=classes)
    image = cv22Image(img)
    return yolo.detect_image(image, counter)
    
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


if __name__ == '__main__':
    OoM = 2
    name = input('index:')
    '''
    name = '//home//kzj18//divide_color//data//' + str(name) + '.png'
    fire = cv2.imread(name)
    result = detectFire(fire)
    print(result)
    ball = cv2.imread('//home//kzj18//Pictures//000000.jpg', cv2.IMREAD_UNCHANGED)
    result = detectBall(ball)
    print(result)
    '''

    '''
    counter = int(name)
    zero_num = OoM - int(m.log10(counter))
    ball = cv2.imread(python_file + '/data/dataset_for_weights/' + '0'*zero_num + str(counter) + '.png')
    result = detectBall(ball, counter)
    print(result)
    '''

    counter = int(name)
    zero_num = OoM - int(m.log10(counter))
    ball = cv2.imread(python_file + '/Basketball_pic/' + '0'*zero_num + str(counter) + '.jpg')
    result = detectBall(ball, counter)
    print(result)
