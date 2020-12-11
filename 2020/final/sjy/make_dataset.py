#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import cv2
import math as m

python_file = os.path.dirname(__file__)
trash_list = python_file + '/data/trash_yolo/'
save_path = python_file + '/data/dataset_for_weights/'
now_time = time.strftime('%b_%d_%Y_%H_%M_%S')

if __name__ == "__main__":
    OoM = 5
    counter = 0
    last_max = 0
    last_max_name = ''

    for dir_name in os.listdir(save_path):
        image_list = save_path + dir_name + '/'
        size = len(os.listdir(image_list))
        if size > last_max:
            last_max_name = image_list

    if not os.path.exists(save_path + now_time + '/'):
        os.makedirs(save_path + now_time + '/')

    if not last_max_name == '':
        for img_name in os.listdir(last_max_name):
            counter += 1
            zero_num = OoM - int(m.log10(counter))
            image_name = last_max_name + img_name
            img = cv2.imread(image_name)
            cv2.imwrite(save_path + now_time + '/' + '0'*zero_num + str(counter) + '.png', img)
    
    for dir_name in os.listdir(trash_list):
        image_list = trash_list + dir_name + '/'
        for img_name in os.listdir(image_list):
            counter += 1
            zero_num = OoM - int(m.log10(counter))
            image_name = image_list + img_name
            img = cv2.imread(image_name)
            cv2.imwrite(save_path + now_time + '/' + '0'*zero_num + str(counter) + '.png', img)
            os.remove(image_name)
        os.rmdir(image_list)