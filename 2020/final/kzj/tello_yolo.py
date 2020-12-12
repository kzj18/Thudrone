#!/usr/bin/python
#-*- encoding: utf8 -*-

import os
import rospy
import threading
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image
import copy
import cv2
import time
import numpy as np
from yolo import YOLO
from PIL import Image

python_file = os.path.dirname(__file__)
data_path = python_file + '/data/dataset/'
save_path = python_file + '/data/trash_yolo/' + time.strftime('%b_%d_%Y_%H_%M_%S/')
'''
txt_path = save_path + 'detect_result.txt'
'''
txt_path = python_file + '/data/detect_results/result.txt'

yolo_command = ''
yolo_command_lock = threading.Lock()

pic_num = 4

def cv22Image(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = np.array(img)
    return Image.fromarray(np.uint8(img))

def confident(results):
    count = [0, 0, 0]
    for result in results:
        if result[1] in [0, 1, 2]:
            count[result[1]] += 1
    if count == [1, 1, 1]:
        return True
    return False

class PictureNode:

    def __init__(self):
        self.answerPub_ = rospy.Publisher('yolo_state', String, queue_size=1)

        self.counter = 0
        self.current_image = None
        self.last_command = ''
        self.yolo_callback = ''
        self.result_list = []

        weight = python_file + '/weights/ball.pth'
        classes = python_file + '/weights/ball.txt'
        yolo = YOLO(weight=weight, classes=classes)

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        data_list = os.listdir(data_path)
        if not data_list == []:
            for image in data_list:
                os.remove(data_path + image)

        while not rospy.is_shutdown():
            data_list = os.listdir(data_path)
            if self.counter == 0:
                self.publishCommand('start')
            if not data_list == []:
                pic_path = data_path + data_list[0]
                img = cv2.imread(pic_path)
                if img is None:
                    print('img is None')
                    continue
                os.remove(pic_path)
                img_copy = copy.deepcopy(img)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path + data_list[0], img_copy)
                self.counter += 1
                if len(self.result_list) < self.counter:
                    self.result_list.append([-1, -1])
                img0 = cv22Image(img)
                result = yolo.detect_image(img0, self.counter)
                if not result == []:
                    answer = max(result, key=lambda x: x[0])
                    self.yolo_callback = str(int(answer[1])) + '_' + str(self.counter)
                    if self.result_list[self.counter - 1] == [-1, -1]:
                        self.result_list[self.counter - 1][0] = int(data_list[0].split('_')[0])
                        self.result_list[self.counter - 1][1] = int(answer[1])
                else:
                    self.yolo_callback = '3_' + str(self.counter)
                    if self.result_list[self.counter - 1] == [-1, -1]:
                        self.result_list[self.counter - 1][0] = int(data_list[0].split('_')[0])
                        self.result_list[self.counter - 1][1] = 3
                if not os.path.exists(python_file + '/data/detect_results/'):
                    os.makedirs(python_file + '/data/detect_results/')
                if len(self.result_list) == pic_num or confident(self.result_list):
                    if not os.path.exists(txt_path):
                        np.savetxt(txt_path, self.result_list, fmt='%d')
                        while True:
                            self.publishCommand('end')
                            time.sleep(0.01)
            else:
                if not self.yolo_callback == '':
                    self.publishCommand(self.yolo_callback)
                #rospy.logwarn('require new picture')

    def publishCommand(self, command_str):
        msg = String()
        msg.data = command_str
        self.answerPub_.publish(msg)
        for _ in range(5):
            time.sleep(0.01)
            self.answerPub_.publish(msg)
        
if __name__ == "__main__":
    rospy.init_node('tello_yolo', anonymous=True)
    rospy.logwarn('Yolo dealer node set up.')
    Pn = PictureNode()