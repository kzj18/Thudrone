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
import detector
import time
import numpy as np
import tello_picture

python_file = os.path.dirname(__file__)
data_path = python_file + '/data/dataset/'
save_path = python_file + '/data/trash_yolo/' + time.strftime('%b_%d_%Y_%H_%M_%S/')
'''
txt_path = save_path + 'detect_result.txt'
'''
txt_path = python_file + '/data/detect_results/result.txt'

yolo_command = ''
yolo_command_lock = threading.Lock()

class info_updater():
    def __init__(self):
        rospy.Subscriber("yolo_command", String, self.update_command)
        self.con_thread = threading.Thread(target = rospy.spin)
        self.con_thread.start()

    def update_command(self, data):
        global yolo_command, yolo_command_lock
        yolo_command_lock.acquire() #thread locker
        yolo_command = data.data
        yolo_command_lock.release()

class PictureNode:
    global yolo_command

    def __init__(self):
        self.answerPub_ = rospy.Publisher('yolo_state', String, queue_size=1)

        self.counter = 0
        self.current_image = None
        self.last_command = ''
        self.now_command = ''
        self.yolo_callback = ''
        self.result_list = []

        while not rospy.is_shutdown():
            self.now_command = yolo_command
            if self.counter == 0:
                self.publishCommand('start')
            if not self.last_command == self.now_command:
                command = self.now_command.split('_')[0]
                pic_command = self.now_command.split('_')[1]
                while len(self.result_list) < pic_command:
                    self.result_list.append([-1, -1])
                self.result_list[pic_command - 1][0] = int(command)
                pic_path = data_path + self.now_command + '.png'
                if os.path.exists(pic_path):
                    img = cv2.imread(pic_path)
                    if os.path.exists(pic_path):
                        os.remove(pic_path)
                        img_copy = copy.deepcopy(img)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        cv2.imwrite(save_path + self.now_command + '.png', img_copy)
                    self.last_command = self.now_command
                    self.counter += 1
                    result = detector.detectBall(img, self.counter)
                    if not result == []:
                        answer = max(result, key=lambda x: x[0])
                        self.yolo_callback = str(int(answer[0])) + '_' + str(self.counter)
                        self.result_list[self.counter - 1][1] = int(answer[0])
                    else:
                        self.yolo_callback = '3_' + str(self.counter)
                        self.result_list[self.counter - 1][1] = 3
                    if not os.path.exists(python_file + '/data/detect_results/'):
                        os.makedirs(python_file + '/data/detect_results/')
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
                    np.savetxt(txt_path, self.result_list, fmt='%d')
                    '''
                    with open(txt_path,'a') as f:
                        f.write(self.yolo_callback)
                        f.write('\n')
                        '''
                    rospy.logwarn('command: ' + self.yolo_callback)
                else:
                    rospy.logwarn('path not exists: ' + pic_path)
            else:
                if not self.yolo_callback == '':
                    self.publishCommand(self.yolo_callback)
                rospy.logwarn('require new picture')

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
    infouper = info_updater()
    Pn = PictureNode()