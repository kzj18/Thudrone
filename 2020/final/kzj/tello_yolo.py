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

# print(os.path.dirname(__file__))
# output: /home/kzj18/catkin_ws/src/tello_control

img = None
yolo_command = ''
img_lock = threading.Lock()
yolo_command_lock = threading.Lock()

names = ['basketball', 'football', 'volleyball', 'balloon']

class info_updater():
    def __init__(self):
        rospy.Subscriber("tello_picture", Image, self.update_img)
        rospy.Subscriber("yolo_command", String, self.update_command)
        self.con_thread = threading.Thread(target = rospy.spin)
        self.con_thread.start()

    def update_img(self,data):
        global img, img_lock
        img_lock.acquire()#thread locker
        img = CvBridge().imgmsg_to_cv2(data, desired_encoding = "passthrough")
        img_lock.release()

    def update_command(self,data):
        global yolo_command, yolo_command_lock
        yolo_command_lock.acquire() #thread locker
        yolo_command = data.data
        yolo_command_lock.release()

class PictureNode:
    global img, yolo_command

    def __init__(self):
        self.answerPub_ = rospy.Publisher('yolo_state', String, queue_size=1)

        self.counter = 0
        self.current_image = None
        self.last_command = '0'
        self.now_command = ''
        self.yolo_callback = ''

        while not rospy.is_shutdown():
            self.now_command = yolo_command
            if self.counter == 0:
                self.publishCommand('start')
            if (not img is None) and (not self.last_command == self.now_command):
                self.last_command = self.now_command
                self.counter += 1
                result = detector.detectBall(img, self.counter)
                if not result == []:
                    answer = max(result, lambda x: x[0])
                    self.yolo_callback = str(answer[0]) + '_' + str(self.counter)
                else:
                    self.yolo_callback = '4_' + str(self.counter)
                print('command: ' + self.yolo_callback)
            else:
                if not self.yolo_callback == '':
                    self.publishCommand(self.yolo_callback)
                print('require new picture')

    def publishCommand(self, command_str):
        msg = String()
        msg.data = command_str
        self.answerPub_.publish(msg)
        rate = rospy.Rate(0.3)
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node('tello_yolo', anonymous=True)
    rospy.logwarn('Yolo dealer node set up.')
    infouper = info_updater()
    Pn = PictureNode()