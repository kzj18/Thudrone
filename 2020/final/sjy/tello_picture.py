#!/usr/bin/python
#-*- encoding: utf8 -*-

import os
import rospy
import threading
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
import copy
import numpy as np
import time
import cv2
import guess

python_file = os.path.dirname(__file__)
data_path = python_file + '/data/dataset/'
save_path = python_file + '/data/detect_results/'
txt_path = save_path + time.strftime('%b_%d_%Y_%H_%M_%S') + '.txt'
success_msg = Bool()
success_msg.data = 1

picture_command = ''
yolo_callback = ''
img = None
picture_command_lock = threading.Lock()
yolo_callback_lock = threading.Lock()
img_lock = threading.Lock()

names = ['b', 'f', 'v', 'e']

class info_updater():
    def __init__(self):
        rospy.Subscriber("picture_command", String, self.update_command)
        rospy.Subscriber("tello_image", Image, self.update_img)
        rospy.Subscriber('yolo_state', String, self.update_state)
        self.con_thread = threading.Thread(target = rospy.spin)
        self.con_thread.start()

    def update_command(self,data):
        global picture_command, picture_command_lock
        picture_command_lock.acquire() #thread locker
        picture_command = data.data
        picture_command_lock.release()

    def update_img(self,data):
        global img, img_lock
        img_lock.acquire()#thread locker
        img = CvBridge().imgmsg_to_cv2(data, desired_encoding = "passthrough")
        img_lock.release()

    def update_state(self,data):
        global yolo_callback, yolo_callback_lock
        yolo_callback_lock.acquire() #thread locker
        yolo_callback = data.data
        yolo_callback_lock.release()

class PictureNode:
    global img, picture_command, yolo_callback

    def __init__(self):
        self.commandPub_ = rospy.Publisher('yolo_command', String, queue_size=1)
        self.resultPub_ = rospy.Publisher('/target_result', String, queue_size=100)
        self.donePub_ = rospy.Publisher('/done', Bool, queue_size=100)
        self.image_list = []
        self.result_list = []
        self.last_state = {
            'picture_command': copy.deepcopy(picture_command),
            'yolo_callback': copy.deepcopy(yolo_callback)
        }
        self.current_state = copy.deepcopy(self.last_state)
        self.last_answer = ''
        self.answer = [
            '0n',
            '0n',
            '0n',
            '0n',
            '0n'
        ]
        self.communicate_start = False
        self.end_start = False
        self.counter = 0
        self.send_times = 0
        self.change_times = 0
        self.command = ''
        start = time.time()
        for picture in os.listdir(data_path):
            os.remove(data_path + picture)
        while not rospy.is_shutdown():
            now = time.time()
            self.last_state = copy.deepcopy(self.current_state)
            self.current_state['picture_command'] = copy.deepcopy(picture_command)
            self.current_state['yolo_callback'] = copy.deepcopy(yolo_callback)
            if now - start > 1:
                start = time.time()
                rospy.logwarn('the result is: ' + str(self.result_list))

            if not self.current_state['picture_command'] == self.last_state['picture_command']:
                command = self.current_state['picture_command'].split('_')[0]
                if command in ['1', '2', '3', '4', '5'] and not img is None:
                    img_copy = copy.deepcopy(img)
                    self.counter += 1
                    while len(self.result_list) < self.counter:
                        self.result_list.append([-1, -1])
                    self.result_list[self.counter - 1][0] = int(command)
                    cv2.imwrite(data_path + '%s_%d.png'%(command, self.counter), img_copy)
                    rospy.logwarn('get image ' + self.current_state['picture_command'])
                elif command == 'end':
                    self.end_start = True
                    rospy.logwarn('begin end')

            self.image_list = os.listdir(python_file + '/data/dataset')
            if self.current_state['yolo_callback'] == 'start':
                if not len(self.image_list) == 0:
                    self.last_answer = self.image_list[0].split('_')[0]
                    self.command = self.image_list[0].split('.')[0]
                    self.communicate_start = True
                    rospy.logwarn('send image %d from box %s'%(self.counter, self.last_answer))

            elif (not self.current_state['yolo_callback'] == self.last_state['yolo_callback']) and self.communicate_start:
                self.change_times += 1
                rospy.logwarn('change times: %d, current state: %s, last state: %s'%(self.change_times, self.current_state['yolo_callback'], self.last_state['yolo_callback']))
            elif self.send_times < self.change_times:
                answer = self.current_state['yolo_callback'].split('_')[0]
                yolo_counter = int(self.current_state['yolo_callback'].split('_')[1])
                while len(self.result_list) < yolo_counter:
                    self.result_list.append([-1, -1])
                self.result_list[yolo_counter - 1][1] = int(answer)
                rospy.logwarn('finish image %d from box %s, the answer is %s'%(self.counter, self.last_answer, answer))

                if self.end_start and self.counter == yolo_counter:
                    self.send_result()
                    rospy.logwarn('finish end')

                if not len(self.image_list) == 0:
                    self.last_answer = self.image_list[0].split('_')[0]
                    self.command = self.image_list[0].split('.')[0]
                    self.send_times += 1
                    rospy.logwarn('send image %d from box %s, send times is %d'%(self.counter, self.last_answer, self.send_times))

            if not self.command == '':
                self.publishCommand(self.command)

    def send_result(self):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        rospy.logwarn('the result is: ' + str(self.result_list))
        rospy.logwarn('result was saved to' + txt_path)
        np.savetxt(txt_path, self.result_list, fmt='%d')
        result = guess.guess(self.result_list)
        for index, item in enumerate(result):
            self.publishResult(str(index+1) + names[item])
        time.sleep(0.01)
        self.donePub_.publish(success_msg)
        

    def publishResult(self, result_str):
        msg = String()
        msg.data = result_str
        self.resultPub_.publish(msg)
            
    def publishCommand(self, command_str):
        msg = String()
        msg.data = command_str
        self.commandPub_.publish(msg)
        rate = rospy.Rate(0.3)
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node('tello_picture', anonymous=True)
    rospy.logwarn('Picture saver node set up.')
    infouper = info_updater()
    Pn = PictureNode()
    