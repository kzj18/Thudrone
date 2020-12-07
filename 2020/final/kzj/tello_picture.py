#!/usr/bin/python
#-*- encoding: utf8 -*-

import os
import rospy
import threading
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image
import copy
import numpy as np
import time
import cv2

python_file = os.path.dirname(__file__)
save_path = python_file + '/data/detect_results/'

picture_command = ''
yolo_callback = ''
img = None
picture_command_lock = threading.Lock()
yolo_callback_lock = threading.Lock()
img_lock = threading.Lock()

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
        self.img_pub = rospy.Publisher('tello_picture', Image, queue_size=5)
        self.commandPub_ = rospy.Publisher('yolo_command', String, queue_size=1)
        self.image_list = [[], [], [], [], []]
        self.result_list = []
        self.last_state = {
            'picture_command': copy.deepcopy(picture_command),
            'yolo_callback': copy.deepcopy(yolo_callback)
        }
        self.current_state = copy.deepcopy(self.last_state)
        self.last_answer = None
        self.answer = [
            '0n',
            '0n',
            '0n',
            '0n',
            '0n'
        ]
        self.communicate_start = False
        self.counter = 0
        self.command = ''
        start = time.time()
        while not rospy.is_shutdown():
            now = time.time()
            self.last_state = copy.deepcopy(self.current_state)
            self.current_state['picture_command'] = copy.deepcopy(picture_command)
            self.current_state['yolo_callback'] = copy.deepcopy(yolo_callback)
            if now - start > 1:
                start = time.time()
                print(self.current_state)

            if not self.current_state['picture_command'] == self.last_state['picture_command']:
                command = self.current_state['picture_command'].split('_')[0]
                if command in ['1', '2', '3', '4', '5']:
                    img_copy = copy.deepcopy(img)
                    self.image_list[int(command)-1].append(img_copy)
                    rospy.logwarn('get image ' + self.current_state['picture_command'])
                elif command is 'end':
                    self.send_result()

            if self.current_state['yolo_callback'] == 'start':
                for index, box in enumerate(self.image_list):
                    if not len(box) == 0:
                        self.counter += 1
                        self.command = str(self.counter)
                        time.sleep(0.1)
                        self.send_image(box[0])
                        self.last_answer = index
                        self.image_list[index].pop(0)
                        while len(self.result_list) < self.counter:
                            self.result_list.append([-1, -1])
                        self.result_list[self.counter - 1][0] = int(index)
                        self.communicate_start = True
                        rospy.logwarn('send image %d from box %d'%(self.counter, index))
            elif (not self.current_state['yolo_callback'] == self.last_state['yolo_callback']) and self.communicate_start:
                answer = self.current_state['yolo_callback'].split('_')[0]
                yolo_counter = self.current_state['yolo_callback'].split('_')[1]
                while len(self.result_list) < yolo_counter:
                    self.result_list.append([-1, -1])
                self.result_list[yolo_counter - 1][1] = int(answer)
                rospy.logwarn('finish image %d from box %d, the answer is %s'%(self.counter, self.last_answer, answer))
                for index, box in enumerate(self.image_list):
                    if not len(box) == 0:
                        self.counter += 1
                        self.command = str(self.counter)
                        time.sleep(0.1)
                        self.send_image(box[0])
                        self.last_answer = index
                        self.image_list[index].pop(0)
                        while len(self.result_list) < self.counter:
                            self.result_list.append([-1, -1])
                        self.result_list[self.counter - 1][0] = int(index)
                        rospy.logwarn('send image %d from box %d'%(self.counter, index))
                else:
                    print('callback error')

            if not self.command == '':
                self.publishCommand(self.command)

    def send_result(self):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savetxt(save_path + time.strftime('%b_%d_%Y_%H_%M_%S') + '.txt', self.result_list, fmt='%d')

    def send_image(self, image):
        if not image is None:
            try:
                img_msg = CvBridge().cv2_to_imgmsg(image, 'bgr8')
                img_msg.header.frame_id = rospy.get_namespace()
            except CvBridgeError as err:
                rospy.logerr('fgrab: cv bridge failed - %s' % str(err))
                return
            self.img_pub.publish(img_msg)
            
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
    