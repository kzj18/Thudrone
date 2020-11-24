#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import threading
import random
import numpy as np
import math
from collections import deque

import rospy
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
# if you can not find cv2 in your python, you can try this. usually happen when you use conda.
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
import tello_base as tello
import detector
import guess

y_max_th = 200
y_min_th = 170

img = None
tello_state='mid:-1;x:100;y:100;z:-170;mpry:1,180,1;pitch:0;roll:0;yaw:-19;'
tello_state_lock = threading.Lock()    
img_lock = threading.Lock()    

# send command to tello
class control_handler: 
    def __init__(self, control_pub):
        self.control_pub = control_pub
    
    def forward(self, cm):
        command = "forward "+(str(cm))
        self.control_pub.publish(command)
    
    def back(self, cm):
        command = "back "+(str(cm))
        self.control_pub.publish(command)
    
    def up(self, cm):
        command = "up "+(str(cm))
        self.control_pub.publish(command)
    
    def down(self, cm):
        command = "down "+(str(cm))
        self.control_pub.publish(command)
    
    def right(self, cm):
        command = "right "+(str(cm))
        self.control_pub.publish(command)
    
    def left(self, cm):
        command = "left "+(str(cm))
        self.control_pub.publish(command)

    def cw(self, cm):
        command = "cw "+(str(cm))
        self.control_pub.publish(command)

    def ccw(self, cm):
        command = "ccw "+(str(cm))
        self.control_pub.publish(command)

    def takeoff(self):
        command = "takeoff"
        self.control_pub.publish(command)
        print ("ready")
        
    def mon(self):
        command = "mon"
        self.control_pub.publish(command)
        print ("mon")

    def land(self):
        command = "land"
        self.control_pub.publish(command)

    def stop(self):
        command = "stop"
        self.control_pub.publish(command)

    def command(self, command):
        self.control_pub.publish(command)

#subscribe tello_state and tello_image
class info_updater():   
    def __init__(self):
        rospy.Subscriber("tello_state", String, self.update_state)
        rospy.Subscriber("tello_image", Image, self.update_img)
        self.con_thread = threading.Thread(target = rospy.spin)
        self.con_thread.start()

    def update_state(self,data):
        global tello_state, tello_state_lock
        tello_state_lock.acquire() #thread locker
        tello_state = data.data
        tello_state_lock.release()
        # print(tello_state)

    def update_img(self,data):
        global img, img_lock
        img_lock.acquire()#thread locker
        img = CvBridge().imgmsg_to_cv2(data, desired_encoding = "passthrough")
        img_lock.release()
        # print(img)


# put string into dict, easy to find
def parse_state():
    global tello_state, tello_state_lock
    tello_state_lock.acquire()
    statestr = tello_state.split(';')
    print (statestr)
    dict={}
    for item in statestr:
        if 'mid:' in item:
            mid = int(item.split(':')[-1])
            dict['mid'] = mid
        elif 'x:' in item:
            x = int(item.split(':')[-1])
            dict['x'] = x
        elif 'z:' in item:
            z = int(item.split(':')[-1])
            dict['z'] = z
        elif 'mpry:' in item:
            mpry = item.split(':')[-1]
            mpry = mpry.split(',')
            dict['mpry'] = [int(mpry[0]),int(mpry[1]),int(mpry[2])]
        # y can be recognized as mpry, so put y first
        elif 'y:' in item:
            y = int(item.split(':')[-1])
            dict['y'] = y
        elif 'pitch:' in item:
            pitch = int(item.split(':')[-1])
            dict['pitch'] = pitch
        elif 'roll:' in item:
            roll = int(item.split(':')[-1])
            dict['roll'] = roll
        elif 'yaw:' in item:
            yaw = int(item.split(':')[-1])
            dict['yaw'] = yaw
    tello_state_lock.release()
    return dict

def showimg():
    global img, img_lock
    img_lock.acquire()
    cv2.imshow("tello_image", img)
    cv2.waitKey(2)
    img_lock.release()

# mini task: take off and fly to the center of the blanket.
class task_handle():
    class taskstages():
        WAITING = 1
        NAVIGATING = 2
        DETECTING_TARGET = 3
        BALL1 = 4
        BALL2 = 5
        BALL3 = 6
        BALL4 = 7
        TOLAND = 8
        LANDING = 9
        WINDOW = 10
        FAST_LANDING = 11

    def __init__(self , ctrl):
        self.States_Dict = None
        self.ctrl = ctrl
        self.now_stage = self.taskstages.WAITING
        self.is_begin_ = False
        self.image_ = None

        self.navigating_destination_ = None
        self.navigating_position_accuracy = 0.4

        self.yaw_err = []
        self.yaw_desired = 0
        self.height_desired = 0
        self.navigating_yaw_accuracy = 10

        self.color_list = [[], [], [], [], []]
        self.red_color_range_ = [(0, 43, 46), (10, 255, 255)] # 红色的HSV范围
        self.detect_times = 0

        self.resultPub_ = rospy.Publisher('/tello/target_result', String, queue_size=100)  # 发布tello result
        self.startSub_ = rospy.Subscriber('/tello/cmd_start', Bool, self.startcommandCallback)  # 接收开始飞行的命令

        while not rospy.is_shutdown():
            if self.is_begin_:
                self.decision()
            
        rospy.logwarn('Controller node shut down.')

    def yaw_PID(self, accuracy = 0):
        '''
        yaw control 
        input 1 to use precise PID
        '''
        if accuracy == 1:
            self.navigating_yaw_accuracy = 10
        else:
            self.navigating_yaw_accuracy = 15
        self.States_Dict = parse_state()
        yaw = self.States_Dict['mpry'][1]
        yaw_diff = yaw - self.yaw_desired
        if yaw_diff > 180:
            yaw_diff = 360 - yaw_diff
        elif yaw_diff < -180:
            yaw_diff = -360 - yaw_diff
        
        if yaw_diff > self.navigating_yaw_accuracy:  # clockwise
            rospy.loginfo('yaw diff: %f'%yaw_diff)
            if yaw_diff > self.navigating_yaw_accuracy:
                self.ctrl.cw(int(0.8*yaw_diff))
            else:
                self.ctrl.cw(self.navigating_yaw_accuracy)            
            return False
        elif yaw_diff < -self.navigating_yaw_accuracy:  # counterclockwise
            rospy.loginfo('yaw diff: %f'%yaw_diff)
            if yaw_diff < -self.navigating_yaw_accuracy:
                self.ctrl.ccw(int(-0.8*yaw_diff))
            else:
                self.ctrl.ccw(self.navigating_yaw_accuracy)
            return False
        return True
        
    
    def yaw_cal(self):
        '''
        no need to calculate roll
        '''
        self.States_Dict = parse_state()
        (x_d, y_d, z_d, phi_d) = self.navigating_destination_
        dx = x_d - self.States_Dict['x']
        dy = y_d - self.States_Dict['y']
        #dz = z_d - self.States_Dict['z']
        if abs(dx) < self.navigating_position_accuracy:
            dx = 0
        if abs(dy) < self.navigating_position_accuracy:
            dy = 0
        
        
        if dx == 0:
            if dy > 0:
                yaw = 90
            elif dy < 0:
                yaw = -90
            else:
                yaw = self.States_Dict['mpry'][1]
        elif dx > 0:
            if dy > 0:
                yaw = math.degrees(math.atan(dy/dx))
            elif dy < 0:
                yaw = -math.degrees(math.atan(-dy/dx))
            else:
                yaw = 0
        else:
            if dy > 0:
                yaw = (180 - math.degrees(math.atan(-dy/dx)))
            elif dy < 0:
                yaw = -(180 - math.degrees(math.atan(dy/dx)))
            else:
                yaw = 180
        rospy.logwarn('yaw %f' %yaw)
        rospy.logwarn('phi_d %f' %phi_d)
        dphi = yaw - phi_d
        if dphi > 180:
            dphi = dphi - 360
        elif dphi < -180:
            dphi = dphi + 360
        rospy.logwarn('dphi %f' %dphi)
        if dphi > -45 and dphi <= 45:
            theta = yaw
            self.cmd_dim = 'forward '
        elif dphi > 45 and dphi <= 135:
            theta = yaw - 90
            if theta < -180:
                theta = theta + 360
            self.cmd_dim = 'left '
        elif dphi > -135 and dphi <= -45:
            theta = yaw +90
            if theta > 180:
                theta = theta - 360
            self.cmd_dim = 'right '
        else:
            theta = yaw -180
            if theta < -180:
                theta = theta + 360
            self.cmd_dim = 'back '
        return theta

    def height_PID(self):
        if self.States_Dict['z'] > self.height_desired + 0.25:
            self.ctrl.down(int(100*(self.States_Dict['z'] - self.height_desired)))
            return False
        elif self.States_Dict['z'] < self.height_desired - 0.25:
            self.ctrl.up(int(-100*(self.States_Dict['z'] - self.height_desired)))
            return False
        return True
    
    def PULL_UP(self):
        '''
        PULL_UP to 3.5 m
        '''
        if self.States_Dict['z'] < 3.5 - 0.25:             
            self.ctrl.up(int(abs(100*(self.States_Dict['z'] - 3.5))))
            return False
        return True
    
    def micro_control(self, xx, yy, zz, phi):
        '''
        phi can only == 0 90 -90 179 -179
        '''
        if self.yaw_PID() == False:
            return False
        rospy.logwarn('micro_controlling (0^0)')
        if phi == 90:
            cmd = ['left ', 'right ', 'back ', 'forward ', 'down ', 'up ']
        elif phi == 179 or phi == -179:
            cmd = ['forward ', 'back ', 'left ' , 'right ', 'down ', 'up ']
        elif phi == 0:
            cmd = ['back ', 'forward ', 'right ', 'left ', 'down ', 'up ']
        elif phi == -90:
            cmd = ['right ', 'left ', 'forward ', 'back ', 'down ', 'up ']
        
        
        if xx != -1:
            if abs(self.States_Dict['x'] - xx) >= 3:
                cmd_var = 300 
            else:
                cmd_var = int(abs(100*(self.States_Dict['x'] - xx)))
            if self.States_Dict['x'] > xx + self.navigating_position_accuracy:
                self.ctrl.command(cmd[0] + str(cmd_var))
                return False
            elif self.States_Dict['x'] < xx - self.navigating_position_accuracy:
                self.ctrl.command(cmd[1] + str(cmd_var))               
                return False
        if yy != -1:
            if abs(self.States_Dict['y'] - yy) >= 3:
                cmd_var = 300 
            else:
                cmd_var = int(abs(100*(self.States_Dict['y'] - yy)))
            if self.States_Dict['y'] > yy + self.navigating_position_accuracy:
                self.ctrl.command(cmd[2] + str(cmd_var))
                return False
            elif self.States_Dict['y'] < yy - self.navigating_position_accuracy:
                self.ctrl.command(cmd[3] + str(cmd_var))               
                return False
        if zz != -1:
            if abs(self.States_Dict['z'] - zz) >= 3:
                cmd_var = 300 
            else:
                cmd_var = int(abs(100*(self.States_Dict['z'] - zz)))
            if self.States_Dict['z'] > zz + 0.25:
                self.ctrl.command(cmd[4] + str(cmd_var))
                return False
            elif self.States_Dict['z'] < zz - 0.25:
                self.ctrl.command(cmd[5] + str(cmd_var))               
                return False
        
        return True

    def sampling(self, BALL_num, yaw_d, next_stage):
        '''
        need self.detect_times = 0 when using.
        input BALL_num, yaw_d, next_stage
        '''
        [self.color, area] = detector.detectBall(self.image_, True)
        self.yaw_desired = yaw_d
        if self.yaw_PID() == True:
            if  self.color == 'e' or area < 150:
                self.detect_times +=1
                self.color_list[BALL_num - 1].append([self.color, area])
                if self.detect_times > 4:
                    
                    rospy.logwarn('e or small' )
                    self.switch_state(next_stage)
                    return True
            else:
                self.color_list[BALL_num - 1].append([self.color, area])
                rospy.logwarn('%s'%self.color )
                self.switch_state(next_stage)
                return True
        return False

    def switch_state(self, next_stage):
        '''
        input next_stage
        '''
        self.next_state_ = next_stage
        self.next_state_navigation = next_stage
        self.switchNavigatingState()
        self.BAll_flag = 0                

    # 按照一定频率进行决策，并发布tello格式控制信号
    def decision(self):
#**************************************************************************************************************************
        if self.flight_state_ == self.taskstages.WAITING:  # 起飞并飞至离墙体（y = 3.0m）适当距离的位置
            rospy.logwarn('State: WAITING')
            self.ctrl.takeoff()
            rate = rospy.Rate(0.6)
            
            rate.sleep()
            self.navigating_queue_ = deque([])
            self.next_state_ = self.taskstages.DETECTING_TARGET
            self.next_state_navigation = self.taskstages.DETECTING_TARGET
            self.switchNavigatingState()
            
#**************************************************************************************************************************
        elif self.flight_state_ == self.taskstages.NAVIGATING:
            
            self.States_Dict = parse_state()
            (x_d, y_d, z_d, phi_d) = self.navigating_destination_
            rospy.logwarn('State: NAVIGATING %f,%f,%f'%(x_d, y_d, z_d))
            rospy.logwarn('position %f,%f,%f'%(self.States_Dict['x'], self.States_Dict['y'], self.States_Dict['z']))
            dz = z_d - self.States_Dict['z']
            
            if self.navigating_destination_ == [0,0,0,0]:
                if len(self.navigating_queue_) == 0:
                        self.next_state_ = self.next_state_navigation
                else:
                    self.next_state_ = self.taskstages.NAVIGATING
                self.switchNavigatingState()
                #return
            cmd_index = 0 if dz > 0 else 1
            command = ['up ', 'down ']
            pull_up = z_d < 3.5 -self.navigating_position_accuracy or command[cmd_index] == 'up '
            if abs(dz) > self.navigating_position_accuracy and pull_up:
                dz = abs(dz)
                self.ctrl.command(command[cmd_index]+str(int(100*dz)))
            else:
                dh = math.sqrt((x_d - self.States_Dict['x'])**2 + (y_d - self.States_Dict['y'])**2)
                
                if dh < self.navigating_position_accuracy*math.sqrt(2):  # 当前段导航结束
                    if len(self.navigating_queue_) == 0:
                        self.next_state_ = self.next_state_navigation
                    else:
                        self.next_state_ = self.taskstages.NAVIGATING
                    self.switchNavigatingState()
                else:
                    phi = self.yaw_cal()
                    self.yaw_desired = phi
                    rospy.logwarn('angle %f'%phi)
                    

                    if self.yaw_PID() == True:
                        if dh < self.navigating_position_accuracy*math.sqrt(3) and (self.yaw_desired == 0 or self.yaw_desired == 90 or self.yaw_desired == -90 or self.yaw_desired == 179 or self.yaw_desired == -179):
                            self.micro_control(x_d, y_d, z_d, phi_d)
                        else:
                            rospy.loginfo('dist: %f'%dh)
                            cmd_h = 100*dh 
                            if self.cmd_dim == 'right ' or self.cmd_dim == 'left ':
                                cmd_h = cmd_h
                            rospy.logwarn('%s'%self.cmd_dim)
                            if dh > 3:
                                self.ctrl.command(self.cmd_dim + str(300))
                            else:
                                self.ctrl.command(self.cmd_dim + str(int(cmd_h)))
                            self.height_desired = z_d
                            self.height_PID()
                            return
            
#**************************************************************************************************************************
        elif self.flight_state_ == self.taskstages.DETECTING_TARGET:
            rospy.logwarn('State: DETECTING_TARGET')

#**************************************************************************************************************************
        elif self.flight_state_ == self.taskstages.WINDOW:
            rospy.logwarn('WINDOW' )

#**************************************************************************************************************************
        elif self.flight_state_ == self.taskstages.BALL1:
            rospy.logwarn('BALL1' )
            if self.BAll_flag == 0:
                rospy.logwarn('st0' )
                self.navigating_queue_ = deque([[6.5, 9.5, 3.5, -90], [6.5, 9.5, 1.72, -90]])
                self.BAll_flag += 1
                self.next_state_ = self.taskstages.NAVIGATING
                self.next_state_navigation = self.taskstages.BALL1
                self.switchNavigatingState()
                self.detect_times = 0

            elif self.BAll_flag == 1:
                rospy.logwarn('st1' )
                self.sampling(1, -90, self.taskstages.BALL3)
                
             
#**************************************************************************************************************************
        elif self.flight_state_ == self.taskstages.BALL3:
            rospy.logwarn('BALL3')
            if self.BAll_flag == 0:
                rospy.logwarn('st0' )
                self.yaw_desired = -179
                if self.yaw_PID() == True:
                    self.BAll_flag += 1
            elif self.BAll_flag == 1:
                rospy.logwarn('st1' )
                if self.micro_control(6.5, 9.5, 1, -179) == True:
                    self.next_state_ = self.taskstages.BALL3
                    self.next_state_navigation = self.taskstages.BALL3
                    self.BAll_flag += 1
                    self.switchNavigatingState()
                    self.detect_times = 0
            elif self.BAll_flag == 2:
                rospy.logwarn('st2' )
                self.sampling(3, -179, self.taskstages.BALL2)
                
#**************************************************************************************************************************
        elif self.flight_state_ == self.taskstages.BALL2:
            rospy.logwarn('BALL2')
            height = 3.5
            self.States_Dict = parse_state()
            if self.BAll_flag == 0:
                rospy.logwarn('st0' )
                if self.States_Dict['z'] < height - 0.25:
                    self.ctrl.up(int(100*(height - self.States_Dict['z'])))
                    rospy.logwarn('up' )
                    return
                else:
                    self.BAll_flag += 1
            elif self.BAll_flag == 1:
                rospy.logwarn('st1' )
                self.navigating_queue_ = deque([[1.5, 7.5, 3.5, 0], [1.5, 7.5, 0.72, 0]])
                self.next_state_ = self.taskstages.NAVIGATING
                self.next_state_navigation = self.taskstages.BALL2
                self.BAll_flag += 1
                self.switchNavigatingState()
                self.detect_times = 0

            elif self.BAll_flag == 2:
                rospy.logwarn('st2' )
                self.sampling(2, 0, self.taskstages.BALL4)
                self.result = guess.confident(self.color_list)
                print(self.color_list)
                if self.result != 'unsure':
                    self.fast_way = 1
                    self.switch_state(self.taskstages.LANDING)
                
#**************************************************************************************************************************
        elif self.flight_state_ == self.taskstages.BALL4:
            rospy.logwarn('BALL4')
            if self.BAll_flag == 0:
                rospy.logwarn('st0' )
                if self.PULL_UP() == True:
                    self.BAll_flag += 1
            if self.BAll_flag == 1:     
                rospy.logwarn('st1' )  
                self.navigating_queue_ = deque([[4, 12.5, 3.5, -90], [4, 12.5, 1.72, -90]])
                self.next_state_ = self.taskstages.NAVIGATING
                self.next_state_navigation = self.taskstages.BALL4
                self.BAll_flag += 1
                self.switchNavigatingState()
                self.detect_times = 0
            elif self.BAll_flag == 2:
                rospy.logwarn('st2' )
                
                self.sampling(4, -90, self.taskstages.LANDING)

                self.result = guess.confident(self.color_list)
                print(self.color_list)
                if self.result == 'unsure':

                    self.result = guess.guess(self.color_list)

#**************************************************************************************************************************
        elif self.flight_state_ == self.taskstages.LANDING:
            rospy.logwarn('State: LANDING')
            if self.BAll_flag == 0:
                rospy.logwarn('st0' )
                if self.PULL_UP() == True:
                    self.BAll_flag += 1
            if self.BAll_flag == 1:     
                self.publishResult(self.result)
                self.BAll_flag += 1
                self.detect_times = 0
            elif self.BAll_flag == 2:
                rospy.logwarn('st2' )
                #s = ''
                #for i in range(5):
                 #   s += self.color_list[i][0]
                #rospy.logwarn('%s'%s )
                if self.fast_way == 1:
                    self.next_state_ = self.taskstages.FAST_LANDING
                    self.next_state_navigation = self.taskstages.FAST_LANDING
                    self.switchNavigatingState()
                    return
                if self.micro_control(7, 14.5, -1, -90) ==True:
                    self.ctrl.land()
#**************************************************************************************************************************
        elif self.flight_state_ == self.taskstages.FAST_LANDING:
            if self.micro_control(7, 14.5, -1, 0) ==True:
                    self.ctrl.land()
                    rospy.logwarn('FAST_LANDING' )
            
#**************************************************************************************************************************
        else:
            pass
#**************************************************************************************************************************
#**************************************************************************************************************************
    # 在向目标点导航过程中，更新导航状态和信息
    def switchNavigatingState(self):
        if len(self.navigating_queue_) == 0:
            self.flight_state_ = self.next_state_
        else: # 从队列头部取出无人机下一次导航的状态信息
            next_nav = self.navigating_queue_.popleft()
            # TODO 3: 更新导航信息和飞行状态
            self.navigating_destination_ = next_nav
            self.flight_state_ = self.next_state_ if not self.next_state_ == None else self.flight_state_
            # end of TODO 3

    # 判断是否检测到目标
    def detectTarget(self):
        if self.image_ is None:
            return False
        image_copy = self.image_.copy()
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

    def publishResult(self, result_str):
        msg = String()
        msg.data = result_str
        self.resultPub_.publish(msg)
        self.resultPub_.publish(msg)
        #rate = rospy.Rate(0.3)
        #rate.sleep()

    # 接收开始信号
    def startcommandCallback(self, msg):
        self.is_begin_ = msg.data

if __name__ == '__main__':
    rospy.init_node('tello_control', anonymous=True)

    control_pub = rospy.Publisher('command', String, queue_size=1)
    ctrl = control_handler(control_pub)
    infouper = info_updater()
    tasker = task_handle(ctrl)
    
    time.sleep(2)
    ctrl.mon()
    time.sleep(5)
    while(1):
        if parse_state()['mid'] == -1:
            ctrl.takeoff( )
            print("take off")
            break
    #print("mon")
    time.sleep(4)
    ctrl.up(60)
    print("up 60")
    time.sleep(2)

    tasker.main()

    ctrl.land()

    

