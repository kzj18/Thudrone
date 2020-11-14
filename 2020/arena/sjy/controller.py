#!/usr/bin/python
#-*- encoding: utf8 -*-

# 对windows.world的一个简单控制策略
# 结合tello的控制接口，控制无人机从指定位置起飞，识别模拟火情标记（红色），穿过其下方对应的窗户，并在指定位置降落
# 本策略尽量使无人机的偏航角保持在初始值（90度）左右
# 运行roslaunch uav_sim windows.launch后，再在另一个终端中运行rostopic pub /tello/cmd_start std_msgs/Bool "data: 1"即可开始飞行
# 代码中的decision()函数和switchNavigatingState()函数共有3个空缺之处，需要同学们自行补全（每个空缺之处需要填上不超过3行代码）

from scipy.spatial.transform import Rotation as R
from collections import deque
from enum import Enum
import rospy
import cv2
import numpy as np
import math
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import detector
import guess
from rosgraph_msgs.msg import Clock

def rostime2str(rostime):
    min = str(rostime.secs // 60).zfill(2)
    sec = str(rostime.secs % 60).zfill(2)
    nsec = str(rostime.nsecs // 10000000).zfill(2)
    result = [int(min), int(sec), int(nsec)]
    return result  
class ControllerNode:
    class FlightState(Enum):  # 飞行状态
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

    def __init__(self):
        rospy.init_node('controller_node', anonymous=True)
        rospy.logwarn('Controller node set up.')

        # 无人机在世界坐标系下的位姿
        self.R_wu_ = R.from_quat([0, 0, 0, 1])
        self.t_wu_ = np.zeros([3], dtype=np.float64)

        self.image_ = None
        self.red_color_range_ = [(0, 43, 46), (10, 255, 255)] # 红色的HSV范围
        self.blue_color_range_ = [(100, 43, 46), (124, 255, 255)] # blue的HSV范围
        self.yellow_color_range_ = [(26, 43, 46), (34, 255, 255)] # yellow的HSV范围
        self.bridge_ = CvBridge()

        self.flight_state_ = self.FlightState.WAITING
        self.navigating_queue_ = deque()  # 存放多段导航信息的队列，队列元素为3元list，(x, y, z)
        self.navigating_destination_ = None
        self.next_state_ = None  # 完成多段导航后将切换的飞行状态
        self.next_state_navigation = None
        self.navigating_position_accuracy = 0.4
        
        self.window_x_list_ = [1.75, 4.25, 6.75] # 窗户中心点对应的x值

        self.is_begin_ = False

        self.commandPub_ = rospy.Publisher('/tello/cmd_string', String, queue_size=100)  # 发布tello格式控制信号
        self.resultPub_ = rospy.Publisher('/tello/target_result', String, queue_size=100)  # 发布tello result
        self.simtimeSub_ = rospy.Subscriber('/clock', Clock, self.simtimeCallback)
        self.poseSub_ = rospy.Subscriber('/tello/states', PoseStamped, self.poseCallback)  # 接收处理含噪无人机位姿信息
        self.imageSub_ = rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.imageCallback)  # 接收摄像头图像
        self.imageSub_ = rospy.Subscriber('/tello/cmd_start', Bool, self.startcommandCallback)  # 接收开始飞行的命令

        self.yaw_err = []
        self.yaw_desired = 0
        self.height_desired = 0
        self.navigating_yaw_accuracy = 10
        self.no_fire = 0
        self.BAll_flag = 0
        self.win_index = 0
        self.color_list = [[], [], [], [], []]
        self.detect_times = 0
        self.fast_way = 0
        self.time_begin_=[0, 0, 0]
        
        while not rospy.is_shutdown():
            if self.is_begin_:
               
                if self.time_judge() == True:
                    print(self.color_list)
                    self.result = guess.guess(self.color_list)
                    self.publishResult(self.result)
                self.decision()
            
        rospy.logwarn('Controller node shut down.')

    def time_judge(self):
        [m_n, s_n, ns_n] = self.time_now_
        [m_b, s_b, ns_b] = self.time_begin_
        
        if s_n - s_b < 0:
            m_n -= 1
            s_n += 60
        print(str(m_n - m_b) + ':' + str(s_n - s_b))
        if m_n - m_b >= 7 and s_n - s_b >= 55:
            return True
        return False


    def simtimeCallback(self, msg):
        x = msg.clock
        self.time_now_ = rostime2str(x)
    
    def yaw_PID(self, accuracy = 0):
        '''
        yaw control 
        input 1 to use precise PID
        '''
        if accuracy == 1:
            self.navigating_yaw_accuracy = 10
        else:
            self.navigating_yaw_accuracy = 15
        (yaw, pitch, roll) = self.R_wu_.as_euler('zyx', degrees=True)
        yaw_diff = yaw - self.yaw_desired
        if yaw_diff > 180:
            yaw_diff = 360 - yaw_diff
        elif yaw_diff < -180:
            yaw_diff = -360 - yaw_diff
        
        if yaw_diff > self.navigating_yaw_accuracy:  # clockwise
            rospy.loginfo('yaw diff: %f'%yaw_diff)
            self.publishCommand('cw %d' % (int(0.8*yaw_diff) if yaw_diff > self.navigating_yaw_accuracy else self.navigating_yaw_accuracy))
            return False
        elif yaw_diff < -self.navigating_yaw_accuracy:  # counterclockwise
            rospy.loginfo('yaw diff: %f'%yaw_diff)
            self.publishCommand('ccw %d' % (int(-0.8*yaw_diff) if yaw_diff < -self.navigating_yaw_accuracy else self.navigating_yaw_accuracy))
            return False
        return True
        
    
    def yaw_cal(self):
        '''
        no need to calculate roll
        '''
        (x_now, y_now, z_now) = self.t_wu_
        (x_d, y_d, z_d, phi_d) = self.navigating_destination_
        dx = x_d - x_now
        dy = y_d - y_now
        #dz = z_d - z_now
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
                yaw = phi_d
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
        if self.t_wu_[2] > self.height_desired + 0.25:
            self.publishCommand('down %d' % int(100*(self.t_wu_[2] - self.height_desired)))
            return False
        elif self.t_wu_[2] < self.height_desired - 0.25:
            self.publishCommand('up %d' % int(-100*(self.t_wu_[2] - self.height_desired)))                
            return False
        return True
    
    def PULL_UP(self):
        '''
        PULL_UP to 3.5 m
        '''
        if self.t_wu_[2] < 3.5 - 0.25:
            self.publishCommand('up %d' % int(abs(100*(self.t_wu_[2] - 3.5))))                
            return False
        return True
    
    def micro_control(self, xx, yy, zz, phi):
        '''
        phi can only == 0 90 -90 179 -179
        '''
        if self.yaw_PID == False:
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
            if abs(self.t_wu_[0] - xx) >= 3:
                cmd_var = 300 
            else:
                cmd_var = int(abs(100*(self.t_wu_[0] - xx)))
            if self.t_wu_[0] > xx + self.navigating_position_accuracy:
                self.publishCommand(cmd[0] + str(cmd_var))
                return False
            elif self.t_wu_[0] < xx - self.navigating_position_accuracy:
                self.publishCommand(cmd[1] + str(cmd_var))               
                return False
        if yy != -1:
            if abs(self.t_wu_[1] - yy) >= 3:
                cmd_var = 300 
            else:
                cmd_var = int(abs(100*(self.t_wu_[1] - yy)))
            if self.t_wu_[1] > yy + self.navigating_position_accuracy:
                self.publishCommand(cmd[2] + str(cmd_var))
                return False
            elif self.t_wu_[1] < yy - self.navigating_position_accuracy:
                self.publishCommand(cmd[3] + str(cmd_var))               
                return False
        if zz != -1:
            if abs(self.t_wu_[2] - zz) >= 3:
                cmd_var = 300 
            else:
                cmd_var = int(abs(100*(self.t_wu_[2] - zz)))
            if self.t_wu_[2] > zz + 0.25:
                self.publishCommand(cmd[4] + str(cmd_var))
                return False
            elif self.t_wu_[2] < zz - 0.25:
                self.publishCommand(cmd[5] + str(cmd_var))               
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
        if self.flight_state_ == self.FlightState.WAITING:  # 起飞并飞至离墙体（y = 3.0m）适当距离的位置
            rospy.logwarn('State: WAITING')
            self.time_begin_ = self.time_now_
            self.publishCommand('takeoff')
            rate = rospy.Rate(0.6)
            
            rate.sleep()
            self.navigating_queue_ = deque([])
            self.next_state_ = self.FlightState.DETECTING_TARGET
            self.next_state_navigation = self.FlightState.DETECTING_TARGET
            self.switchNavigatingState()
            
#**************************************************************************************************************************
        elif self.flight_state_ == self.FlightState.NAVIGATING:
            
            (x_now, y_now, z_now) = self.t_wu_
            (x_d, y_d, z_d, phi_d) = self.navigating_destination_
            rospy.logwarn('State: NAVIGATING %f,%f,%f'%(x_d, y_d, z_d))
            rospy.logwarn('position %f,%f,%f'%(x_now, y_now, z_now))
            dz = z_d - z_now
            
            if self.navigating_destination_ == [0,0,0,0]:
                if len(self.navigating_queue_) == 0:
                        self.next_state_ = self.next_state_navigation
                else:
                    self.next_state_ = self.FlightState.NAVIGATING
                self.switchNavigatingState()
                #return
            cmd_index = 0 if dz > 0 else 1
            command = ['up ', 'down ']
            pull_up = z_d < 3.5 -self.navigating_position_accuracy or command[cmd_index] == 'up '
            if abs(dz) > self.navigating_position_accuracy and pull_up:
                dz = abs(dz)
                self.publishCommand(command[cmd_index]+str(int(100*dz)))
            else:
                dh = math.sqrt((x_d - x_now)**2 + (y_d - y_now)**2)
                
                if dh < self.navigating_position_accuracy*math.sqrt(2):  # 当前段导航结束
                    if len(self.navigating_queue_) == 0:
                        self.next_state_ = self.next_state_navigation
                    else:
                        self.next_state_ = self.FlightState.NAVIGATING
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
                                self.publishCommand(self.cmd_dim + str(300))
                            else:
                                self.publishCommand(self.cmd_dim + str(int(cmd_h)))
                            self.height_desired = z_d
                            self.height_PID()
                            return
            
#**************************************************************************************************************************
        elif self.flight_state_ == self.FlightState.DETECTING_TARGET:
            rospy.logwarn('State: DETECTING_TARGET')
            rospy.logwarn('index %d'% self.win_index)
            # 如果无人机飞行高度与标识高度（1.75m）相差太多，则需要进行调整
            fire_height = 1.75
            self.yaw_desired = 90
            #test
            if self.win_index > 2:
                self.win_index = 0
            if self.t_wu_[0] > self.window_x_list_[self.win_index] + 0.3:
                if abs(self.t_wu_[0] - self.window_x_list_[self.win_index]) >= 3.5:
                    self.publishCommand('left 350')
                    return
                self.publishCommand('left %d' % int(100*abs(self.t_wu_[0] - self.window_x_list_[self.win_index])))
                rospy.logwarn('adjust x')
                return
            elif self.t_wu_[0] < self.window_x_list_[self.win_index] - 0.3:
                if abs(self.t_wu_[0] - self.window_x_list_[self.win_index]) >= 3.5:
                    self.publishCommand('left 350')
                    return
                self.publishCommand('right %d' % int(100*abs(self.t_wu_[0] - self.window_x_list_[self.win_index])))
                rospy.logwarn('adjust x')
                return
            elif self.t_wu_[2] > fire_height + 0.25:
                self.publishCommand('down %d' % int(100*(self.t_wu_[2] - fire_height)))
                rospy.logwarn('adjust z')
                return
            elif self.t_wu_[2] < fire_height - 0.25:
                self.publishCommand('up %d' % int(-100*(self.t_wu_[2] - fire_height)))
                rospy.logwarn('adjust z')
                return
            elif self.t_wu_[1] > 1.8:
                self.publishCommand('back %d' % int(80*(self.t_wu_[1] - 1.8)))
                rospy.logwarn('adjust y under-relaxation')
                return
            elif self.t_wu_[1] < 1:
                self.publishCommand('forward %d' % int(-80*(self.t_wu_[1] - 1.8)))
                rospy.logwarn('adjust y under-relaxation')
                return
            elif self.yaw_PID() == False:
                return
            elif self.detectTarget():
                rospy.loginfo('Target detected.')
                self.navigating_queue_ = deque([])
                self.switch_state(self.FlightState.WINDOW)
            else:
                self.win_index += 1
            

#**************************************************************************************************************************
        elif self.flight_state_ == self.FlightState.WINDOW:
            rospy.logwarn('WINDOW' )
            height = 0.8
            alpha = 1.3
            if self.BAll_flag == 0:
                rospy.logwarn('st0 up down' )

                if self.yaw_PID() == False:
                    return
                    
                if self.t_wu_[2] > height + 0.2:
                    self.publishCommand('down %d' % int(alpha*100*(self.t_wu_[2] - height)))
                    rospy.logwarn('down' )
                    return
                elif self.t_wu_[2] < height - 0.2:
                    self.publishCommand('up %d' % int(-alpha*100*(self.t_wu_[2] - height)))
                    rospy.logwarn('up' )
                    return
                self.BAll_flag += 1
            if self.BAll_flag == 1:
                rospy.logwarn('st1 forward smally' )
                if self.t_wu_[1] < 1.8:
                    if self.t_wu_[1] < 1.5:
                        self.publishCommand('forward %d' % int(-alpha*100*(self.t_wu_[1] - 1.8)))
                    else:
                        self.publishCommand('forward %d' % int(30))
                        rospy.logwarn('micro' )
                    rospy.logwarn('forward' )
                    return
                self.BAll_flag += 1
            if self.BAll_flag == 2:
                
                rospy.logwarn('st2 micro control' )
                if self.t_wu_[0] > self.window_x_list_[self.win_index]+0.2:
                    self.publishCommand('left %d' % int(alpha*100*(self.t_wu_[0] - self.window_x_list_[self.win_index])))
                    rospy.logwarn('left' )
                    return
                elif self.t_wu_[0] < self.window_x_list_[self.win_index]-0.2:
                    self.publishCommand('right %d' % int(-alpha*100*(self.t_wu_[0] - self.window_x_list_[self.win_index])))
                    rospy.logwarn('right' )
                    return
                self.yaw_desired = 90
                if self.yaw_PID(1) == True:
                    self.BAll_flag += 1
                
                #if self.t_wu_[1] <= 3.6:
                    #self.publishCommand('forward %d' % int(-alpha*100*(self.t_wu_[1] - 3.6)))
                   # rospy.logwarn('forawrd' )
               # if self.t_wu_[1] > 3.6:
                 #   self.BAll_flag += 1
            if self.BAll_flag == 3:
                rospy.logwarn('st3 rushB' )
                if self.t_wu_[2] > height + 0.2:
                    self.publishCommand('down %d' % int(alpha*100*(self.t_wu_[2] - height)))
                    rospy.logwarn('down' )
                    return
                elif self.t_wu_[2] < height - 0.2:
                    self.publishCommand('up %d' % int(-alpha*100*(self.t_wu_[2] - height)))
                    rospy.logwarn('up' )
                    return
                if self.t_wu_[1] <= 3.6:
                    if abs(100*(self.t_wu_[1] - 3.7)) <= 0.3:
                        self.publishCommand('forward 30')
                    else:
                        self.publishCommand('forward %d' % int(-100*(self.t_wu_[1] - 3.7)))
                    rospy.logwarn('forawrd' )
                if self.t_wu_[1] > 3.6:
                    self.BAll_flag += 1
            if self.BAll_flag == 4:
                rospy.logwarn('st4 PULL_UP' )
                if self.PULL_UP() == True:
                    self.switch_state(self.FlightState.BALL1)

#**************************************************************************************************************************
        elif self.flight_state_ == self.FlightState.BALL1:
            rospy.logwarn('BALL1' )
            if self.BAll_flag == 0:
                rospy.logwarn('st0' )
                self.navigating_queue_ = deque([[6.5, 9.5, 3.5, -90], [6.5, 9.5, 1.72, -90]])
                self.BAll_flag += 1
                self.next_state_ = self.FlightState.NAVIGATING
                self.next_state_navigation = self.FlightState.BALL1
                self.switchNavigatingState()
                self.detect_times = 0

            elif self.BAll_flag == 1:
                rospy.logwarn('st1' )
                self.sampling(1, -90, self.FlightState.BALL3)
                
             
#**************************************************************************************************************************
        elif self.flight_state_ == self.FlightState.BALL3:
            rospy.logwarn('BALL3')
            if self.BAll_flag == 0:
                rospy.logwarn('st0' )
                self.yaw_desired = -179
                if self.yaw_PID() == True:
                    self.BAll_flag += 1
            elif self.BAll_flag == 1:
                rospy.logwarn('st1' )
                if self.micro_control(6.5, 9.5, 1, -179) == True:
                    self.next_state_ = self.FlightState.BALL3
                    self.next_state_navigation = self.FlightState.BALL3
                    self.BAll_flag += 1
                    self.switchNavigatingState()
                    self.detect_times = 0
            elif self.BAll_flag == 2:
                rospy.logwarn('st2' )
                self.sampling(3, -179, self.FlightState.BALL2)
                
#**************************************************************************************************************************
        elif self.flight_state_ == self.FlightState.BALL2:
            rospy.logwarn('BALL2')
            height = 3.5
            if self.BAll_flag == 0:
                rospy.logwarn('st0' )
                if self.t_wu_[2] < height - 0.25:
                    self.publishCommand('up %d' % int(100*(height - self.t_wu_[2])))
                    rospy.logwarn('up' )
                    return
                else:
                    self.BAll_flag += 1
            elif self.BAll_flag == 1:
                rospy.logwarn('st1' )
                self.navigating_queue_ = deque([[1.5, 7.5, 3.5, 0], [1.5, 7.5, 0.72, 0]])
                self.next_state_ = self.FlightState.NAVIGATING
                self.next_state_navigation = self.FlightState.BALL2
                self.BAll_flag += 1
                self.switchNavigatingState()
                self.detect_times = 0

            elif self.BAll_flag == 2:
                rospy.logwarn('st2' )
                self.sampling(2, 0, self.FlightState.BALL4)
                self.result = guess.confident(self.color_list)
                print(self.color_list)
                if self.result != 'unsure':
                    self.fast_way = 1
                    self.switch_state(self.FlightState.LANDING)
                
#**************************************************************************************************************************
        elif self.flight_state_ == self.FlightState.BALL4:
            rospy.logwarn('BALL4')
            if self.BAll_flag == 0:
                rospy.logwarn('st0' )
                if self.PULL_UP() == True:
                    self.BAll_flag += 1
            if self.BAll_flag == 1:     
                rospy.logwarn('st1' )  
                self.navigating_queue_ = deque([[4, 12.5, 3.5, -90], [4, 12.5, 1.72, -90]])
                self.next_state_ = self.FlightState.NAVIGATING
                self.next_state_navigation = self.FlightState.BALL4
                self.BAll_flag += 1
                self.switchNavigatingState()
                self.detect_times = 0
            elif self.BAll_flag == 2:
                rospy.logwarn('st2' )
                
                self.sampling(4, -90, self.FlightState.LANDING)

                self.result = guess.confident(self.color_list)
                print(self.color_list)
                if self.result == 'unsure':

                    self.result = guess.guess(self.color_list)

#**************************************************************************************************************************
        elif self.flight_state_ == self.FlightState.LANDING:
            rospy.logwarn('State: LANDING')
            if self.BAll_flag == 0:
                rospy.logwarn('st0' )
                if self.PULL_UP() == True:
                    self.BAll_flag += 1
            if self.BAll_flag == 1:     
                rospy.logwarn('st1' )  
                self.yaw_desired = -90
                if self.yaw_PID() == True:
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
                    self.navigating_queue_ = deque([[7, 14.5, 3.5, -90]])
                    self.next_state_ = self.FlightState.NAVIGATING
                    self.next_state_navigation = self.FlightState.FAST_LANDING
                    self.switchNavigatingState()
                    return
                if self.micro_control(7, 14.5, -1, -90) ==True:
                    self.publishCommand('land')
#**************************************************************************************************************************
        elif self.flight_state_ == self.FlightState.FAST_LANDING:
            rospy.logwarn('FAST_LANDING' )
            self.publishCommand('land')
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

    # 向相关topic发布tello命令
    def publishCommand(self, command_str):
        msg = String()
        msg.data = command_str
        self.commandPub_.publish(msg)
        rate = rospy.Rate(0.3)
        rate.sleep()
        
    def publishResult(self, result_str):
        msg = String()
        msg.data = result_str
        self.resultPub_.publish(msg)
        self.resultPub_.publish(msg)
        #rate = rospy.Rate(0.3)
        #rate.sleep()


    # 接收无人机位姿
    def poseCallback(self, msg):
        self.t_wu_ = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.R_wu_ = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        pass

    # 接收相机图像
    def imageCallback(self, msg):
        try:
            self.image_ = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as err:
            print(err)

    # 接收开始信号
    def startcommandCallback(self, msg):
        self.is_begin_ = msg.data


if __name__ == '__main__':
    cn = ControllerNode()

