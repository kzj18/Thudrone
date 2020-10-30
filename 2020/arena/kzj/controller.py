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
import navigation
import detector


class ControllerNode:
    class FlightState(Enum):  # 飞行状态
        WAITING = 1
        NAVIGATING = 2
        DETECTING_TARGET = 3
        LANDING = 4

    def __init__(self):
        rospy.init_node('controller_node', anonymous=True)
        rospy.logwarn('Controller node set up.')

        # 无人机在世界坐标系下的位姿
        self.R_wu_ = R.from_quat([0, 0, 0, 1])
        self.t_wu_ = np.zeros([3], dtype=np.float64)

        self.image_ = None
        self.bridge_ = CvBridge()

        self.flight_state_ = self.FlightState.WAITING
        self.navigating_queue_ = deque()  # 存放多段导航信息的队列，队列元素为二元list，list的第一个元素代表导航维度（'x' or 'y' or 'z'），第二个元素代表导航目的地在该维度的坐标
        self.navigating_dimension_ = None  # 'x' or 'y' or 'z'
        self.navigating_destination_ = None
        self.next_state_ = None  # 完成多段导航后将切换的飞行状态
        self.next_state_navigation = None
        self.navigating_position_accuracy = 0.3
        self.navigating_yaw_accuracy = 10

        self.window_x_list_ = [1.75, 4.25, 6.75] # 窗户中心点对应的x值

        self.is_begin_ = False

        self.commandPub_ = rospy.Publisher('/tello/cmd_string', String, queue_size=100)  # 发布tello格式控制信号

        self.poseSub_ = rospy.Subscriber('/tello/states', PoseStamped, self.poseCallback)  # 接收处理含噪无人机位姿信息
        self.imageSub_ = rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.imageCallback)  # 接收摄像头图像
        self.imageSub_ = rospy.Subscriber('/tello/cmd_start', Bool, self.startcommandCallback)  # 接收开始飞行的命令

        rate = rospy.Rate(0.5)
        while not rospy.is_shutdown():
            if self.is_begin_:
                self.decision()
            rate.sleep()
        rospy.logwarn('Controller node shut down.')

    # 按照一定频率进行决策，并发布tello格式控制信号
    def decision(self):
        if self.flight_state_ == self.FlightState.WAITING:  # 起飞并飞至离墙体（y = 3.0m）适当距离的位置
            rospy.logwarn('State: WAITING')
            self.publishCommand('takeoff')
            self.navigating_queue_ = deque([['y', 1.8]])
            self.switchNavigatingState()
            self.next_state_ = self.FlightState.NAVIGATING
            self.next_state_navigation = self.FlightState.DETECTING_TARGET

        elif self.flight_state_ == self.FlightState.NAVIGATING:

            command = navigation.navigation(
                self.R_wu_.as_euler('zyx', degrees=True),
                self.navigating_yaw_accuracy,
                self.navigating_dimension_,
                self.t_wu_,
                self.navigating_destination_,
                self.navigating_position_accuracy
            )

            if command=='OK':  # 当前段导航结束
                if len(self.navigating_queue_) == 0:
                    self.next_state_ = self.next_state_navigation
                else:
                    self.next_state_ = self.FlightState.NAVIGATING
                self.switchNavigatingState()
            else:
                self.publishCommand(command)

        elif self.flight_state_ == self.FlightState.DETECTING_TARGET:
            rospy.logwarn('State: DETECTING_TARGET')
            # 如果无人机飞行高度与标识高度（1.75m）相差太多，则需要进行调整
            if self.t_wu_[2] > 2.0:
                self.publishCommand('down %d' % int(100*(self.t_wu_[2] - 1.75)))
                return
            elif self.t_wu_[2] < 1.5:
                self.publishCommand('up %d' % int(-100*(self.t_wu_[2] - 1.75)))
                return
            # 如果yaw与90度相差超过正负10度，需要进行旋转调整yaw
            (yaw, pitch, roll) = self.R_wu_.as_euler('zyx', degrees=True)
            yaw_diff = yaw - 90 if yaw > -90 else yaw + 270
            if yaw_diff > 10:  # clockwise
                self.publishCommand('cw %d' % (int(yaw_diff) if yaw_diff > 15 else 15))
                return
            elif yaw_diff < -10:  # counterclockwise
                self.publishCommand('ccw %d' % (int(-yaw_diff) if yaw_diff < -15 else 15))
                return

            if detector.detectTarget(self.image_, 'red') == 'red':
                #rospy.loginfo('Target detected.')
                # 根据无人机当前x坐标判断正确的窗口是哪一个
                # 实际上可以结合目标在图像中的位置和相机内外参数得到标记点较准确的坐标，这需要相机成像的相关知识
                # 此处仅仅是做了一个粗糙的估计
                win_dist = [abs(self.t_wu_[0]-win_x) for win_x in self.window_x_list_]
                win_index = win_dist.index(min(win_dist))  # 正确的窗户编号
                # self.navigating_queue_ = deque([['x', self.window_x_list_[win_index]], ['y', 2.4], ['z', 1.0], ['x', self.window_x_list_[win_index]], ['y', 10.0], ['x', 7.0]])  # 通过窗户并导航至终点上方
                self.navigating_position_accuracy = 0.10
                self.navigating_yaw_accuracy = 8
                self.navigating_queue_ = deque([['z', 1.0], ['x', self.window_x_list_[win_index]], ['y', 5.0], ['x', 7.0]])
                self.switchNavigatingState()
                self.next_state_ = self.FlightState.NAVIGATING
                self.next_state_navigation = self.FlightState.LANDING
            else:
                if self.t_wu_[0] > 7.5:
                    rospy.loginfo('Detection failed, ready to land.')
                    self.flight_state_ = self.FlightState.LANDING
                else:  # 向右侧平移一段距离，继续检测
                    self.publishCommand('right 75')

        elif self.flight_state_ == self.FlightState.LANDING:
            rospy.logwarn('State: LANDING')
            self.publishCommand('land')
        else:
            pass

    # 在向目标点导航过程中，更新导航状态和信息
    def switchNavigatingState(self):
        if len(self.navigating_queue_) == 0:
            self.flight_state_ = self.next_state_
        else: # 从队列头部取出无人机下一次导航的状态信息
            next_nav = self.navigating_queue_.popleft()
            # TODO 3: 更新导航信息和飞行状态
            self.navigating_dimension_ = next_nav[0]
            self.navigating_destination_ = next_nav[1]
            self.flight_state_ = self.next_state_ if not self.next_state_ == None else self.flight_state_
            # end of TODO 3

    # 向相关topic发布tello命令
    def publishCommand(self, command_str):
        msg = String()
        msg.data = command_str
        self.commandPub_.publish(msg)

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

