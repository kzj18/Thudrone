#!/usr/bin/python
#-*- encoding: utf8 -*-

import rospy
import time
import os
from enum import Enum
from std_msgs.msg import Bool, String


class JudgeNode:
    class FSMState(Enum):
        IDLE = 0
        TAKEOFF = 1
        WAIT_DOOR = 2
        DETECTING_TARGET = 3
        FINISHED = 4

    def __init__(self):
        rospy.init_node('judge_node', anonymous=True)

        self.time_begin_ = None
        self.time_end_ = None

        self.fsm_state_ = self.FSMState.IDLE

        self.target_groundtruth_ = 'bfvee'
        self.target_result_ = 'uuuuu'

        self.score_ = 0.0

        self.is_changed_ = [False, False, False, False, False]

        self.takeoffPub_ = rospy.Publisher('/takeoff', Bool, queue_size=100)

        self.readySub_ = rospy.Subscriber('/ready', Bool, self.readyCallback)
        self.seenfireSub_ = rospy.Subscriber('/seenfire', Bool, self.seenfireCallback)
        self.targetresultSub_ = rospy.Subscriber('/target_result', String, self.targetresultCallback)
        self.doneSub_ = rospy.Subscriber('/done', Bool, self.doneCallback)

        self.echomessageTimer_ = rospy.Timer(rospy.Duration(0.037), self.echoMessage)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            pass

    def echoMessage(self, event):
        if self.fsm_state_ == self.FSMState.IDLE:
            os.system('clear')
            print('上位机状态: IDLE')
            print('')
            print('')
            print('目标检测结果真值：' + self.target_groundtruth_)
            print('')
            print('等待无人机向/ready话题发送准备完毕信号……')
            print('')
            print('当前分数：' + str(int(self.score_ + self.getTargetScore())))
            pass
        elif self.fsm_state_ == self.FSMState.TAKEOFF:
            os.system('clear')
            print('上位机状态: TAKEOFF')
            print('')
            print('')
            print('目标检测结果真值：' + self.target_groundtruth_)
            print('')
            print('比赛进行中。')
            print('上位机即将向无人机发送起飞命令。')
            print('')
            print('当前分数：' + str(int(self.score_ + self.getTargetScore())))
            self.printTime()
            # 此处直接发布准许起飞命令并转为WAIT_DOOR状态
            self.publishTakeoffMessage()
        elif self.fsm_state_ == self.FSMState.WAIT_DOOR:
            os.system('clear')
            print('上位机状态: WAIT_DOOR')
            print('')
            print('')
            print('目标检测结果真值：' + self.target_groundtruth_)
            print('')
            print('比赛进行中。')
            print('等待无人机穿过着火点……')
            print('')
            print('当前分数：' + str(int(self.score_ + self.getTargetScore())))
            self.printTime()
            pass
        elif self.fsm_state_ == self.FSMState.DETECTING_TARGET:
            os.system('clear')
            print('上位机状态: DETECTING_TARGET')
            print('')
            print('')
            print('目标检测结果真值：' + self.target_groundtruth_)
            print('当前目标检测结果：' + self.target_result_)
            print('')
            print('比赛进行中。')
            print('无人机正在检测目标……')
            print('')
            print('当前分数：' + str(int(self.score_ + self.getTargetScore())))
            self.printTime()
            pass
        elif self.fsm_state_ == self.FSMState.FINISHED:
            os.system('clear')
            print('上位机状态: FINISHED')
            print('')
            print('')
            print('目标检测结果真值：' + self.target_groundtruth_)
            print('当前目标检测结果：' + self.target_result_)
            print('')
            print('比赛结束。')
            print('')
            print('当前分数：' + str(int(self.score_ + self.getTargetScore())))
            self.printTime()
            pass
        pass

    def getTargetScore(self):
        num = 0
        for i in range(5):
            if self.target_groundtruth_[i] == self.target_result_[i]:
                num += 1
        return 0 if num <= 2 else (10.0*(num-2))

    def printTime(self):
        if self.time_begin_ is None:
            return
        if self.time_end_ is None:
            print('当前用时：%.2f秒' % (time.time() - self.time_begin_))
        else:
            print('当前用时：%.2f秒' % (self.time_end_ - self.time_begin_))

    def publishTakeoffMessage(self):
        # 连发10次，间隔0.1秒
        takeoff_msg = Bool()
        takeoff_msg.data = 1
        for i in range(10):
            self.takeoffPub_.publish(takeoff_msg)
        self.fsm_state_ = self.FSMState.WAIT_DOOR
        pass

    def readyCallback(self, msg):
        if msg.data == 0 or self.fsm_state_ != self.FSMState.IDLE:
            return
        self.fsm_state_ = self.FSMState.TAKEOFF
        self.score_ += 30.0
        self.time_begin_ = time.time()
        pass

    def seenfireCallback(self, msg):
        if msg.data == 0 or self.fsm_state_ != self.FSMState.WAIT_DOOR:
            return
        self.fsm_state_ = self.FSMState.DETECTING_TARGET
        self.score_ += 30.0
        pass

    def targetresultCallback(self, msg):
        if len(msg.data) != 2 or self.fsm_state_ != self.FSMState.DETECTING_TARGET:
            return
        num = msg.data[0]
        t_type = msg.data[1]
        if not (num == '1' or num == '2' or num == '3' or num == '4' or num == '5'):
            return
        if not (t_type == 'e' or t_type == 'b' or t_type == 'f' or t_type == 'v'):
            return
        num = eval(num)
        if self.is_changed_[num-1]:
            return
        target_list = list(self.target_result_)
        target_list[num-1] = t_type
        self.target_result_ = ''.join(target_list)
        self.is_changed_[num-1] = True
        pass

    def doneCallback(self, msg):
        if msg.data == 0 or self.fsm_state_ != self.FSMState.DETECTING_TARGET:
            return
        self.fsm_state_ = self.FSMState.FINISHED
        self.score_ += 10.0
        self.time_end_ = time.time()
        pass


if __name__ == '__main__':
    jn = JudgeNode()
