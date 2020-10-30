#!/usr/bin/python
#-*- encoding: utf8 -*-

import rospy

def navigation(angle, yaw_accuracy, dimension, current_position, target_destination, position_accuracy):
    rospy.logwarn('State: NAVIGATING %s %f'%(dimension, target_destination))
    # 如果yaw与90度相差超过正负10度，需要进行旋转调整yaw
    (yaw, pitch, roll) = angle
    yaw_diff = yaw - 90 if yaw > -90 else yaw + 270
    rospy.loginfo('yaw diff: %f'%yaw_diff)
    if yaw_diff > yaw_accuracy:  # clockwise
        return 'cw %d' % (int(yaw_diff) if yaw_diff > 15 else 15)
    elif yaw_diff < -yaw_accuracy:  # counterclockwise
        # TODO 1: 发布相应的tello控制命令
        return 'ccw %d' % (int(-yaw_diff) if yaw_diff < -15 else 15)
        # end of TODO 1

    dim_index = 0 if dimension == 'x' else (1 if dimension == 'y' else 2)
    dist = target_destination - current_position[dim_index]
    rospy.loginfo('dist: %f'%dist)
    if abs(dist) > position_accuracy:
        dir_index = 0 if dist > 0 else 1  # direction index
        # TODO 2: 根据维度（dim_index）和导航方向（dir_index）决定使用哪个命令
        command = [['right', 'left'], ['forward', 'back'], ['up', 'down']]
        command = command[dim_index][dir_index]+' '
        # end of TODO 2
        if abs(dist) > 1.5:
            return command+'100'
        else:
            return command+str(int(abs(100*dist)))
    else:
        return 'OK'