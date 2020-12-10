#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import threading
import time

import rospy
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
# if you can not find cv2 in your python, you can try this. usually happen when you use conda.
#sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import tello_base as tello

def callback(data, drone):
	command=data.data
	drone.send_command(command)

def subscribe():
	rospy.Subscriber("command", String, callback, drone)
	#rospy.spin()


def control():
	pass
	# you can send command to tello without ROS, for example:(if you use this function, make sure commit "pass" above!!!)
	# drone.send_command("takeoff")
	# drone.send_command("go 0 50 0 10")
	# drone.send_command("land")
	# print drone.send_command("battery?")
        # drone.send_command("mon")
        # print("mon")

if __name__ == '__main__':
	drone = tello.Tello('', 8888)

	state_pub = rospy.Publisher('tello_state',String, queue_size=3)
	img_pub = rospy.Publisher('tello_image', Image, queue_size=5)

    # you can subscribe command directly, or you can just commit this function
	sub_thread = threading.Thread(target = subscribe)
	sub_thread.start()
    # you can control tello without ROS, or you can just commit this function
	con_thread = threading.Thread(target = control)
	con_thread.start()

	try:
		while not rospy.is_shutdown():

			state=drone.read_state()
			if state is None or len(state) == 0:
				continue
			tello_state="".join(state)
			state_pub.publish(tello_state)

			frame = drone.read_frame()
			if frame is None or frame.size == 0:
				continue
			img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			img_copy = img.copy()
			rect = img.copy()
			rect = rect[0:40, 0:rect.shape[1]]
			cv2.rectangle(rect, (0, 0), (img_copy.shape[1], 40), (255, 255, 255), -1)
			cv2.putText(rect, tello_state, (0, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
			img_copy = np.vstack([rect, img_copy])
			cv2.imshow("tello_image", img_copy)
			cv2.waitKey(2)
			try:
				img_msg = CvBridge().cv2_to_imgmsg(img, 'bgr8')
				img_msg.header.frame_id = rospy.get_namespace()
			except CvBridgeError as err:
				rospy.logerr('fgrab: cv bridge failed - %s' % str(err))
				continue
			img_pub.publish(img_msg)
	except rospy.ROSInterruptException:
		pass

