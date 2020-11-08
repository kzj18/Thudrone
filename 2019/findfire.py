#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tello_base as tello
import cv2
import sys
import time
from sys import platform
from models import *
import threading
import random
import numpy as np
import math

import rospy
from std_msgs.msg import String, Int16, Bool
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
# if you can not find cv2 in your python, you can try this. usually happen when you use conda.
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

img = None
window_state = 0  # window_state % 4 == 0 mean left, 1 and 3 mean middle, 2 mean right

location = [-1, -1, -1, -1, -1]
losetimes = 0
redindex = 0

target_x = [100]
dx = [20]
target_y = [100]
dy = [20]
target_z = [220]
dz = [20]
target_mpry = [0]
dmpry = [8]

ball_real = False
ball_cx = 0
ball_cy = 0
ball_radius = 0

target_cx = 480
dcx = 80
target_cy = 380
dcy = 80

radius_max = 90
radius_min = 40

IsObject = False
NowObject = -1
wheretarget = -1

tello_state = 'mid:-1;x:100;y:100;z:-170;mpry:1,180,1;pitch:0;roll:0;yaw:-19;'
tello_state_lock = threading.Lock()
img_lock = threading.Lock()

picindex = 0

GROUP_INDEX = 1
takeoff_pub, seenfire_pub, tgt1_pub ,tgt2_pub, tgt3_pub, done_pub = None,None,None,None,None,None
state_fail = 0
state_received = 0
state_receivedtarget1 = 0
state_receivedtarget2 = 0
state_receivedtarget3 = 0

fail_lock = threading.Lock()
received_lock = threading.Lock()
receivedtarget1_lock = threading.Lock()
receivedtarget2_lock = threading.Lock()
receivedtarget3_lock = threading.Lock()

target_id = [-1,-1,-1]
target_received_all = False


def IsFindAllLocation():
    global location
    findindex = 0
    unfindindex = -1
    for i in range(5):
        if location[i] != -1:
            findindex += 1
        else:
            unfindindex = i
    if findindex == 4:
        a = [-1, -1, -1, -1, -1]
        for i in range(5):
            if i != unfindindex:
                a[location[i]] = 1
        for i in range(5):
            if a[i] == -1:
                location[unfindindex] = i

def IsFindTargetLocation(tat):
    global wheretarget
    for i in range(5):
        if location[i] == tat:
            wheretarget = i+1
            return True
    return False


def IsLocationCorrect(ObjectKind):
    for i in range(5):
        if location[i] == ObjectKind:
            return False
    return True

# detect


def load_weight():
    # Initialize this once
    cfg = 'cfg/yolov3_fire.cfg'
    data = 'data/fire.data'
    weights = 'weights/fire_new.pt'
    output = 'data/output'
    img_size = 416
    conf_thres = 0.5
    nms_thres = 0.5
    save_txt = False
    save_images = True
    save_path = 'data/output/result.jpg'

    device = torch_utils.select_device(force_cpu=ONNX_EXPORT)
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    if ONNX_EXPORT:
        # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
        s = (320, 192)
        model = Darknet(cfg, s)
    else:
        model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(
            weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        frame = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, frame, 'weights/export.onnx', verbose=True)
        return

    return model, device


def detect_ball(model, device, frame):
    global picindex
    # Initialized  for every detection
    cfg = 'cfg/yolov3_fire.cfg'
    data = 'data/fire.data'
    weights = 'weights/fire_new.pt'
    output = 'data/output'
    img_size = 416
    conf_thres = 0.5
    nms_thres = 0.5
    save_txt = False
    save_images = True
    save_path = 'data/output/result'+str(picindex)+'.jpg'
    # Set Dataloader
    img0 = frame  # BGR

    # Padded resize
    tmpresultimg = letterbox(img0, new_shape=img_size)
    frame = tmpresultimg[0]

    # Normalize RGB
    frame = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    frame = np.ascontiguousarray(frame, dtype=np.float32)  # uint8 to fp16/fp32
    frame /= 255.0  # 0 - 255 to 0.0 - 1.0

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(classes))]

    # Run inference
    t0 = time.time()

    # Get detections

    frame = torch.from_numpy(frame).unsqueeze(0).to(device)
    pred, _ = model(frame)
    det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]

    if det is not None and len(det) > 0:
        # Rescale boxes from 416 to true image size
        det[:, :4] = scale_coords(
            frame.shape[2:], det[:, :4], img0.shape).round()
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()

        # Draw bounding boxes and labels of detections
        for det_pack in det:

            xyxy = []
            result_obj = []

            for index in range(4):
                xyxy.append(det_pack[index])
            conf = det_pack[4]
            cls_conf = det_pack[5]
            cls = det_pack[6]

            if save_txt:  # Write to file
                with open(save_path + '.txt', 'a') as file:
                    file.write(('%g ' * 6 + '\n') % (xyxy, cls, conf))

            # Add bbox to the image
            label = '%s %.2f' % (classes[int(cls)], conf)
            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)])
            cv2.imshow('result', img0)
            cv2.waitKey(3)
            if save_images:  # Save image with detections
                cv2.imwrite(save_path, img0)

    if save_images:
        if platform == 'darwin':  # macos
            os.system('open ' + output + ' ' + save_path)
    if (det is None):
        return [[-1, -1], [-1, -1]]
    if (det.shape[0] <= 0):
        return [[-1, -1], [-1, -1]]
    else:
        cv2.imwrite('detect_result%d.jpg' % picindex, img0)
        cv2.waitKey(3)
        picindex += 1
        return det


def letterbox(frame, new_shape=416, color=(128, 128, 128), mode='auto'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = frame.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode is 'scaleFill':
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

    if shape[::-1] != new_unpad:  # resize
        # INTER_AREA is better, INTER_LINEAR is faster
        frame = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_AREA)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    frame = cv2.copyMakeBorder(
        frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return (frame, ratiow, ratioh, dw, dh)


def Object_Kind(frame):
    global IsObject, NowObject
    model, device = load_weight()
    result_obj = detect_ball(model, device, frame)
    if result_obj[0][0] == -1:
        IsObject = False
        NowObject = -1
        return
    j = 0
    for i in range(len(result_obj)):
        if result_obj[i][6]==2 or result_obj[i][6]==3:
            if result_obj[i][5].item()<0.8:
                continue
        if result_obj[i][5].item() > result_obj[j][5].item():
            j = i
    if result_obj[i][6]==2 or result_obj[i][6]==3:
        if result_obj[i][5].item()<0.8:
            IsObject = False
            NowObject = -1
            return
    elif result_obj[j][5].item() < 0.5:
        IsObject = False
        NowObject = -1
        return
    IsObject = True
    NowObject = int(result_obj[j][6])
    return


def RedCircle_Center(frame):
    global ball_real, ball_cx, ball_cy, ball_radius,redindex
    lowHue = 0
    lowSat = 80
    lowVal = 50
    highHue = 8
    highSat = 255
    highVal = 220
    cv2.imwrite("red%d.png" % redindex, frame)
    redindex+=1
    frameBGR = cv2.GaussianBlur(frame, (7, 7), 0)

    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)

    colorLow = np.array([lowHue, lowSat, lowVal])
    colorHigh = np.array([highHue, highSat, highVal])

    mask = cv2.inRange(hsv, colorLow, colorHigh)

    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

    framemask = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(framemask, cv2.COLOR_BGR2GRAY)

    circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                               100, param1=100, param2=20, minRadius=10, maxRadius=200)
    if circle1 is None:
        ball_real = False
        ball_cx = 0
        ball_cy = 0
        ball_radius = 0
        return framemask
    circles = circle1[0, :, :]
    circles = np.uint16(np.around(circles))
    if len(circles) == 1:
        ball_cx = circles[0][0]
        ball_cy = circles[0][1]
        ball_radius = circles[0][2]
    else:
        j = 0
        for i in range(len(circles)):
            if circles[i][2] > circles[j][2]:
                j = i
        ball_cx = circles[j][0]
        ball_cy = circles[j][1]
        ball_radius = circles[j][2]
    ball_real = True
    return framemask

# send command to master

def failure_handle(data):
    global fail_lock,state_fail
    if state_fail == 0:
        fail_lock.acquire()
        state_fail = data.data
        print ("state_fail = {state_fail}".format(state_fail=state_fail) )
        fail_lock.release()


def received_handle(data):
    global received_lock,state_received
    if state_received == 0:
        received_lock.acquire()
        state_received = data.data
        print ("state_received = {state_received}".format(state_received=state_received) )
        received_lock.release()


def receivedtarget1_handle(data):
    global receivedtarget1_lock,state_receivedtarget1
    if (state_receivedtarget1 == 0):
        receivedtarget1_lock.acquire()
        state_receivedtarget1 = data.data
        print ("state_receivedtarget1 = {state_receivedtarget1}".format(state_receivedtarget1=state_receivedtarget1) )
        receivedtarget1_lock.release()


def receivedtarget2_handle(data):
    global receivedtarget2_lock,state_receivedtarget2
    if (state_receivedtarget2 == 0):
        receivedtarget2_lock.acquire()
        state_receivedtarget2 = data.data
        print ("state_receivedtarget2 = {state_receivedtarget2}".format(state_receivedtarget2=state_receivedtarget2) )
        receivedtarget2_lock.release()


def receivedtarget3_handle(data):
    global receivedtarget3_lock,state_receivedtarget3
    if state_receivedtarget3 == 0:
        receivedtarget3_lock.acquire()
        state_receivedtarget3 = data.data
        print ("state_receivedtarget3 = {state_receivedtarget3}".format(state_receivedtarget3=state_receivedtarget3) )
        receivedtarget3_lock.release()

def target1_handle (data):
    global target_id,target_received_all
    target_id [0] = data.data
    if min(target_id) >= 0:
        target_received_all = True

def target2_handle (data):
    global target_id,target_received_all
    target_id [1] = data.data
    if min(target_id) >= 0:
        target_received_all = True


def target3_handle (data):
    global target_id,target_received_all
    target_id [2] = data.data
    if min(target_id) >= 0:
        target_received_all = True
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
        print("ready")

    def land(self):
        command = "land"
        self.control_pub.publish(command)

    def stop(self):
        command = "stop"
        self.control_pub.publish(command)

    def go(self,x,y,z,s):
        command = "go "+str(x)+" "+str(y)+" "+str(-z)+" "+str(s)
        self.control_pub.publish(command)

# subscribe tello_state and tello_image


class info_updater():
    def __init__(self):
        rospy.Subscriber("tello_state", String, self.update_state)
        rospy.Subscriber("tello_image", Image, self.update_img)
        self.con_thread = threading.Thread(target=rospy.spin)
        self.con_thread.start()

    def update_state(self, data):
        global tello_state, tello_state_lock
        tello_state_lock.acquire()  # thread locker
        tello_state = data.data
        tello_state_lock.release()
        # print(tello_state)

    def update_img(self, data):
        global img, img_lock
        img_lock.acquire()  # thread locker
        img = CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")
        img_lock.release()
        # print(img)


# put string into dict, easy to find
def parse_state():
    global tello_state, tello_state_lock
    tello_state_lock.acquire()
    statestr = tello_state.split(';')
    # print(statestr)
    dict = {}
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
            dict['mpry'] = [int(mpry[0]), int(mpry[1]), int(mpry[2])]
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
    return dict, statestr


# mini task: take off and fly to the center of the blanket.


class task_handle():
    class taskstages():  # 任务阶段
        finding_location = 0  # find locating blanket
        finding_fire = 1
        finding_window = 2
        finished = 8

    def __init__(self, ctrl):
        self.States_Dict = None  # 初始化无人机状态
        self.ctrl = ctrl  # 初始化无人机控制方式
        self.now_stage = self.taskstages.finding_location  # 初始化无人机任务阶段为：finding_location

    def main(self):  # main function: examine whether tello finish the task
        global window_state
        self.ctrl.takeoff()
        time.sleep(2)
        while not (self.now_stage == self.taskstages.finished):
            if(self.now_stage == self.taskstages.finding_fire):
                self.finding_fire()
            elif(self.now_stage == self.taskstages.finding_window):
                self.finding_window()
            elif(self.now_stage == self.taskstages.finding_location):
                self.finding_location()
        self.ctrl.land()
        time.sleep(2)

    def losemid(self):
        global losetimes
        self.States_Dict,statestr=parse_state()
        while self.States_Dict['mid']==-1 and losetimes<=2:
            self,ctrl.up(50)
            time.sleep(4)
            self.States_Dict,statestr=parse_state()
            losetimes+=1
        return

    def locate(self, truex, targetx, ddx, truey, targety, ddy, truez, targetz, ddz, truempry, targetmpry, ddmpry):
        # if truex == True, we will not do x
        state_conf = 0
        self.States_Dict, statestr = parse_state()
        # target situation

        while not ((self.States_Dict['mpry'][1] + 0 <= ddmpry and self.States_Dict['mpry'][1] + 0 >= -ddmpry and self.States_Dict['mid'] != -1)or (truex and truey and truez)):
            print(statestr)
            print("mpry is not 0")
            if self.States_Dict['mid'] == -1:
                print("mid == -1")
                self.losemid()
            elif (self.States_Dict['mpry'][1] + 0 < -ddmpry):
                self.ctrl.cw(-self.States_Dict['mpry'][1])
                print("rrr")
                time.sleep(4)
            elif(self.States_Dict['mpry'][1] + 0 > ddmpry):
                self.ctrl.ccw(self.States_Dict['mpry'][1])
                print("lll")
                time.sleep(4)
            self.ctrl.stop()
            time.sleep(4)
            self.States_Dict, statestr = parse_state()  # 更新无人机状态
        print("finish mpry == 0")

        state_dx = self.States_Dict['x']-targetx
        state_dy = self.States_Dict['y']-targety
        state_dz = abs(self.States_Dict['z'])-targetz

        while not (((state_dx <= ddx and state_dx >= -ddx)or truex) and ((state_dy <= ddy and state_dy >= -ddy)or truey) and ((state_dz >= -ddz and state_dz <= ddz)or truez)and self.States_Dict['mid'] != -1):
            print(statestr)
            if self.States_Dict['mid'] == -1:
                print("mid == -1")
                self.losemid()
            elif (state_dx > ddx or state_dx < -ddx and not truex):
                if not truex:
                    print("x is not correct")
                    if (state_dx < -ddx):
                        if abs(state_dx) < 150:
                            self.ctrl.forward(abs(state_dx))
                            time.sleep(4)
                        else:
                            self.ctrl.forward(150)
                            time.sleep(4)
                    elif (state_dx > ddx):
                        if abs(state_dx) < 150:
                            self.ctrl.back(state_dx)
                            time.sleep(4)
                        else:
                            self.ctrl.back(150)
                            time.sleep(4)
            elif (state_dy > ddy or state_dy < -ddy and not truey):
                if not truey:
                    print("y is not correct")
                    if (state_dy < -ddy):

                        if abs(state_dy) < 150:
                            self.ctrl.right(abs(state_dy))
                            time.sleep(4)
                        else:
                            self.ctrl.right(150)
                            time.sleep(4)
                    elif (state_dy > ddy):
                        if abs(state_dy) < 150:
                            self.ctrl.left(state_dy)
                            time.sleep(4)
                        else:
                            self.ctrl.left(150)
                            time.sleep(4)
            elif (state_dz > ddz or state_dz < -ddz and not truez):
                if not truez:
                    print("z is not correct")
                    if (state_dz < -ddz):
                        if abs(state_dz) < 150:
                            self.ctrl.up(abs(state_dz))
                            time.sleep(4)
                        else:
                            self.ctrl.up(150)
                            time.sleep(4)
                    elif (state_dz > ddz):
                        if abs(state_dz) < 150:
                            self.ctrl.down(state_dz)
                            time.sleep(4)
                        else:
                            self.ctrl.down(150)
                            time.sleep(4)
            else:
                state_conf += 1  # 记录调整次数
                print("stop")
            self.ctrl.stop()
            time.sleep(2)
            self.States_Dict, statestr = parse_state()  # 更新无人机状态
            state_dx = self.States_Dict['x']-targetx
            state_dy = self.States_Dict['y']-targety
            state_dz = abs(self.States_Dict['z'])-targetz
        print("finish xyz")

        if abs(targetmpry)+ddmpry >= 180:
            if self.States_Dict['mpry'][1]*targetmpry < 0:
                if targetmpry > 0:
                    state_dmpry = 360+self.States_Dict['mpry'][1]-targetmpry
                else:
                    state_dmpry = -360+self.States_Dict['mpry'][1]-targetmpry
            else:
                state_dmpry = self.States_Dict['mpry'][1]-targetmpry
        else:
            state_dmpry = self.States_Dict['mpry'][1]-targetmpry
        while not ((state_dmpry <= ddmpry and state_dmpry >= -ddmpry and self.States_Dict['mid'] != -1)or truempry):
            print(statestr)
            print("mpry is not correct")
            if self.States_Dict['mid'] == -1:
                print("mid == -1")
                self.losemid()
            elif(state_dmpry < -ddmpry):
                if abs(state_dmpry) < 90:
                    self.ctrl.cw(abs(state_dmpry))
                    time.sleep(2)
                else:
                    self.ctrl.cw(90)
                    time.sleep(2)
            elif(state_dmpry > ddmpry):
                if state_dmpry < 90:
                    self.ctrl.ccw(state_dmpry)
                    time.sleep(2)
                else:
                    self.ctrl.ccw(90)
                    time.sleep(2)
            self.ctrl.stop()
            time.sleep(2)
            self.States_Dict, statestr = parse_state()  # 更新无人机状态
            if abs(targetmpry)+ddmpry >= 180:
                if self.States_Dict['mpry'][1]*targetmpry < 0:
                    if targetmpry > 0:
                        state_dmpry = 360 + \
                            self.States_Dict['mpry'][1]-targetmpry
                    else:
                        state_dmpry = -360 + \
                            self.States_Dict['mpry'][1]-targetmpry
                else:
                    state_dmpry = self.States_Dict['mpry'][1]-targetmpry
            else:
                state_dmpry = self.States_Dict['mpry'][1]-targetmpry
        print("finish mpry")

        return state_conf

    def smartlocate(self, truex, targetx, ddx, truey, targety, ddy, truez, targetz, ddz, truempry, targetmpry, ddmpry,s):
        # if truex == True, we will not do x
        state_conf = 0
        self.States_Dict, statestr = parse_state()
        # target situation

        state_dx = self.States_Dict['x']-targetx
        state_dy = self.States_Dict['y']-targety
        state_dz = abs(self.States_Dict['z'])-targetz
        if (abs(state_dx)>100 and abs (state_dy)>100 and abs(state_dz)>100):
            self.locate(truex, targetx, ddx, truey, targety, ddy, truez, targetz, ddz, truempry, targetmpry, ddmpry)
            return

        while not ((self.States_Dict['mpry'][1] + 0 <= ddmpry and self.States_Dict['mpry'][1] + 0 >= -ddmpry and self.States_Dict['mid'] != -1)or (truex and truey and truez)):
            print("mpry is not 0")
            if self.States_Dict['mid'] == -1:
                print("mid == -1")
                self.losemid()
            elif (self.States_Dict['mpry'][1] + 0 < -ddmpry):
                self.ctrl.cw(-self.States_Dict['mpry'][1])
                print("rrr")
                time.sleep(4)
            elif(self.States_Dict['mpry'][1] + 0 > ddmpry):
                self.ctrl.ccw(self.States_Dict['mpry'][1])
                print("lll")
                time.sleep(4)
            self.ctrl.stop()
            time.sleep(4)
            self.States_Dict, statestr = parse_state()  # 更新无人机状态
            print(statestr)            
        print("finish mpry == 0")

        state_dx = self.States_Dict['x']-targetx
        state_dy = self.States_Dict['y']-targety
        state_dz = abs(self.States_Dict['z'])-targetz

        while not (((state_dx <= ddx and state_dx >= -ddx)or truex) and ((state_dy <= ddy and state_dy >= -ddy)or truey) and ((state_dz >= -ddz and state_dz <= ddz)or truez)and self.States_Dict['mid'] != -1):
            if self.States_Dict['mid'] == -1:
                print("mid == -1")
                self.losemid()
            else:
                self.ctrl.go(-state_dx,state_dy,state_dz,s)
                time.sleep(3)
            self.States_Dict, statestr = parse_state()  # 更新无人机状态
            state_dx = self.States_Dict['x']-targetx
            state_dy = self.States_Dict['y']-targety
            state_dz = abs(self.States_Dict['z'])-targetz
            print(statestr)
        print("finish xyz")

        if abs(targetmpry)+ddmpry >= 180:
            if self.States_Dict['mpry'][1]*targetmpry < 0:
                if targetmpry > 0:
                    state_dmpry = 360+self.States_Dict['mpry'][1]-targetmpry
                else:
                    state_dmpry = -360+self.States_Dict['mpry'][1]-targetmpry
            else:
                state_dmpry = self.States_Dict['mpry'][1]-targetmpry
        else:
            state_dmpry = self.States_Dict['mpry'][1]-targetmpry
        while not ((state_dmpry <= ddmpry and state_dmpry >= -ddmpry and self.States_Dict['mid'] != -1)or truempry):
            print(statestr)
            print("mpry is not correct")
            if self.States_Dict['mid'] == -1:
                print("mid == -1")
                self.losemid()
            elif(state_dmpry < -ddmpry):
                if abs(state_dmpry) < 90:
                    self.ctrl.cw(abs(state_dmpry))
                    time.sleep(2)
                else:
                    self.ctrl.cw(90)
                    time.sleep(2)
            elif(state_dmpry > ddmpry):
                if state_dmpry < 90:
                    self.ctrl.ccw(state_dmpry)
                    time.sleep(2)
                else:
                    self.ctrl.ccw(90)
                    time.sleep(2)
            self.ctrl.stop()
            time.sleep(2)
            self.States_Dict, statestr = parse_state()  # 更新无人机状态
            if abs(targetmpry)+ddmpry >= 180:
                if self.States_Dict['mpry'][1]*targetmpry < 0:
                    if targetmpry > 0:
                        state_dmpry = 360 + \
                            self.States_Dict['mpry'][1]-targetmpry
                    else:
                        state_dmpry = -360 + \
                            self.States_Dict['mpry'][1]-targetmpry
                else:
                    state_dmpry = self.States_Dict['mpry'][1]-targetmpry
            else:
                state_dmpry = self.States_Dict['mpry'][1]-targetmpry
        print("finish mpry")

        return state_conf

    def fastlocate(self, truex, targetx, ddx, truey, targety, ddy, truez, targetz, ddz, truempry, targetmpry, ddmpry,s):
        # if truex == True, we will not do x
        state_conf = 0
        self.States_Dict, statestr = parse_state()
        # target situation

        state_dx = self.States_Dict['x']-targetx
        state_dy = self.States_Dict['y']-targety
        state_dz = abs(self.States_Dict['z'])-targetz
        if (abs(state_dx)>100 or abs (state_dy)>100 or abs(state_dz)>100):
            self.locate(truex, targetx, ddx, truey, targety, ddy, truez, targetz, ddz, truempry, targetmpry, ddmpry)
            return

        if abs(targetmpry)+ddmpry >= 180:
            if self.States_Dict['mpry'][1]*targetmpry < 0:
                if targetmpry > 0:
                    state_dmpry = 360+self.States_Dict['mpry'][1]-targetmpry
                else:
                    state_dmpry = -360+self.States_Dict['mpry'][1]-targetmpry
            else:
                state_dmpry = self.States_Dict['mpry'][1]-targetmpry
        else:
            state_dmpry = self.States_Dict['mpry'][1]-targetmpry
        
        while not(((state_dmpry <= ddmpry and state_dmpry >= -ddmpry and self.States_Dict['mid'] != -1)or truempry)and((state_dx <= ddx and state_dx >= -ddx)or truex) and ((state_dy <= ddy and state_dy >= -ddy)or truey) and ((state_dz >= -ddz and state_dz <= ddz)or truez)and self.States_Dict['mid'] != -1):

            while not ((state_dmpry <= ddmpry and state_dmpry >= -ddmpry and self.States_Dict['mid'] != -1)or truempry):
                print(statestr)
                print("mpry is not correct")
                if self.States_Dict['mid'] == -1:
                    print("mid == -1")
                    self.losemid()
                elif(state_dmpry < -ddmpry):
                    if abs(state_dmpry) < 90:
                        self.ctrl.cw(abs(state_dmpry))
                        time.sleep(2)
                    else:
                        self.ctrl.cw(90)
                        time.sleep(2)
                elif(state_dmpry > ddmpry):
                    if state_dmpry < 90:
                        self.ctrl.ccw(state_dmpry)
                        time.sleep(2)
                    else:
                        self.ctrl.ccw(90)
                        time.sleep(2)
                self.ctrl.stop()
                time.sleep(2)
                self.States_Dict, statestr = parse_state()  # 更新无人机状态
                if abs(targetmpry)+ddmpry >= 180:
                    if self.States_Dict['mpry'][1]*targetmpry < 0:
                        if targetmpry > 0:
                            state_dmpry = 360 + \
                                self.States_Dict['mpry'][1]-targetmpry
                        else:
                            state_dmpry = -360 + \
                                self.States_Dict['mpry'][1]-targetmpry
                    else:
                        state_dmpry = self.States_Dict['mpry'][1]-targetmpry
                else:
                    state_dmpry = self.States_Dict['mpry'][1]-targetmpry
            print("finish mpry")

            state_dx = self.States_Dict['x']-targetx
            state_dy = self.States_Dict['y']-targety
            state_dz = abs(self.States_Dict['z'])-targetz
            new_dx = math.sin(math.radians(targetmpry))*state_dy-math.cos(math.radians(targetmpry))*state_dx
            new_dy = math.sin(math.radians(targetmpry))*state_dx+math.cos(math.radians(targetmpry))*state_dy

            while not (((state_dx <= ddx and state_dx >= -ddx)or truex) and ((state_dy <= ddy and state_dy >= -ddy)or truey) and ((state_dz >= -ddz and state_dz <= ddz)or truez)and self.States_Dict['mid'] != -1):
                if self.States_Dict['mid'] == -1:
                    print("mid == -1")
                    self.losemid()
                else:
                    self.ctrl.go(new_dx,new_dy,state_dz,s)
                    time.sleep(3)
                self.States_Dict, statestr = parse_state()  # 更新无人机状态
                state_dx = self.States_Dict['x']-targetx
                state_dy = self.States_Dict['y']-targety
                state_dz = abs(self.States_Dict['z'])-targetz
                new_dx = math.sin(math.radians(targetmpry))*state_dy-math.cos(math.radians(targetmpry))*state_dx
                new_dy = math.sin(math.radians(targetmpry))*state_dx+math.cos(math.radians(targetmpry))*state_dy
                print(statestr)
            print("finish xyz")


        return state_conf

    def finding_location(self):  # find locating blanket (the higher, the easier)
        # 声明，不符合条件就不运行
        assert (self.now_stage == self.taskstages.finding_location)
        global window_state
        self.ctrl.go(30,80,-80,60)
        time.sleep(3)
        window_state = 2
        self.now_stage = self.taskstages.finding_window
        return

    def finding_fire(self):
        assert (self.now_stage == self.taskstages.finding_fire)
        self.ctrl.stop()
        time.sleep(2)
        RedCircle_Center(img)
        self.States_Dict, statestr = parse_state()
        real = ball_real
        cx = ball_cx
        cy = ball_cy
        radius = ball_radius
        errortimes = 0
        print("the circle is %d" % real)
        print(window_state)
        if real == True:
            while not (cx <= target_cx+dcx and cx >= target_cx-dcx and cy <= target_cy+dcy and cy >= target_cy-dcy and ((self.States_Dict['x'] <= 100 and self.States_Dict['x'] >= 60)or self.States_Dict['mid'] == -1)and((radius <= radius_max and radius >= radius_min)or self.States_Dict['mid'] != -1)):
                if (cx > target_cx+dcx or cx < target_cx-dcx):
                    print("cx - target_cx = %d" % (cx - target_cx))
                    if (cx > target_cx+dcx):
                        self.ctrl.right(20)
                        time.sleep(2)
                        print("mr")
                    elif (cx < target_cx-dcx and cx > 0):
                        self.ctrl.left(20)
                        time.sleep(2)
                        print("ml")
                elif (cy > target_cy+dcy or cy < target_cy-dcy):
                    print("cy - target_cy = %d" % (cy - target_cy))
                    if (cy > target_cy+dcy):
                        self.ctrl.down(20)
                        time.sleep(2)
                        print("mu")
                    elif (cy < target_cy-dcy):
                        self.ctrl.up(20)
                        time.sleep(2)
                        print("md")
                if ((self.States_Dict['x'] > 80+20 or self.States_Dict['x'] < 80-20)and (self.States_Dict['mid'] == 1 or self.States_Dict['mid'] == 6)):
                    print("x")
                    print(self.States_Dict['x'])
                    if (self.States_Dict['x'] > 80+20):
                        self.ctrl.back(20)
                        time.sleep(2)
                        print("bk")
                    elif (self.States_Dict['x'] < 80-20):
                        self.ctrl.forward(20)
                        time.sleep(2)
                        print("fw")
                elif ((radius > radius_max or radius < radius_min)and self.States_Dict['mid'] == -1 and radius > 0):
                    print("r")
                    print("radius = %d" % radius)
                    if (radius > radius_max):
                        self.ctrl.back(20)
                        time.sleep(2)
                        print("bk")
                    elif (radius < radius_min):
                        self.ctrl.forward(20)
                        time.sleep(2)
                        print("fw")
                self.ctrl.stop()
                time.sleep(2)
                RedCircle_Center(img)
                cx = ball_cx
                cy = ball_cy
                real = ball_real
                radius = ball_radius
                self.States_Dict, statestr = parse_state()
                if real == False:
                    errortimes += 1
                else:
                    errortimes = 0
                if errortimes >= 5:
                    self.now_stage = self.taskstages.finding_window
                    return
            print("red_done")
            if self.States_Dict['mid'] != -1:
                f1 = 200-int(self.States_Dict['x'])
                print("forward:%d" % f1)
                self.ctrl.forward(f1)
                time.sleep(4)
            self.now_stage = self.taskstages.finished
        else:
            self.now_stage = self.taskstages.finding_window
        return

    def finding_window(self):
        assert (self.now_stage == self.taskstages.finding_window)
        global window_state
        self.ctrl.stop()
        time.sleep(2)
        RedCircle_Center(img)
        self.States_Dict, statestr = parse_state()
        real = ball_real
        while not real == True:
            if window_state % 4 == 3:
                self.ctrl.right(170)
                time.sleep(4)
                self.ctrl.stop()
                time.sleep(2)
                self.smartlocate(False, 70, 20, False, 300, 20, False,
                            target_z[0]-60, dz[0], False, target_mpry[0], dmpry[0],100)
                
            elif window_state % 4 == 1:
                self.ctrl.left(170)
                time.sleep(4)
                self.ctrl.stop()
                time.sleep(2)
                self.smartlocate(False, 70, 20, False, 100, 20,
                            False, target_z[0], dz[0], False, target_mpry[0], dmpry[0],100)
                
            elif window_state % 4 == 2:
                self.smartlocate(False, 70, 20, False, 100, 20,
                            False, target_z[0]-60, dz[0], False, target_mpry[0], dmpry[0],100)
                
            else:
                self.smartlocate(False, 70, 20, False, 300, 20, False,
                            target_z[0], dz[0], False, target_mpry[0], dmpry[0],100)
            window_state += 1
            self.ctrl.stop()
            time.sleep(2)
            RedCircle_Center(img)
            real = ball_real
        self.now_stage = self.taskstages.finding_fire
        return


if __name__ == '__main__':

    rospy.init_node('tello_control', anonymous=True)

    control_pub = rospy.Publisher('command', String, queue_size=1)
    ctrl = control_handler(control_pub)
    infouper = info_updater()
    tasker = task_handle(ctrl)

    groupid = '/group'+str(GROUP_INDEX)
    takeoff_pub = rospy.Publisher(groupid+'/takeoff', Int16, queue_size=3)
    seenfire_pub = rospy.Publisher(groupid+'/seenfire', Int16, queue_size=3)
    tgt1_pub = rospy.Publisher(groupid+'/seentarget1', Int16, queue_size=3)
    tgt2_pub = rospy.Publisher(groupid+'/seentarget2', Int16, queue_size=3)
    tgt3_pub = rospy.Publisher(groupid+'/seentarget3', Int16, queue_size=3)
    done_pub = rospy.Publisher(groupid+'/done', Int16, queue_size=3)

    time.sleep(2)
    tasker.main()
