#!/usr/bin/python
#-*- encoding: utf8 -*-

import os
import cv2
import math
import numpy as np
import time
from sys import platform
from models import *
import copy

test_mode = False
#img_file = '//home//kzj18//Pictures//data'
#record_data = '//home//kzj18//Pictures//data//record'
python_file = os.path.dirname(__file__)
img_file = python_file + '/data'
record_data = python_file + '/data/record'
COLOR_RANGE = {
    'r': [(0, 100, 72), (10, 255, 255)],
    'y': [(26, 43, 46), (34, 255, 255)],
    'b': [(100, 43, 46), (124, 255, 255)]
}

# 判断是否检测到目标
def detectFire(image, color='r', record_mode = False):
    if image is None:
        return 'No Picture'
    width = image.shape[1]
    image_copy = image.copy()
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGRA2GRAY)
    opened = get_mask(image_copy, COLOR_RANGE, color, 'fire')
    gray_image = cv2.bitwise_and(gray_image, gray_image, mask=opened)
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        1,
        100,
        param1=100,
        param2=10,
        minRadius=30,
        maxRadius=100
        )
    
    if not circles is None:
        circle_r_max = 0
        r_max_circle = None
        for c in circles[0]:
            if c[2] > circle_r_max:
                circle_r_max = c[2]
                r_max_circle = c
        
        if test_mode:
            circle_pic = image_copy.copy()
            cv2.circle(circle_pic, (r_max_circle[0], r_max_circle[1]), r_max_circle[2], (0, 255, 0), 1)
            savepic(img_file, 'circle', circle_pic)
            savepic(img_file, 'gray', gray_image)
        if record_mode:
            circle_pic = image_copy.copy()
            recordpic(record_data, 'original_success', circle_pic)
            cv2.circle(circle_pic, (r_max_circle[0], r_max_circle[1]), r_max_circle[2], (0, 255, 0), 1)
            recordpic(record_data, 'circle_success', circle_pic)
            recordpic(record_data, 'gray_success', gray_image)
        return r_max_circle[0:2]
    elif record_mode:
        circle_pic = image_copy.copy()
        recordpic(record_data, 'original_unsuccess', circle_pic)
        recordpic(record_data, 'gray_unsuccess', gray_image)
    return None

def load_weight():
    # Initialize this once
    cfg = python_file + '/cfg/yolov3.cfg'
    data = python_file + '/data/ball.data'
    weights = python_file + '/weights/ball.pt'
    img_size = 416
    conf_thres = 0.5
    nms_thres = 0.5
    save_txt = False
    save_images = True
    save_path = python_file + '/data/output/picture'
    
    device = torch_utils.select_device(force_cpu=ONNX_EXPORT)
    torch.backends.cudnn.benchmark = False  # set False for reproducible results

    # Initialize model
    if ONNX_EXPORT:
        s = (320, 192)  # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
        model = Darknet(cfg, s)
    else:
        model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return
    
    return model, device

def detectBall(img, counter):
    # Initialized  for every detection
    model, device = load_weight()
    cfg = python_file + '/cfg/yolov3.cfg'
    data = python_file + '/data/ball.data'
    weights = python_file + '/weights/ball.pt'
    img_size = 416
    conf_thres = 0.5
    nms_thres = 0.5
    save_txt = False
    save_images = True
    save_path = python_file + '/data/output/picture'
    answer_list = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Set Dataloader
    img0 = img  # BGR
    img_copy = copy.deepcopy(img)

    # Padded resize
    tmpresultimg = letterbox(img0, new_shape=img_size)
    img = tmpresultimg[0]

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp16/fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    t0 = time.time()
    
    # Get detections
    
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    #print("img.shape")
    #print(img.shape )
    pred, _ = model(img)
    det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]
        
    if det is not None and len(det) > 0:
        # Rescale boxes from 416 to true image size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
        # Print results to screen
        #print("image_size")
        #print('%gx%g ' % img.shape[2:])  # print image size
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()
            print("result")
            print('%g %ss' % (n, classes[int(c)]))
            answer_list.append([float(n), int(c)])  #classes definement come from .name file
        
        # Draw bounding boxes and labels of detections
        for det_pack in det:
            xyxy = []
            result_obj=[]
            for index in range(4):
                xyxy.append(det_pack[index])
            conf = det_pack[4]
            cls_conf= det_pack[5]
            cls = det_pack[6]
            #print((xyxy,conf, cls_conf, cls ))
            if save_txt:  # Write to file
                with open(save_path + '/picture_record.txt', 'a') as file:
                    file.write(('%g ' * 6 + '\n') % (xyxy, cls, conf))

            # Add bbox to the image
            label = '%s %.2f' % (classes[int(cls)], conf)
            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)])
            cv2.imshow('result',img0)
            cv2.waitKey(3)
            cv2.destroyWindow('result')
            if save_images:  # Save image with detections
                cv2.imwrite(save_path + '/yolo_counter_' + str(counter) + time.strftime('_%b_%d_%Y_%H_%M_%S_result.png'), img0)
                cv2.imwrite(save_path + '/yolo_counter_' + str(counter) + time.strftime('_%b_%d_%Y_%H_%M_%S_original.png'), img_copy)
    
    print('Done. (%.3fs)' % (time.time() - t0))
    if (det is None):
        return []
    if (det.shape[0] <= 0):
        return []
    else:
        cv2.imwrite(save_path + '/yolo_counter_' + str(counter) + time.strftime('_%b_%d_%Y_%H_%M_%S_result.png'), img0)
        cv2.imwrite(save_path + '/yolo_counter_' + str(counter) + time.strftime('_%b_%d_%Y_%H_%M_%S_original.png'), img_copy)
        cv2.waitKey(3)
        return answer_list

def letterbox(img, new_shape=416, color=(128, 128, 128), mode='auto'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]

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
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return (img, ratiow, ratioh, dw, dh)
    
def get_mask(image, color_range, color, task):
    name = task + '_' + color + '_'
    height = image.shape[0]
    width = image.shape[1]

    frame = image.copy()

    frame = cv2.resize(frame, (width, height), cv2.INTER_LINEAR)  # 将图片缩放
    frame = cv2.GaussianBlur(frame, (7, 7), 0)  # 高斯模糊
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 将图片转换到HSV空间

    frame = cv2.inRange(frame, color_range[color][0], color_range[color][1])  # 对原图像和掩模进行位运算
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)  # 闭运算
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)  # 开运算
    if test_mode:
        savepic(img_file, name + 'original', image)
        savepic(img_file, name + 'frame', frame)
        savepic(img_file, name + 'opened', opened)
        savepic(img_file, name + 'closed', closed)
    return opened

def savepic(folder_name, file_name, pic):
    current = time.strftime('%b_%d_%Y_%H_%M_%S')
    folder_name += '//' + current
    file_name = folder_name + '//' + file_name + '.png'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if not os.path.exists(file_name):
        cv2.imwrite(file_name, pic)
    return

def recordpic(folder_name, file_name, pic):
    current = time.strftime('%b_%d_%Y_%H_%M_%S_')
    file_name = folder_name + '//' + current + file_name + '.png'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if not os.path.exists(file_name):
        cv2.imwrite(file_name, pic)
    return

def info():
    print(img_file)
    print(record_data)

if __name__ == '__main__':
    name = input('index:')
    name = '//home//kzj18//divide_color//data//' + str(name) + '.png'
    fire = cv2.imread(name)
    result = detectFire(fire)
    print(result)
    ball = cv2.imread('//home//kzj18//Pictures//000000.jpg', cv2.IMREAD_UNCHANGED)
    result = detectBall(ball)
    print(result)
