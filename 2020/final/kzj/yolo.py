# /usr/bin/env python
# -*-coding:utf-8*-

'''
对图片进行检测并画框
'''

import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from nets.yolo3 import YoloBody
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFont, ImageDraw
from torch.autograd import Variable
from utils.config import Config
from utils.utils import non_max_suppression, DecodeBox, letterbox_image, yolo_correct_boxes
import copy
import time

python_file = os.path.dirname(__file__)
save_path = python_file + '/data/output/picture'

class YOLO(object):
    _defaults = {
        "model_path"        : '',
        "classes_path"      : '',
        "model_image_size"  : (416, 416, 3),
        "confidence"        : 0.5,
        "iou"               : 0.3,
        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    
    def __init__(self, weight, classes, **kwargs):
        '''
        初始化
        '''
        self._defaults['model_path'] = weight
        self._defaults['classes_path'] = classes
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        Config_copy = copy.deepcopy(Config)
        Config_copy['yolo']['classes'] = len(self.class_names)
        self.config = Config
        self.generate()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
    def _get_class(self):
        '''
        从classes_path中获得检测的类别
        '''
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
        
    def generate(self):
        '''
        从model_path中加载模型，并对画框颜色进行初始化
        '''
        self.config["yolo"]["classes"] = len(self.class_names)
        self.net = YoloBody(self.config)    # 初始化yolo_v3网络

        # 加快模型训练的效率
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 对于没有独显的电脑使用CPU进行计算
        state_dict = torch.load(self.model_path, map_location=device)   # 加载已经训练好的模型
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)    # 如果电脑存在多张显卡，则通过多GPU训练加快训练速度
            self.net = self.net.cuda()

        # 往yolo_decodes中添加三个不同大小的解码器，分别解码三个输出结果
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.config["yolo"]["anchors"][i], self.config["yolo"]["classes"],  (self.model_image_size[1], self.model_image_size[0])))


        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def detect_image(self, image, counter):
        '''
        使用模型对图片进行检测
        '''
        image_copy = copy.deepcopy(image)
        result_list = []
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))   # 使用utils中的letterbox_image进行灰条的添加，使之符合标准输入格式的大小
        photo = np.array(crop_img, dtype = np.float32)
        photo /= 255.0  # 对RGB三个通道的值进行归一化
        photo = np.transpose(photo, (2, 0, 1))  # 调整通道，方便GPU处理
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)    # 扩充维度

        images = np.asarray(images)
        images = torch.from_numpy(images)   # 将图片转换为tensor格式
        if torch.cuda.is_available():
            images = images.cuda()  # 将图片转移到cuda上
        
        with torch.no_grad():
            outputs = self.net(images)  # 将图片传入YOLOV3网络，并将预测结果赋给outputs
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))    # 利用三个解码器对预测结果进行解码
            output = torch.cat(output_list, 1)  # 对预测结果进行堆叠
            # 对预测结果进行非极大抑制，同时包含了置信度筛选
            batch_detections = non_max_suppression(output, self.config["yolo"]["classes"],
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)
        try :
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return []    # 判断原图中是否还存在框，如果经过非极大抑制，框已经消失，则返回原图
        top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence   # 用置信度再次筛选
        top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
        top_label = np.array(batch_detections[top_index,-1],np.int32)
        top_bboxes = np.array(batch_detections[top_index,:4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

        # 将预测框的坐标转换到没有灰条的图片的坐标下
        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            result_list.append([score, int(c)])
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label), fill=(0, 0, 0), font=font)
            del draw
            image.save(save_path + '/yolo_counter_' + str(counter) + time.strftime('_%b_%d_%Y_%H_%M_%S_result.png'))
            image_copy.save(save_path + '/yolo_counter_' + str(counter) + time.strftime('_%b_%d_%Y_%H_%M_%S_original.png'))
        return result_list

