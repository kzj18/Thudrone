# /usr/bin/env python
# -*-coding:utf-8*-

import torch
import torch.nn as nn
from collections import OrderedDict
from nets.darknet import darknet53

def conv2d(filter_in, filter_out, kernel_size):
    '''
    卷积模块，包括卷积、标准化和激活
    '''
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

def make_last_layers(filters_list, in_filters, out_filter):
    '''
    输出网络的实现
    '''
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1), # 调整输出通道数
        conv2d(filters_list[0], filters_list[1], 3),    # 特征提取，这样的结构堆叠可以减小网络的参数量，使网络更加轻便
        conv2d(filters_list[1], filters_list[0], 1),    # 调整输出通道数
        conv2d(filters_list[0], filters_list[1], 3),    # 特征提取
        conv2d(filters_list[1], filters_list[0], 1),    # 调整输出通道数
        conv2d(filters_list[0], filters_list[1], 3),    # 特征提取
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)   # 输出结果用于分类预测和回归预测
    ])
    return m

class YoloBody(nn.Module):  # 自定义神经网络模型
    def __init__(self, config):
        '''
        初始化YOLOV3神经网络
        '''
        super(YoloBody, self).__init__()
        self.config = config
        
        self.backbone = darknet53(None) # 获取DarkNet53神经网络结构

        out_filters = self.backbone.layers_out_filters
        
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], final_out_filter0)

        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest') # 初始化上采样网络
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1)

        final_out_filter2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2)


    def forward(self, x):
        def _branch(last_layer, layer_in):
            '''
            输出网络的前五层网络的输出out_branch，以及最终的输出layer_in
            '''
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch
        
        x2, x1, x0 = self.backbone(x)   # 对应DarkNet53网络的三个输出
        
        out0, out0_branch = _branch(self.last_layer0, x0)

        # 对最底层网络的结果进行卷积、上采样并和次底层结果进行堆叠，从而构建特征金字塔。利用特征金字塔可以进行多尺度特征的融合，提取出更有效的特征
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.last_layer1, x1_in)

        # 重复上述操作
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, _ = _branch(self.last_layer2, x2_in)
        return out0, out1, out2 # 通过这三个输出参数可以判断先验框内是否有物体（1）、物体的种类（n）以及如何调整先验框的大小（4），输出参数的通道数都为3x(1+n+4)

