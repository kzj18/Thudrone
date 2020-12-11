# /usr/bin/env python
# -*-coding:utf-8*-
'''
主干网络——特征提取网络DarkNet53的实现
'''

import torch
import torch.nn as nn
import math
from collections import OrderedDict

class BasicBlock(nn.Module):    # 自定义神经网络模型
    '''
    实现残差网络，包含主干边和残差边，残差边的存在使得残差网络更加容易优化和训练
    '''
    def __init__(self, inplanes, planes):
        '''
        对主干边进行初始化，包括两组卷积、标准化和激活
        '''
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False) # 使用1x1的卷积核减少通道数
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)    # 使用3x3的卷积核增加通道数，这两步卷积配合使用可以减少数据量
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        '''
        残差网络前向传播
        '''
        residual = x    # 生成残差边

        # 主干边的前向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual # 残差边与主干边的输出相加
        return out


class DarkNet(nn.Module):   # 自定义神经网络模型
    '''
    实现DarkNet特征提取网络
    '''
    def __init__(self, layers):
        '''
        初始化DarkNet网络
        '''
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)    # 使用3x3的卷积核进行卷积，输出通道数（卷积核个数）为32
        self.bn1 = nn.BatchNorm2d(self.inplanes)    # 初始化BatchNorm归一化层，该层的作用是，在ReLU非线性激活层之前，对数据进行归一化处理，防止数据量过大
        self.relu1 = nn.LeakyReLU(0.1)  # 采用LeakyReLU函数作为非线性激活函数，避免了在使用DeadReLU函数遇到的“输出为零的神经元不会更新”问题

        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        '''
        每层包括下采样和残差残差网络的堆叠两步
        '''
        layers = []
        # 下采样
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False))) # 下采样，步长为2（使得网络的尺寸逐步减半），卷积核大小为3
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1]))) # 标准化
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))   # 激活
        # 堆叠残差网络
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes))) # 堆叠残差网络
        return nn.Sequential(OrderedDict(layers))   # 返回简单的顺序连接模型

    def forward(self, x):
        '''
        DarkNet53网络的前向传播
        '''
        # 第一个卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 第一个残差层
        x = self.layer2(x)  # 第二个残差层
        out3 = self.layer3(x)   # 第三个残差层
        out4 = self.layer4(out3)    # 第四个残差层
        out5 = self.layer5(out4)    # 第五个残差层

        return out3, out4, out5 # 将第三、四、五个残差层的结果输出

def darknet53(pretrained, **kwargs):
    model = DarkNet([1, 2, 8, 8, 4])    # 将DarkNet53特征提取网络中残差块使用的次数传入
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
