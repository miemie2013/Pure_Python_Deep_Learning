#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-07-05 13:35:27
#   Description : 《纯python实现一个深度学习框架》课程源码
#
# ================================================================
import numpy as np
from layers.base_layer import ActivationLayer


class Sigmoid(ActivationLayer):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def test_forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def train_forward(self, x):
        self.input = np.copy(x)      # 保存一下输入，反向传播时会使用到
        out = self.test_forward(x)
        self.output = np.copy(out)   # 保存一下输出，反向传播时会使用到
        return out

    def train_backward(self, grad, lr):
        '''
        对本层的输入x求偏导，以更新前面的层的权重。
        y = sigmoid(x)
        dy/dx = sigmoid(x)*(1-sigmoid(x)) = y*(1-y)
        dloss/dx = dloss/dy * dy/dx = dloss/dy * y*(1-y)
        '''
        x, out = self.input, self.output   # 获取训练时前向传播的输入输出

        # loss对输入x的偏导数，用来更新前面的层的权重
        dX = grad * out * (1-out)   # 形状任意，反正out和grad形状是一样的，支持逐元素相乘
        return dX


class ReLU(ActivationLayer):
    def __init__(self):
        super(ReLU, self).__init__()

    def test_forward(self, x):
        return np.maximum(0, x)

    def train_forward(self, x):
        self.input = np.copy(x)      # 保存一下输入，反向传播时会使用到
        out = self.test_forward(x)
        self.output = np.copy(out)   # 保存一下输出，反向传播时会使用到
        return out

    def train_backward(self, grad, lr):
        '''
        对本层的输入x求偏导，以更新前面的层的权重。
        y = ReLU(x)
        dy/dx = 1 if x>0 else 0
        '''
        x, out = self.input, self.output   # 获取训练时前向传播的输入输出

        # loss对输入x的偏导数，用来更新前面的层的权重
        pos = (x > 0).astype(np.float32)
        dX = grad * pos   # 形状任意，反正out和grad形状是一样的，支持逐元素相乘
        return dX




class LeakyReLU(ActivationLayer):
    def __init__(self, alpha=0.1):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha

    def test_forward(self, x):
        pos = (x > 0).astype(np.float32)
        return pos * x + (1.0 - pos) * self.alpha * x

    def train_forward(self, x):
        self.input = np.copy(x)      # 保存一下输入，反向传播时会使用到
        out = self.test_forward(x)
        self.output = np.copy(out)   # 保存一下输出，反向传播时会使用到
        return out

    def train_backward(self, grad, lr):
        '''
        对本层的输入x求偏导，以更新前面的层的权重。
        y = LeakyReLU(x)
        dy/dx = 1 if x>0 else alpha
        '''
        x, out = self.input, self.output   # 获取训练时前向传播的输入输出

        # loss对输入x的偏导数，用来更新前面的层的权重
        pos = (x > 0).astype(np.float32)
        dX = grad * pos + (1.0 - pos) * self.alpha * grad
        return dX





