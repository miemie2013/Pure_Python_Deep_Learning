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
from layers.base_layer import LossLayer


class MSELoss(LossLayer):
    def __init__(self):
        super(MSELoss, self).__init__()

    def train_forward(self, y_pred, y_true):
        self.input_1 = np.copy(y_pred)      # 保存一下输入，反向传播时会使用到
        self.input_2 = np.copy(y_true)      # 保存一下输入，反向传播时会使用到

        loss = (y_pred - y_true) ** 2  # 先把差值逐项平方
        loss = np.mean(loss)  # 再求平均，即mse损失函数

        self.output = np.copy(loss)   # 保存一下输出，反向传播时会使用到
        return loss

    def train_backward(self, optimizer):
        '''
        因为损失层mseloss损失层没有可训练权重，所以直接对本层的输入y_pred求偏导，以更新前面的层的权重。
        y_pred里的元素用y表示，y_true里的元素用Y表示。为了提高泛用性，y_pred不一定是4维。
        loss = [(y_1 - Y_1)^2 + (y_2 - Y_2)^2 + ... + (y_i - Y_i)^2 + ...] / len(y_pred)
        对y_i求偏导，偏导符号打不出来，用d表示了。
        dloss/dy_i = 2*(y_i - Y_i) / len(y_pred)， 这个偏导数的表达式还是很简单的。
        :param lr:  学习率
        :return:
        '''
        y_pred, y_true, loss = self.input_1, self.input_2, self.output
        shape = np.shape(y_pred)
        len = 1.0  # y_pred的元素个数，即公式里的len(y_pred)
        for s in shape:
            len *= s
        grad = 2.0 * (y_pred - y_true) / len
        return grad





