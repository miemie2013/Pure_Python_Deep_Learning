#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-07-05 13:35:27
#   Description : 《纯python实现一个深度学习框架》课程源码
#
# ================================================================
import datetime
import json
from collections import deque
import paddle.fluid as fluid
import paddle.fluid.layers as P
import sys
import time
import shutil
import math
import copy
import random
import threading
import numpy as np
import os
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from layers.fc import FC
from layers.conv2d import Conv2D
from layers.bn import BatchNorm
from layers.loss import MSELoss
from layers.activations import *
from layers.optimizer import *

from test.custom_layers import *

import paddle


class PaddleNet(paddle.nn.Layer):
    def __init__(self, norm_type='bn', name=''):
        super(PaddleNet, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn', 'affine_channel']
        bn, gn, af = get_norm(norm_type)
        self.name = name
        self.conv1 = Conv2dUnit(3, 8, 1, stride=1, bias_attr=True, bn=bn, gn=gn, af=af, freeze_norm=False, norm_decay=0.0, act='leaky', name='conv01')
        self.conv2 = Conv2dUnit(8, 8, 3, stride=1, bias_attr=True, bn=bn, gn=gn, af=af, freeze_norm=False, norm_decay=0.0, act='leaky', name='conv02')

        # freeze
        # self.conv1.freeze()
        # self.conv2.freeze()

    def __call__(self, input_tensor):
        conv01_out, norm01_out, act01_out = self.conv1(input_tensor)
        conv02_out, norm02_out, act02_out = self.conv2(act01_out)
        return conv01_out, norm01_out, act01_out, conv02_out, norm02_out, act02_out




if __name__ == '__main__':
    use_gpu = False
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()

    lr = 0.1

    model = PaddleNet()
    model.train()

    # optimizer
    optim_args = dict()
    regularization = None
    optimizer = paddle.optimizer.SGD(learning_rate=lr,
                                     parameters=model.parameters(),
                                     weight_decay=regularization,   # 正则化
                                     grad_clip=None,   # 梯度裁剪
                                     **optim_args)


    # 参数初始化
    param_state_dict = model.state_dict()


    # 纯python搭建的神经网络的权重。初始值是paddle相同层的初始值。为了模拟paddle训练过程。
    # 1.卷积层
    paddle_conv01_weights = param_state_dict['conv1.conv.weight'].numpy()
    paddle_conv01_bias = param_state_dict['conv1.conv.bias'].numpy()
    # 2.bn层
    paddle_bn01_scale = param_state_dict['conv1.bn.weight'].numpy()
    paddle_bn01_offset = param_state_dict['conv1.bn.bias'].numpy()
    paddle_bn01_mean = param_state_dict['conv1.bn._mean'].numpy()
    paddle_bn01_variance = param_state_dict['conv1.bn._variance'].numpy()
    # 3.激活层
    # 4.卷积层
    paddle_conv02_weights = param_state_dict['conv2.conv.weight'].numpy()
    paddle_conv02_bias = param_state_dict['conv2.conv.bias'].numpy()
    # 5.bn层
    paddle_bn02_scale = param_state_dict['conv2.bn.weight'].numpy()
    paddle_bn02_offset = param_state_dict['conv2.bn.bias'].numpy()
    paddle_bn02_mean = param_state_dict['conv2.bn._mean'].numpy()
    paddle_bn02_variance = param_state_dict['conv2.bn._variance'].numpy()
    # 6.激活层
    # 7.损失函数层，没有权重。

    #  纯python搭建的神经网络
    conv01 = Conv2D(3, num_filters=8, filter_size=1, stride=1, padding=0, use_bias=True, name='conv01')
    bn01 = BatchNorm(8, momentum=0.9, epsilon=1e-05, name='bn01')   # 我们跟随paddle的bn层，使用了相同的momentum值和epsilon值
    act01 = LeakyReLU(alpha=0.1)
    conv02 = Conv2D(8, num_filters=8, filter_size=3, stride=1, padding=1, use_bias=True, name='conv02')
    bn02 = BatchNorm(8, momentum=0.9, epsilon=1e-05, name='bn02')   # 我们跟随paddle的bn层，使用了相同的momentum值和epsilon值
    act02 = LeakyReLU(alpha=0.1)
    mse01 = MSELoss()
    optimizer2 = SGD(lr=lr)
    # 初始化自己网络的权重
    conv01.init_weights(paddle_conv01_weights, paddle_conv01_bias)
    bn01.init_weights(paddle_bn01_scale, paddle_bn01_offset)
    bn01.init_means_vars(paddle_bn01_mean, paddle_bn01_variance)
    conv02.init_weights(paddle_conv02_weights, paddle_conv02_bias)
    bn02.init_weights(paddle_bn02_scale, paddle_bn02_offset)
    bn02.init_means_vars(paddle_bn02_mean, paddle_bn02_variance)


    # 只训练8步
    for step in range(8):
        print('------------------ step %d ------------------' % step)
        # ==================== train ====================
        batch_data = np.random.normal(loc=0, scale=1, size=(2, 3, 28, 28)).astype(np.float32)
        y_true_arr = np.random.normal(loc=0, scale=1, size=(2, 8, 28, 28)).astype(np.float32)

        batch_data2 = paddle.to_tensor(batch_data, place=place)
        y_true_arr2 = paddle.to_tensor(y_true_arr, place=place)

        paddle_conv01_out, paddle_bn01_out, paddle_act01_out, paddle_conv02_out, paddle_bn02_out, paddle_act02_out = model(batch_data2)

        # 建立损失函数
        # 先把差值逐项平方，可以用P.pow()这个op，也可以用python里的运算符**。
        mseloss = P.pow(y_true_arr2 - paddle_act02_out, 2)
        mseloss = P.reduce_mean(mseloss)       # 再求平均，即mse损失函数

        paddle_mseloss_out = mseloss.numpy()
        paddle_bn01_out = paddle_bn01_out.numpy()
        paddle_bn02_out = paddle_bn02_out.numpy()

        # 更新权重
        mseloss.backward()
        optimizer.step()
        optimizer.clear_grad()


        print('train_forward:')
        # python代码模拟训练过程，与paddle的输出校验。我们希望和飞桨有相同的输出。
        my_conv01_out = conv01.train_forward(batch_data)
        my_bn01_out = bn01.train_forward(my_conv01_out)
        my_act01_out = act01.train_forward(my_bn01_out)
        my_conv02_out = conv02.train_forward(my_act01_out)
        my_bn02_out = bn02.train_forward(my_conv02_out)
        my_act02_out = act02.train_forward(my_bn02_out)
        my_mseloss_out = mse01.train_forward(my_act02_out, y_true_arr)


        diff_bn01_out = np.sum((paddle_bn01_out - my_bn01_out)**2)
        print('diff_bn01_out=%.6f' % diff_bn01_out)   # 若是0，则表示成功模拟出PaddlePaddle的输出结果
        diff_bn02_out = np.sum((paddle_bn02_out - my_bn02_out)**2)
        print('diff_bn02_out=%.6f' % diff_bn02_out)   # 若是0，则表示成功模拟出PaddlePaddle的输出结果
        diff_mseloss_out = np.sum((paddle_mseloss_out - my_mseloss_out)**2)
        print('diff_mseloss_out=%.6f' % diff_mseloss_out)   # 若是0，则表示成功模拟出PaddlePaddle bn层的输出结果

        print('\ntrain_backward:')
        # 纯python搭建的神经网络进行反向传播啦！求偏导数即可。反向传播会更新权重，我们期望和飞桨有相同的权重。
        my_act02_out_grad = mse01.train_backward(optimizer2)
        my_bn02_out_grad = act02.train_backward(my_act02_out_grad, optimizer2)
        my_conv02_out_grad = bn02.train_backward(my_bn02_out_grad, optimizer2)
        my_act01_out_grad = conv02.train_backward(my_conv02_out_grad, optimizer2)
        my_bn01_out_grad = act01.train_backward(my_act01_out_grad, optimizer2)
        my_conv01_out_grad = bn01.train_backward(my_bn01_out_grad, optimizer2)
        inputs_grad = conv01.train_backward(my_conv01_out_grad, optimizer2)

        # 和飞桨更新后的权重校验。
        paddle_bn01_scale = param_state_dict['conv1.bn.weight'].numpy()
        paddle_bn01_offset = param_state_dict['conv1.bn.bias'].numpy()
        paddle_bn01_mean = param_state_dict['conv1.bn._mean'].numpy()
        paddle_bn01_variance = param_state_dict['conv1.bn._variance'].numpy()
        paddle_bn02_scale = param_state_dict['conv2.bn.weight'].numpy()
        paddle_bn02_offset = param_state_dict['conv2.bn.bias'].numpy()
        paddle_bn02_mean = param_state_dict['conv2.bn._mean'].numpy()
        paddle_bn02_variance = param_state_dict['conv2.bn._variance'].numpy()


        diff_bn02_scale = np.sum((paddle_bn02_scale - bn02.scale)**2)
        print('diff_bn02_scale=%.6f' % diff_bn02_scale)   # 若是0，则表示成功模拟出权重更新
        diff_bn02_offset = np.sum((paddle_bn02_offset - bn02.offset)**2)
        print('diff_bn02_offset=%.6f' % diff_bn02_offset)   # 若是0，则表示成功模拟出权重更新
        diff_bn01_scale = np.sum((paddle_bn01_scale - bn01.scale)**2)
        print('diff_bn01_scale=%.6f' % diff_bn01_scale)   # 若是0，则表示成功模拟出权重更新
        diff_bn01_offset = np.sum((paddle_bn01_offset - bn01.offset)**2)
        print('diff_bn01_offset=%.6f' % diff_bn01_offset)   # 若是0，则表示成功模拟出权重更新

        # 均值和方差，在train_forward()阶段就已经被更新
        diff_bn02_mean = np.sum((paddle_bn02_mean - bn02.mean)**2)
        print('diff_bn02_mean=%.6f' % diff_bn02_mean)   # 若是0，则表示成功模拟出均值更新
        diff_bn02_variance = np.sum((paddle_bn02_variance - bn02.var)**2)
        print('diff_bn02_variance=%.6f' % diff_bn02_variance)   # 若是0，则表示成功模拟出方差更新
        diff_bn01_mean = np.sum((paddle_bn01_mean - bn01.mean)**2)
        print('diff_bn01_mean=%.6f' % diff_bn01_mean)   # 若是0，则表示成功模拟出均值更新
        diff_bn01_variance = np.sum((paddle_bn01_variance - bn01.var)**2)
        print('diff_bn01_variance=%.6f' % diff_bn01_variance)   # 若是0，则表示成功模拟出方差更新

        # ==================== test ====================
        test_data = np.random.normal(loc=0, scale=1, size=(2, 3, 28, 28)).astype(np.float32)
        model.eval()
        test_data2 = paddle.to_tensor(test_data, place=place)
        paddle_conv01_out, paddle_bn01_out, paddle_act01_out, paddle_conv02_out, paddle_bn02_out, paddle_act02_out = model(test_data2)
        paddle_test_bn01_out = paddle_bn01_out.numpy()
        paddle_test_bn02_out = paddle_bn02_out.numpy()
        model.train()
        # 自己网络的test
        print('\ntest_forward:')
        my_test_conv01_out = conv01.test_forward(test_data)
        my_test_bn01_out = bn01.test_forward(my_test_conv01_out)
        my_test_act01_out = act01.test_forward(my_test_bn01_out)
        my_test_conv02_out = conv02.test_forward(my_test_act01_out)
        my_test_bn02_out = bn02.test_forward(my_test_conv02_out)
        my_test_act02_out = act02.test_forward(my_test_bn02_out)
        diff_test_bn01_out = np.sum((paddle_test_bn01_out - my_test_bn01_out)**2)
        print('diff_test_bn01_out=%.6f' % diff_test_bn01_out)   # 若是0，则表示成功模拟出推理
        diff_test_bn02_out = np.sum((paddle_test_bn02_out - my_test_bn02_out)**2)
        print('diff_test_bn02_out=%.6f' % diff_test_bn02_out)   # 若是0，则表示成功模拟出推理



