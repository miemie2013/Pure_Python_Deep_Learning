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
from layers.gnnorm import GroupNorm
from layers.loss import MSELoss
from layers.activations import *
from layers.optimizer import *


import paddle
paddle.enable_static()




def gnnorm(input, norm_groups=4, name=''):
    scale_name = name + "_scale"
    offset_name = name + "_offset"
    scale_param = fluid.ParamAttr(
        name=scale_name,
        initializer=fluid.initializer.Constant(1.0),
        trainable=True)
    offset_param = fluid.ParamAttr(
        name=offset_name,
        initializer=fluid.initializer.Constant(0.0),
        trainable=True)

    out = fluid.layers.group_norm(
        input=input,
        act=None,
        groups=norm_groups,
        param_attr=scale_param,
        bias_attr=offset_param)
    return out




if __name__ == '__main__':
    use_gpu = False

    lr = 0.1
    epsilon = 1e-5


    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs = P.data(name='input_1', shape=[-1, 3, 28, 28], append_batch_size=False, dtype='float32')
            conv01_out_tensor = fluid.layers.conv2d(input=inputs, num_filters=8, filter_size=1, stride=1, padding=0,
                                                    param_attr=ParamAttr(name="conv01_weights"),
                                                    bias_attr=ParamAttr(name="conv01_bias"))
            gn_name = "gn01"
            gn01_out_tensor = gnnorm(conv01_out_tensor, norm_groups=4, name=gn_name)
            act01_out_tensor = fluid.layers.leaky_relu(gn01_out_tensor, alpha=0.1)

            conv02_out_tensor = fluid.layers.conv2d(input=act01_out_tensor, num_filters=8, filter_size=3, stride=1, padding=1,
                                                    param_attr=ParamAttr(name="conv02_weights"),
                                                    bias_attr=ParamAttr(name="conv02_bias"))
            gn_name = "gn02"
            gn02_out_tensor = gnnorm(conv02_out_tensor, norm_groups=2, name=gn_name)
            act02_out_tensor = fluid.layers.leaky_relu(gn02_out_tensor, alpha=0.1)


            # 建立损失函数
            y_true = P.data(name='y_true', shape=[-1, 8, 28, 28], append_batch_size=False, dtype='float32')
            # 先把差值逐项平方，可以用P.pow()这个op，也可以用python里的运算符**。
            mseloss = P.pow(y_true - act02_out_tensor, 2)
            mseloss = P.reduce_mean(mseloss)       # 再求平均，即mse损失函数

            # 优化器，选SGD
            optimizer = fluid.optimizer.SGD(learning_rate=lr)
            optimizer.minimize(mseloss)


    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            # 重新建立一次网络，用相同的张量名，不用写损失层
            inputs = P.data(name='input_1', shape=[-1, 3, 28, 28], append_batch_size=False, dtype='float32')
            conv01_out_tensor = fluid.layers.conv2d(input=inputs, num_filters=8, filter_size=1, stride=1, padding=0,
                                                    param_attr=ParamAttr(name="conv01_weights"),
                                                    bias_attr=ParamAttr(name="conv01_bias"))
            gn_name = "gn01"
            gn01_out_tensor = gnnorm(conv01_out_tensor, norm_groups=4, name=gn_name)
            act01_out_tensor = fluid.layers.leaky_relu(gn01_out_tensor, alpha=0.1)

            conv02_out_tensor = fluid.layers.conv2d(input=act01_out_tensor, num_filters=8, filter_size=3, stride=1, padding=1,
                                                    param_attr=ParamAttr(name="conv02_weights"),
                                                    bias_attr=ParamAttr(name="conv02_bias"))
            gn_name = "gn02"
            gn02_out_tensor = gnnorm(conv02_out_tensor, norm_groups=2, name=gn_name)
            act02_out_tensor = fluid.layers.leaky_relu(gn02_out_tensor, alpha=0.1)
            eval_fetch_list = [gn01_out_tensor, gn02_out_tensor]
    eval_prog = eval_prog.clone(for_test=True)
    # 参数初始化
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    compiled_eval_prog = fluid.compiler.CompiledProgram(eval_prog)


    # 纯python搭建的神经网络的权重。初始值是paddle相同层的初始值。为了模拟paddle训练过程。
    # 1.卷积层
    paddle_conv01_weights = np.array(fluid.global_scope().find_var('conv01_weights').get_tensor())
    paddle_conv01_bias = np.array(fluid.global_scope().find_var('conv01_bias').get_tensor())
    # 2.gn层
    paddle_gn01_scale = np.array(fluid.global_scope().find_var('gn01_scale').get_tensor())
    paddle_gn01_offset = np.array(fluid.global_scope().find_var('gn01_offset').get_tensor())
    # 3.激活层
    # 4.卷积层
    paddle_conv02_weights = np.array(fluid.global_scope().find_var('conv02_weights').get_tensor())
    paddle_conv02_bias = np.array(fluid.global_scope().find_var('conv02_bias').get_tensor())
    # 5.gn层
    paddle_gn02_scale = np.array(fluid.global_scope().find_var('gn02_scale').get_tensor())
    paddle_gn02_offset = np.array(fluid.global_scope().find_var('gn02_offset').get_tensor())
    # 6.激活层
    # 7.损失函数层，没有权重。

    #  纯python搭建的神经网络
    conv01 = Conv2D(3, num_filters=8, filter_size=1, stride=1, padding=0, use_bias=True, name='conv01')
    gn01 = GroupNorm(8, num_groups=4, epsilon=epsilon, name='gn01')
    act01 = LeakyReLU(alpha=0.1)
    conv02 = Conv2D(8, num_filters=8, filter_size=3, stride=1, padding=1, use_bias=True, name='conv02')
    gn02 = GroupNorm(8, num_groups=2, epsilon=epsilon, name='gn02')
    act02 = LeakyReLU(alpha=0.1)
    mse01 = MSELoss()
    optimizer2 = SGD(lr=lr)
    # 初始化自己网络的权重
    conv01.init_weights(paddle_conv01_weights, paddle_conv01_bias)
    gn01.init_weights(paddle_gn01_scale, paddle_gn01_offset)
    conv02.init_weights(paddle_conv02_weights, paddle_conv02_bias)
    gn02.init_weights(paddle_gn02_scale, paddle_gn02_offset)


    # 只训练8步
    for step in range(8):
        print('------------------ step %d ------------------' % step)
        # ==================== train ====================
        batch_data = np.random.normal(loc=0, scale=1, size=(2, 3, 28, 28)).astype(np.float32)
        y_true_arr = np.random.normal(loc=0, scale=1, size=(2, 8, 28, 28)).astype(np.float32)

        paddle_mseloss_out, paddle_gn01_out, paddle_gn02_out = exe.run(train_prog, feed={"input_1": batch_data, "y_true": y_true_arr, },
                                                                       fetch_list=[mseloss, gn01_out_tensor, gn02_out_tensor])

        print('train_forward:')
        # python代码模拟训练过程，与paddle的输出校验。我们希望和飞桨有相同的输出。
        my_conv01_out = conv01.train_forward(batch_data)
        my_gn01_out = gn01.train_forward(my_conv01_out)
        my_act01_out = act01.train_forward(my_gn01_out)
        my_conv02_out = conv02.train_forward(my_act01_out)
        my_gn02_out = gn02.train_forward(my_conv02_out)
        my_act02_out = act02.train_forward(my_gn02_out)
        my_mseloss_out = mse01.train_forward(my_act02_out, y_true_arr)


        diff_gn01_out = np.sum((paddle_gn01_out - my_gn01_out) ** 2)
        print('diff_gn01_out=%.6f' % diff_gn01_out)   # 若是0，则表示成功模拟出PaddlePaddle的输出结果
        diff_gn02_out = np.sum((paddle_gn02_out - my_gn02_out) ** 2)
        print('diff_gn02_out=%.6f' % diff_gn02_out)   # 若是0，则表示成功模拟出PaddlePaddle的输出结果
        diff_mseloss_out = np.sum((paddle_mseloss_out - my_mseloss_out)**2)
        print('diff_mseloss_out=%.6f' % diff_mseloss_out)   # 若是0，则表示成功模拟出PaddlePaddle in层的输出结果

        print('\ntrain_backward:')
        # 纯python搭建的神经网络进行反向传播啦！求偏导数即可。反向传播会更新权重，我们期望和飞桨有相同的权重。
        my_act02_out_grad = mse01.train_backward(optimizer2)
        my_gn02_out_grad = act02.train_backward(my_act02_out_grad, optimizer2)
        my_conv02_out_grad = gn02.train_backward(my_gn02_out_grad, optimizer2)
        my_act01_out_grad = conv02.train_backward(my_conv02_out_grad, optimizer2)
        my_gn01_out_grad = act01.train_backward(my_act01_out_grad, optimizer2)
        my_conv01_out_grad = gn01.train_backward(my_gn01_out_grad, optimizer2)
        inputs_grad = conv01.train_backward(my_conv01_out_grad, optimizer2)

        # 和飞桨更新后的权重校验。
        paddle_gn01_scale = np.array(fluid.global_scope().find_var('gn01_scale').get_tensor())
        paddle_gn01_offset = np.array(fluid.global_scope().find_var('gn01_offset').get_tensor())
        paddle_gn02_scale = np.array(fluid.global_scope().find_var('gn02_scale').get_tensor())
        paddle_gn02_offset = np.array(fluid.global_scope().find_var('gn02_offset').get_tensor())


        diff_gn02_scale = np.sum((paddle_gn02_scale - gn02.scale) ** 2)
        print('diff_gn02_scale=%.6f' % diff_gn02_scale)   # 若是0，则表示成功模拟出权重更新
        diff_gn02_offset = np.sum((paddle_gn02_offset - gn02.offset) ** 2)
        print('diff_gn02_offset=%.6f' % diff_gn02_offset)   # 若是0，则表示成功模拟出权重更新
        diff_gn01_scale = np.sum((paddle_gn01_scale - gn01.scale) ** 2)
        print('diff_gn01_scale=%.6f' % diff_gn01_scale)   # 若是0，则表示成功模拟出权重更新
        diff_gn01_offset = np.sum((paddle_gn01_offset - gn01.offset) ** 2)
        print('diff_gn01_offset=%.6f' % diff_gn01_offset)   # 若是0，则表示成功模拟出权重更新

        # ==================== test ====================
        test_data = np.random.normal(loc=0, scale=1, size=(2, 3, 28, 28)).astype(np.float32)
        paddle_test_in01_out, paddle_test_in02_out = exe.run(compiled_eval_prog, feed={"input_1": test_data, }, fetch_list=eval_fetch_list)
        # 自己网络的test
        print('\ntest_forward:')
        my_test_conv01_out = conv01.test_forward(test_data)
        my_test_in01_out = gn01.test_forward(my_test_conv01_out)
        my_test_act01_out = act01.test_forward(my_test_in01_out)
        my_test_conv02_out = conv02.test_forward(my_test_act01_out)
        my_test_in02_out = gn02.test_forward(my_test_conv02_out)
        my_test_act02_out = act02.test_forward(my_test_in02_out)
        diff_test_in01_out = np.sum((paddle_test_in01_out - my_test_in01_out)**2)
        print('diff_test_in01_out=%.6f' % diff_test_in01_out)   # 若是0，则表示成功模拟出推理
        diff_test_in02_out = np.sum((paddle_test_in02_out - my_test_in02_out)**2)
        print('diff_test_in02_out=%.6f' % diff_test_in02_out)   # 若是0，则表示成功模拟出推理



