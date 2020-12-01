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


import paddle
paddle.enable_static()

if __name__ == '__main__':
    # 测试L2Decay()、卷积层参数学习率设为0.3和2.0（默认都是1.0）、Momentum优化算法
    use_gpu = False

    lr = 0.1

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs = P.data(name='input_1', shape=[-1, 3, 28, 28], append_batch_size=False, dtype='float32')
            conv01_out_tensor = fluid.layers.conv2d(input=inputs, num_filters=8, filter_size=1, stride=1, padding=0,
                                                    param_attr=ParamAttr(name="conv01_weights"))
            bn_name = "bn01"
            bn01_out_tensor = fluid.layers.batch_norm(
                input=conv01_out_tensor,
                is_test=False,
                param_attr=ParamAttr(initializer=fluid.initializer.Constant(1.0), name=bn_name + '_scale', regularizer=L2Decay(0.)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name=bn_name + '_offset', regularizer=L2Decay(0.)),
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance')
            act01_out_tensor = fluid.layers.leaky_relu(bn01_out_tensor, alpha=0.1)

            conv02_out_tensor = fluid.layers.conv2d(input=act01_out_tensor, num_filters=8, filter_size=3, stride=1, padding=1,
                                                    param_attr=ParamAttr(name="conv02_weights", learning_rate=0.3),
                                                    bias_attr=ParamAttr(name="conv02_bias", regularizer=L2Decay(0.), learning_rate=2.0))


            # 建立损失函数
            y_true = P.data(name='y_true', shape=[-1, 8, 28, 28], append_batch_size=False, dtype='float32')
            # 先把差值逐项平方，可以用P.pow()这个op，也可以用python里的运算符**。
            mseloss = P.pow(y_true - conv02_out_tensor, 2)
            mseloss = P.reduce_mean(mseloss)       # 再求平均，即mse损失函数

            # 优化器
            optim_args = dict(momentum=0.9,)
            regularization = fluid.regularizer.L2Decay(0.0005)
            optimizer = fluid.optimizer.Momentum(learning_rate=lr, regularization=regularization, **optim_args)
            # optimizer = fluid.optimizer.SGD(learning_rate=lr, regularization=regularization)
            optimizer.minimize(mseloss)


    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            # 重新建立一次网络，用相同的张量名，不用写损失层
            inputs = P.data(name='input_1', shape=[-1, 3, 28, 28], append_batch_size=False, dtype='float32')
            conv01_out_tensor = fluid.layers.conv2d(input=inputs, num_filters=8, filter_size=1, stride=1, padding=0,
                                                    param_attr=ParamAttr(name="conv01_weights"))
            bn_name = "bn01"
            bn01_out_tensor = fluid.layers.batch_norm(
                input=conv01_out_tensor,
                is_test=False,
                param_attr=ParamAttr(initializer=fluid.initializer.Constant(1.0), name=bn_name + '_scale', regularizer=L2Decay(0.)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name=bn_name + '_offset', regularizer=L2Decay(0.)),
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance')
            act01_out_tensor = fluid.layers.leaky_relu(bn01_out_tensor, alpha=0.1)

            conv02_out_tensor = fluid.layers.conv2d(input=act01_out_tensor, num_filters=8, filter_size=3, stride=1, padding=1,
                                                    param_attr=ParamAttr(name="conv02_weights", learning_rate=0.3),
                                                    bias_attr=ParamAttr(name="conv02_bias", regularizer=L2Decay(0.), learning_rate=2.0))
            eval_fetch_list = [bn01_out_tensor, conv02_out_tensor]
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
    # 2.bn层
    paddle_bn01_scale = np.array(fluid.global_scope().find_var('bn01_scale').get_tensor())
    paddle_bn01_offset = np.array(fluid.global_scope().find_var('bn01_offset').get_tensor())
    paddle_bn01_mean = np.array(fluid.global_scope().find_var('bn01_mean').get_tensor())
    paddle_bn01_variance = np.array(fluid.global_scope().find_var('bn01_variance').get_tensor())
    # 3.激活层
    # 4.卷积层
    paddle_conv02_weights = np.array(fluid.global_scope().find_var('conv02_weights').get_tensor())
    paddle_conv02_bias = np.array(fluid.global_scope().find_var('conv02_bias').get_tensor())


    #  纯python搭建的神经网络
    conv01 = Conv2D(3, num_filters=8, filter_size=1, stride=1, padding=0, use_bias=False, w_decay_type='L2Decay', w_decay=0.0005, name='conv01')
    bn01 = BatchNorm(8, momentum=0.9, epsilon=1e-05, name='bn01')   # 我们跟随paddle的bn层，使用了相同的momentum值和epsilon值
    act01 = LeakyReLU(alpha=0.1)
    conv02 = Conv2D(8, num_filters=8, filter_size=3, stride=1, padding=1, use_bias=True, w_decay_type='L2Decay', w_decay=0.0005, w_lr=0.3, b_lr=2.0, name='conv02')
    mse01 = MSELoss()
    optimizer2 = Momentum(lr=lr, momentum=0.9, use_nesterov=False)
    # optimizer2 = SGD(lr=lr)
    # 初始化自己网络的权重
    conv01.init_weights(paddle_conv01_weights, None)
    bn01.init_weights(paddle_bn01_scale, paddle_bn01_offset)
    bn01.init_means_vars(paddle_bn01_mean, paddle_bn01_variance)
    conv02.init_weights(paddle_conv02_weights, paddle_conv02_bias)


    # 只训练8步
    for step in range(8):
        print('------------------ step %d ------------------' % step)
        # ==================== train ====================
        batch_data = np.random.normal(loc=0, scale=1, size=(2, 3, 28, 28)).astype(np.float32)
        y_true_arr = np.random.normal(loc=0, scale=1, size=(2, 8, 28, 28)).astype(np.float32)

        paddle_mseloss_out, paddle_bn01_out, paddle_conv02_out = exe.run(train_prog, feed={"input_1": batch_data, "y_true": y_true_arr, },
                                                                       fetch_list=[mseloss, bn01_out_tensor, conv02_out_tensor])

        print('train_forward:')
        # python代码模拟训练过程，与paddle的输出校验。我们希望和飞桨有相同的输出。
        my_conv01_out = conv01.train_forward(batch_data)
        my_bn01_out = bn01.train_forward(my_conv01_out)
        my_act01_out = act01.train_forward(my_bn01_out)
        my_conv02_out = conv02.train_forward(my_act01_out)
        my_mseloss_out = mse01.train_forward(my_conv02_out, y_true_arr)


        diff_bn01_out = np.sum((paddle_bn01_out - my_bn01_out)**2)
        print('diff_bn01_out=%.6f' % diff_bn01_out)   # 若是0，则表示成功模拟出PaddlePaddle的输出结果
        diff_conv02_out = np.sum((paddle_conv02_out - my_conv02_out)**2)
        print('diff_conv02_out=%.6f' % diff_conv02_out)   # 若是0，则表示成功模拟出PaddlePaddle的输出结果
        diff_mseloss_out = np.sum((paddle_mseloss_out - my_mseloss_out)**2)
        print('diff_mseloss_out=%.6f' % diff_mseloss_out)   # 若是0，则表示成功模拟出PaddlePaddle bn层的输出结果

        print('\ntrain_backward:')
        # 纯python搭建的神经网络进行反向传播啦！求偏导数即可。反向传播会更新权重，我们期望和飞桨有相同的权重。
        my_conv02_out_grad = mse01.train_backward(optimizer2)
        my_act01_out_grad = conv02.train_backward(my_conv02_out_grad, optimizer2)
        my_bn01_out_grad = act01.train_backward(my_act01_out_grad, optimizer2)
        my_conv01_out_grad = bn01.train_backward(my_bn01_out_grad, optimizer2)
        inputs_grad = conv01.train_backward(my_conv01_out_grad, optimizer2)

        # 和飞桨更新后的权重校验。
        paddle_conv01_weights = np.array(fluid.global_scope().find_var('conv01_weights').get_tensor())
        paddle_bn01_scale = np.array(fluid.global_scope().find_var('bn01_scale').get_tensor())
        paddle_bn01_offset = np.array(fluid.global_scope().find_var('bn01_offset').get_tensor())
        paddle_bn01_mean = np.array(fluid.global_scope().find_var('bn01_mean').get_tensor())
        paddle_bn01_variance = np.array(fluid.global_scope().find_var('bn01_variance').get_tensor())
        paddle_conv02_weights = np.array(fluid.global_scope().find_var('conv02_weights').get_tensor())
        paddle_conv02_bias = np.array(fluid.global_scope().find_var('conv02_bias').get_tensor())


        diff_conv02_weights = np.sum((paddle_conv02_weights - conv02.w)**2)
        print('diff_conv02_weights=%.6f' % diff_conv02_weights)   # 若是0，则表示成功模拟出权重更新
        diff_conv02_bias = np.sum((paddle_conv02_bias - conv02.b)**2)
        print('diff_conv02_bias=%.6f' % diff_conv02_bias)   # 若是0，则表示成功模拟出权重更新

        diff_bn01_scale = np.sum((paddle_bn01_scale - bn01.scale)**2)
        print('diff_bn01_scale=%.6f' % diff_bn01_scale)   # 若是0，则表示成功模拟出权重更新
        diff_bn01_offset = np.sum((paddle_bn01_offset - bn01.offset)**2)
        print('diff_bn01_offset=%.6f' % diff_bn01_offset)   # 若是0，则表示成功模拟出权重更新

        diff_conv01_weights = np.sum((paddle_conv01_weights - conv01.w)**2)
        print('diff_conv01_weights=%.6f' % diff_conv01_weights)   # 若是0，则表示成功模拟出权重更新

        # 均值和方差，在train_forward()阶段就已经被更新
        diff_bn01_mean = np.sum((paddle_bn01_mean - bn01.mean)**2)
        print('diff_bn01_mean=%.6f' % diff_bn01_mean)   # 若是0，则表示成功模拟出均值更新
        diff_bn01_variance = np.sum((paddle_bn01_variance - bn01.var)**2)
        print('diff_bn01_variance=%.6f' % diff_bn01_variance)   # 若是0，则表示成功模拟出方差更新

        # ==================== test ====================
        test_data = np.random.normal(loc=0, scale=1, size=(2, 3, 28, 28)).astype(np.float32)
        paddle_test_bn01_out, paddle_test_conv02_out = exe.run(compiled_eval_prog, feed={"input_1": test_data, }, fetch_list=eval_fetch_list)
        # 自己网络的test
        print('\ntest_forward:')
        my_test_conv01_out = conv01.test_forward(test_data)
        my_test_bn01_out = bn01.test_forward(my_test_conv01_out)
        my_test_act01_out = act01.test_forward(my_test_bn01_out)
        my_test_conv02_out = conv02.test_forward(my_test_act01_out)
        diff_test_bn01_out = np.sum((paddle_test_bn01_out - my_test_bn01_out)**2)
        print('diff_test_bn01_out=%.6f' % diff_test_bn01_out)   # 若是0，则表示成功模拟出推理
        diff_test_conv02_out = np.sum((paddle_test_conv02_out - my_test_conv02_out)**2)
        print('diff_test_conv02_out=%.6f' % diff_test_conv02_out)   # 若是0，则表示成功模拟出推理



