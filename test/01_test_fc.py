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
from layers.bn import BatchNorm
from layers.loss import MSELoss
from layers.optimizer import *

import paddle
paddle.enable_static()

if __name__ == '__main__':
    use_gpu = False

    lr = 0.001

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs = P.data(name='input_1', shape=[-1, 3], append_batch_size=False, dtype='float32')
            fc01_out_tensor = fluid.layers.fc(input=inputs, size=8,
                                                    param_attr=ParamAttr(name="fc01_weights"),
                                                    bias_attr=ParamAttr(name="fc01_bias"))
            fc02_out_tensor = fluid.layers.fc(input=fc01_out_tensor, size=8,
                                                    param_attr=ParamAttr(name="fc02_weights"),
                                                    bias_attr=ParamAttr(name="fc02_bias"))


            # 建立损失函数
            y_true = P.data(name='y_true', shape=[-1, 8], append_batch_size=False, dtype='float32')
            # 先把差值逐项平方，可以用P.pow()这个op，也可以用python里的运算符**。
            mseloss = P.pow(y_true - fc02_out_tensor, 2)
            # mseloss = (y_true - bn01_out_tensor) ** 2   # 也可以用python里的运算符**。
            mseloss = P.reduce_mean(mseloss)       # 再求平均，即mse损失函数

            # 优化器，选SGD
            optimizer = fluid.optimizer.SGD(learning_rate=lr)
            optimizer.minimize(mseloss)


    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            # 重新建立一次网络，用相同的张量名，不用写损失层
            inputs = P.data(name='input_1', shape=[-1, 3], append_batch_size=False, dtype='float32')
            fc01_out_tensor = fluid.layers.fc(input=inputs, size=8,
                                                    param_attr=ParamAttr(name="fc01_weights"),
                                                    bias_attr=ParamAttr(name="fc01_bias"))
            fc02_out_tensor = fluid.layers.fc(input=fc01_out_tensor, size=8,
                                                    param_attr=ParamAttr(name="fc02_weights"),
                                                    bias_attr=ParamAttr(name="fc02_bias"))
            eval_fetch_list = [fc01_out_tensor, fc02_out_tensor]
    eval_prog = eval_prog.clone(for_test=True)
    # 参数初始化
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    compiled_eval_prog = fluid.compiler.CompiledProgram(eval_prog)


    # 纯python搭建的神经网络的权重。初始值是paddle相同层的初始值。为了模拟paddle训练过程。
    # 1.fc层
    paddle_fc01_weights = np.array(fluid.global_scope().find_var('fc01_weights').get_tensor())
    paddle_fc01_bias = np.array(fluid.global_scope().find_var('fc01_bias').get_tensor())
    # 2.fc层
    paddle_fc02_weights = np.array(fluid.global_scope().find_var('fc02_weights').get_tensor())
    paddle_fc02_bias = np.array(fluid.global_scope().find_var('fc02_bias').get_tensor())
    # 3.损失函数层，没有权重。

    #  纯python搭建的神经网络
    fc01 = FC(3, 8, use_bias=True)
    fc02 = FC(8, 8, use_bias=True)
    mse01 = MSELoss()
    optimizer2 = SGD(lr=lr)
    # 初始化自己网络的权重
    fc01.init_weights(paddle_fc01_weights, paddle_fc01_bias)
    fc02.init_weights(paddle_fc02_weights, paddle_fc02_bias)


    # 只训练8步
    for step in range(8):
        print('------------------ step %d ------------------' % step)
        # ==================== train ====================
        batch_data = np.random.normal(loc=0, scale=1, size=(2, 3)).astype(np.float32)
        y_true_arr = np.random.normal(loc=0, scale=1, size=(2, 8)).astype(np.float32)

        paddle_mseloss_out, paddle_fc01_out, paddle_fc02_out = exe.run(train_prog, feed={"input_1": batch_data, "y_true": y_true_arr, },
                                                                       fetch_list=[mseloss, fc01_out_tensor, fc02_out_tensor])

        print('train_forward:')
        # python代码模拟训练过程，与paddle的输出校验。我们希望和飞桨有相同的输出。
        my_fc01_out = fc01.train_forward(batch_data)
        my_fc02_out = fc02.train_forward(my_fc01_out)
        my_mseloss_out = mse01.train_forward(my_fc02_out, y_true_arr)


        diff_fc01_out = np.sum((paddle_fc01_out - my_fc01_out)**2)
        print('diff_fc01_out=%.6f' % diff_fc01_out)   # 若是0，则表示成功模拟出PaddlePaddle的输出结果
        diff_fc02_out = np.sum((paddle_fc02_out - my_fc02_out)**2)
        print('diff_fc02_out=%.6f' % diff_fc02_out)   # 若是0，则表示成功模拟出PaddlePaddle的输出结果
        diff_mseloss_out = np.sum((paddle_mseloss_out - my_mseloss_out)**2)
        print('diff_mseloss_out=%.6f' % diff_mseloss_out)   # 若是0，则表示成功模拟出PaddlePaddle的输出结果

        print('\ntrain_backward:')
        # 纯python搭建的神经网络进行反向传播啦！求偏导数即可。反向传播会更新权重，我们期望和飞桨有相同的权重。
        my_fc02_out_grad = mse01.train_backward(lr)
        my_fc01_out_grad = fc02.train_backward(my_fc02_out_grad, lr)
        inputs_grad = fc01.train_backward(my_fc01_out_grad, lr)

        # 和飞桨更新后的权重校验。
        paddle_fc01_weights = np.array(fluid.global_scope().find_var('fc01_weights').get_tensor())
        paddle_fc01_bias = np.array(fluid.global_scope().find_var('fc01_bias').get_tensor())
        paddle_fc02_weights = np.array(fluid.global_scope().find_var('fc02_weights').get_tensor())
        paddle_fc02_bias = np.array(fluid.global_scope().find_var('fc02_bias').get_tensor())


        diff_fc02_weights = np.sum((paddle_fc02_weights - fc02.w)**2)
        print('diff_fc02_weights=%.6f' % diff_fc02_weights)   # 若是0，则表示成功模拟出权重更新
        diff_fc02_bias = np.sum((paddle_fc02_bias - fc02.b)**2)
        print('diff_fc02_bias=%.6f' % diff_fc02_bias)   # 若是0，则表示成功模拟出权重更新
        diff_fc01_weights = np.sum((paddle_fc01_weights - fc01.w)**2)
        print('diff_fc01_weights=%.6f' % diff_fc01_weights)   # 若是0，则表示成功模拟出权重更新
        diff_fc01_bias = np.sum((paddle_fc01_bias - fc01.b)**2)
        print('diff_fc01_bias=%.6f' % diff_fc01_bias)   # 若是0，则表示成功模拟出权重更新

        # ==================== test ====================
        test_data = np.random.normal(loc=0, scale=1, size=(2, 3)).astype(np.float32)
        paddle_test_fc01_out, paddle_test_fc02_out = exe.run(compiled_eval_prog, feed={"input_1": test_data, }, fetch_list=eval_fetch_list)
        # 自己网络的test
        print('\ntest_forward:')
        my_test_fc01_out = fc01.test_forward(test_data)
        my_test_fc02_out = fc02.test_forward(my_test_fc01_out)
        diff_test_fc01_out = np.sum((paddle_test_fc01_out - my_test_fc01_out)**2)
        print('diff_test_fc01_out=%.6f' % diff_test_fc01_out)   # 若是0，则表示成功模拟出推理
        diff_test_fc02_out = np.sum((paddle_test_fc02_out - my_test_fc02_out)**2)
        print('diff_test_fc02_out=%.6f' % diff_test_fc02_out)   # 若是0，则表示成功模拟出推理



