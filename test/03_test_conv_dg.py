#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-07-05 13:35:27
#   Description : 《纯python实现一个深度学习框架》课程源码
#
# ================================================================
import os

from paddle.fluid.initializer import Normal
from paddle.fluid.param_attr import ParamAttr

from layers.conv2d import Conv2D
from layers.loss import MSELoss
from layers.optimizer import *

import paddle
import paddle.nn as nn


class PaddleNet(paddle.nn.Layer):
    def __init__(self):
        super(PaddleNet, self).__init__()
        self.conv1 = nn.Conv2D(3, 8, 1, stride=1, padding=0,
                               bias_attr=ParamAttr(name="conv01_bias", initializer=Normal()))
        self.conv2 = nn.Conv2D(8, 8, 3, stride=1, padding=1, groups=1,
                               bias_attr=ParamAttr(name="conv02_bias", initializer=Normal()))

    def __call__(self, input_tensor):
        conv01_out_tensor = self.conv1(input_tensor)
        conv02_out_tensor = self.conv2(conv01_out_tensor)
        return conv01_out_tensor, conv02_out_tensor



if __name__ == '__main__':
    use_gpu = False
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()

    lr = 0.001

    model = PaddleNet()
    model.train()

    # optimizer
    optim_args = dict(momentum=0.0, use_nesterov=False)
    optimizer = paddle.optimizer.Momentum(learning_rate=lr,
                                          parameters=model.parameters(),
                                          grad_clip=None,
                                          **optim_args)

    # 参数初始化
    param_state_dict = model.state_dict()


    # 纯python搭建的神经网络的权重。初始值是paddle相同层的初始值。为了模拟paddle训练过程。
    # 1.卷积层
    paddle_conv01_weights = param_state_dict['conv1.weight'].numpy()
    paddle_conv01_bias = param_state_dict['conv1.bias'].numpy()
    # 2.卷积层
    paddle_conv02_weights = param_state_dict['conv2.weight'].numpy()
    paddle_conv02_bias = param_state_dict['conv2.bias'].numpy()
    # 3.损失函数层，没有权重。

    #  纯python搭建的神经网络
    conv01 = Conv2D(3, num_filters=8, filter_size=1, stride=1, padding=0, use_bias=True, name='conv01')
    conv02 = Conv2D(8, num_filters=8, filter_size=3, stride=1, padding=1, groups=1, use_bias=True, name='conv02')
    mse01 = MSELoss()
    optimizer2 = SGD(lr=lr)
    # 初始化自己网络的权重
    conv01.init_weights(paddle_conv01_weights, paddle_conv01_bias)
    conv02.init_weights(paddle_conv02_weights, paddle_conv02_bias)


    # 只训练8步
    for step in range(8):
        print('------------------ step %d ------------------' % step)
        # ==================== train ====================
        batch_data = np.random.normal(loc=0, scale=1, size=(2, 3, 28, 28)).astype(np.float32)
        y_true_arr = np.random.normal(loc=0, scale=1, size=(2, 8, 28, 28)).astype(np.float32)

        batch_data2 = paddle.to_tensor(batch_data, place=place)
        y_true_arr2 = paddle.to_tensor(y_true_arr, place=place)

        paddle_conv01_out_tensor, paddle_conv02_out_tensor = model(batch_data2)

        # 先把差值逐项平方，可以用P.pow()这个op，也可以用python里的运算符**。
        mseloss = paddle.pow(y_true_arr2 - paddle_conv02_out_tensor, 2)
        mseloss = paddle.mean(mseloss)  # 再求平均，即mse损失函数

        paddle_mseloss_out = mseloss.numpy()
        paddle_conv01_out = paddle_conv01_out_tensor.numpy()
        paddle_conv02_out = paddle_conv02_out_tensor.numpy()

        # 更新权重
        optimizer.clear_grad()
        mseloss.backward()
        if step % 1 == 0:
            optimizer.step()

        print('train_forward:')
        # python代码模拟训练过程，与paddle的输出校验。我们希望和飞桨有相同的输出。
        my_conv01_out = conv01.train_forward(batch_data)
        my_conv02_out = conv02.train_forward(my_conv01_out)
        my_mseloss_out = mse01.train_forward(my_conv02_out, y_true_arr)


        diff_conv01_out = np.sum((paddle_conv01_out - my_conv01_out)**2)
        print('diff_conv01_out=%.6f' % diff_conv01_out)   # 若是0，则表示成功模拟出PaddlePaddle的输出结果
        diff_conv02_out = np.sum((paddle_conv02_out - my_conv02_out)**2)
        print('diff_conv02_out=%.6f' % diff_conv02_out)   # 若是0，则表示成功模拟出PaddlePaddle的输出结果
        diff_mseloss_out = np.sum((paddle_mseloss_out - my_mseloss_out)**2)
        print('diff_mseloss_out=%.6f' % diff_mseloss_out)   # 若是0，则表示成功模拟出PaddlePaddle bn层的输出结果

        print('\ntrain_backward:')
        # 纯python搭建的神经网络进行反向传播啦！求偏导数即可。反向传播会更新权重，我们期望和飞桨有相同的权重。
        my_conv02_out_grad = mse01.train_backward(optimizer2)
        my_conv01_out_grad = conv02.train_backward(my_conv02_out_grad, optimizer2)
        inputs_grad = conv01.train_backward(my_conv01_out_grad, optimizer2)

        # 和飞桨更新后的权重校验。
        paddle_conv01_weights = param_state_dict['conv1.weight'].numpy()
        paddle_conv01_bias = param_state_dict['conv1.bias'].numpy()
        paddle_conv02_weights = param_state_dict['conv2.weight'].numpy()
        paddle_conv02_bias = param_state_dict['conv2.bias'].numpy()


        diff_conv02_weights = np.sum((paddle_conv02_weights - conv02.w)**2)
        print('diff_conv02_weights=%.6f' % diff_conv02_weights)   # 若是0，则表示成功模拟出权重更新
        diff_conv02_bias = np.sum((paddle_conv02_bias - conv02.b)**2)
        print('diff_conv02_bias=%.6f' % diff_conv02_bias)   # 若是0，则表示成功模拟出权重更新
        diff_conv01_weights = np.sum((paddle_conv01_weights - conv01.w)**2)
        print('diff_conv01_weights=%.6f' % diff_conv01_weights)   # 若是0，则表示成功模拟出权重更新
        diff_conv01_bias = np.sum((paddle_conv01_bias - conv01.b)**2)
        print('diff_conv01_bias=%.6f' % diff_conv01_bias)   # 若是0，则表示成功模拟出权重更新

        # ==================== test ====================
        test_data = np.random.normal(loc=0, scale=1, size=(2, 3, 28, 28)).astype(np.float32)
        model.eval()
        test_data2 = paddle.to_tensor(test_data, place=place)
        paddle_conv01_out_tensor, paddle_conv02_out_tensor = model(test_data2)
        paddle_test_conv01_out = paddle_conv01_out_tensor.numpy()
        paddle_test_conv02_out = paddle_conv02_out_tensor.numpy()
        model.train()
        # 自己网络的test
        print('\ntest_forward:')
        my_test_conv01_out = conv01.test_forward(test_data)
        my_test_conv02_out = conv02.test_forward(my_test_conv01_out)
        diff_test_conv01_out = np.sum((paddle_test_conv01_out - my_test_conv01_out)**2)
        print('diff_test_conv01_out=%.6f' % diff_test_conv01_out)   # 若是0，则表示成功模拟出推理
        diff_test_conv02_out = np.sum((paddle_test_conv02_out - my_test_conv02_out)**2)
        print('diff_test_conv02_out=%.6f' % diff_test_conv02_out)   # 若是0，则表示成功模拟出推理



