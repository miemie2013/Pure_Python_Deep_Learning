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
from layers.base_layer import Layer


class InstanceNorm(Layer):
    def __init__(self,
                 in_C,
                 epsilon=1e-05,
                 w_decay_type=None,
                 w_decay=0.,
                 b_decay_type=None,
                 b_decay=0.,
                 w_lr=1.,
                 b_lr=1.,
                 name=''):
        super(InstanceNorm, self).__init__()

        self.in_C = in_C
        self.epsilon = epsilon
        assert w_decay_type in ['L1Decay', 'L2Decay', None]
        assert b_decay_type in ['L1Decay', 'L2Decay', None]
        self.w_decay_type = w_decay_type
        self.b_decay_type = b_decay_type
        self.w_decay = w_decay
        self.b_decay = b_decay
        self.w_lr = w_lr
        self.b_lr = b_lr
        self.name = name

        self.scale = np.ones((self.in_C, ), np.float32)    # 1初始化
        self.offset = np.zeros((self.in_C, ), np.float32)  # 0初始化

    def init_weights(self, scale, offset):
        self.scale = np.copy(scale)
        self.offset = np.copy(offset)

    def test_forward(self, x):
        Scale = self.scale
        Bias = self.offset
        epsilon = self.epsilon

        input_shape = x.shape
        N, C, H, W = input_shape

        # 求均值。
        U = np.mean(x, axis=(2, 3), keepdims=True)  # [N, C, 1, 1]
        # 求方差。
        V = np.var(x, axis=(2, 3), keepdims=True)   # [N, C, 1, 1]

        # gamma和beta扩展成x的形状
        S = np.reshape(Scale, (1, -1, 1, 1))  # [1, C, 1, 1]
        B = np.reshape(Bias, (1, -1, 1, 1))   # [1, C, 1, 1]

        # bn测试时前向传播的公式。用的是统计的均值和方差进行归一化。
        y = (x - U) / np.sqrt(V + epsilon) * S + B
        return y

    def train_forward(self, x):
        self.input = np.copy(x)    # 保存一下输入，反向传播时会使用到
        y = self.test_forward(x)
        self.output = np.copy(y)   # 保存一下输出，反向传播时会使用到
        return y

    def train_backward(self, grad, optimizer):
        '''
        对本层的权重求偏导，以更新本层的权重。对本层的输入x求偏导，以更新前面的层的权重。
        设本层的权重是w，若loss = f(a, b, c, ...) = a(w)+b(w)+c(w)+...，那么loss对w的偏导数(偏导符号打不出来，用d表示了)
        dloss/dw = dloss/da * da/dw + dloss/db * db/dw + dloss/dc * dc/dw + ...
        a, b, c是中间变量，在这一层，self.output里的所有元素就是这个例子里的a, b, c, ...是中间变量，
        而dloss/da, dloss/db, dloss/dc, ...即grad里的所有元素。所以我们只需要求出
        da/dw, db/dw, dc/dw, ...就可以求出dloss/dw了！ 求loss对本层输入的偏导数，同理。
        '''
        x, y = self.input, self.output   # 获取训练时前向传播的输入输出
        Scale = self.scale    # 可训练参数
        Bias = self.offset    # 可训练参数
        epsilon = self.epsilon

        axis = (2, 3)
        # 求均值。
        U = np.mean(x, axis=axis, keepdims=True)  # [N, C, 1, 1]
        # 求方差。
        V = np.var(x, axis=axis, keepdims=True)   # [N, C, 1, 1]

        # gamma和beta扩展成x的形状
        S = np.reshape(Scale, (1, -1, 1, 1))  # [1, C, 1, 1]
        B = np.reshape(Bias, (1, -1, 1, 1))   # [1, C, 1, 1]


        input_shape = x.shape
        N, C, H, W = input_shape
        HW = H*W

        # loss对Bias的偏导数。很简单，就是dL_dBias = grad
        dL_dBias = np.sum(grad, axis=(0, 2, 3))  # [M, C] -> [C, ]
        # loss对Scale的偏导数。很简单，就是dL_dScale = grad * normX
        dL_dScale = grad * (x-U) / (V + epsilon)**0.5  # [M, C]
        dL_dScale = np.sum(dL_dScale, axis=(0, 2, 3))  # [M, C] -> [C, ]

        # loss对输入x的偏导数，用来更新前面的层的权重。这是最难的一个。
        dL_dnormX = S * grad
        dnormX_dV = (x - U) * -0.5 / (V + epsilon) ** 1.5
        dL_dV = np.sum(dL_dnormX * dnormX_dV, axis=axis, keepdims=True)
        std = np.sqrt(V + epsilon)
        dV_dx_part2 = 2 * (x - U) / HW

        # intermediate for convenient calculation
        di = dL_dnormX / std + dL_dV * dV_dx_part2
        dmean = -1 * np.sum(di, axis=axis, keepdims=True)
        dX = di + dmean / HW


        # 更新可训练参数
        Scale = optimizer.update(self.name+'_scale', Scale, dL_dScale, self.w_lr, self.w_decay_type, self.w_decay)
        Bias = optimizer.update(self.name+'_offset', Bias, dL_dBias, self.b_lr, self.b_decay_type, self.b_decay)
        self.offset = Bias
        self.scale = Scale
        return dX




