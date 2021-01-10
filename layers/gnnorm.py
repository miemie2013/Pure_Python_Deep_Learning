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


class GroupNorm(Layer):
    def __init__(self,
                 in_C,
                 num_groups=1,
                 epsilon=1e-05,
                 w_decay_type=None,
                 w_decay=0.,
                 b_decay_type=None,
                 b_decay=0.,
                 w_lr=1.,
                 b_lr=1.,
                 name=''):
        super(GroupNorm, self).__init__()
        if in_C % num_groups != 0:
            raise ArithmeticError
        self.in_C = in_C
        self.n = num_groups
        self.g = in_C // num_groups
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
        n = self.n
        g = self.g

        N, C, H, W = x.shape
        x = np.reshape(x, (N, n, g, H, W))   # [N, n, g, H, W]   C=n*g

        # 求均值。
        U = np.mean(x, axis=(2, 3, 4), keepdims=True)  # [N, n, 1, 1, 1]
        # 求方差。
        V = np.var(x, axis=(2, 3, 4), keepdims=True)   # [N, n, 1, 1, 1]
        # 归一化
        normX = (x - U) / np.sqrt(V + epsilon)   # [N, n, g, H, W]
        normX = np.reshape(normX, (N, C, H, W))


        # gamma和beta扩展成x的形状
        S = np.reshape(Scale, (1, -1, 1, 1))  # [1, C, 1, 1]
        B = np.reshape(Bias, (1, -1, 1, 1))   # [1, C, 1, 1]

        # in测试时前向传播的公式。用的是当前图片的均值和方差进行归一化。
        y = normX * S + B
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
        n = self.n
        g = self.g

        N, C, H, W = x.shape
        x = np.reshape(x, (N, n, g, H, W))   # [N, n, g, H, W]   C=n*g

        axis = (2, 3, 4)
        gHW = g*H*W
        # 求均值。
        U = np.mean(x, axis=axis, keepdims=True)  # [N, n, 1, 1, 1]
        # 求方差。
        V = np.var(x, axis=axis, keepdims=True)   # [N, n, 1, 1, 1]
        # 归一化
        normX = (x - U) / np.sqrt(V + epsilon)   # [N, n, g, H, W]
        normX = np.reshape(normX, (N, C, H, W))

        # gamma和beta扩展成normX的形状
        S = np.reshape(Scale, (1, n, g, 1, 1))  # [1, n, g, 1, 1]
        B = np.reshape(Bias, (1, n, g, 1, 1))   # [1, n, g, 1, 1]


        # loss对Bias的偏导数。很简单，就是dL_dBias = grad
        dL_dBias = np.sum(grad, axis=(0, 2, 3))  # [N, C, H, W] -> [C, ]
        # loss对Scale的偏导数。很简单，就是dL_dScale = grad * normX
        dL_dScale = grad * normX   # [N, C, H, W]
        dL_dScale = np.sum(dL_dScale, axis=(0, 2, 3))  # [N, C, H, W] -> [C, ]

        # loss对输入x的偏导数，用来更新前面的层的权重。这是最难的一个。
        grad = np.reshape(grad, (N, n, g, H, W))
        dL_dnormX = S * grad
        dnormX_dV = (x - U) * -0.5 / (V + epsilon) ** 1.5
        dL_dV = np.sum(dL_dnormX * dnormX_dV, axis=axis, keepdims=True)
        std = np.sqrt(V + epsilon)
        dV_dx_part2 = 2 * (x - U) / gHW

        # intermediate for convenient calculation
        di = dL_dnormX / std + dL_dV * dV_dx_part2
        dmean = -1 * np.sum(di, axis=axis, keepdims=True)
        dX = di + dmean / gHW
        dX = np.reshape(dX, (N, C, H, W))


        # 更新可训练参数
        Scale = optimizer.update(self.name+'_scale', Scale, dL_dScale, self.w_lr, self.w_decay_type, self.w_decay)
        Bias = optimizer.update(self.name+'_offset', Bias, dL_dBias, self.b_lr, self.b_decay_type, self.b_decay)
        self.offset = Bias
        self.scale = Scale
        return dX




