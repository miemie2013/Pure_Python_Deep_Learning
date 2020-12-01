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


class FC(Layer):
    def __init__(self,
                 in_C,
                 size,
                 use_bias=False,
                 w_decay_type=None,
                 w_decay=0.,
                 b_decay_type=None,
                 b_decay=0.,
                 w_lr=1.,
                 b_lr=1.,
                 name=''):
        super(FC, self).__init__()

        self.in_C = in_C
        self.size = size
        assert w_decay_type in ['L1Decay', 'L2Decay', None]
        assert b_decay_type in ['L1Decay', 'L2Decay', None]
        self.w_decay_type = w_decay_type
        self.b_decay_type = b_decay_type
        self.w_decay = w_decay
        self.b_decay = b_decay
        self.w_lr = w_lr
        self.b_lr = b_lr
        self.name = name

        self.w = np.zeros((self.in_C, self.size), np.float32)
        self.b = None
        if use_bias:
            self.b = np.zeros((self.size, ), np.float32)

    def init_weights(self, w):
        self.w = np.copy(w)

    def init_weights(self, w, b):
        self.w = np.copy(w)
        if self.b is not None:
            self.b = np.copy(b)

    def test_forward(self, x):
        # 全连接层推理时的前向传播。
        w = self.w
        b = self.b
        out = np.matmul(x, w)
        if b is not None:
            # 先让b增加第0维变成形状为(1, self.size)的张量exp_b。
            exp_b = np.expand_dims(b, 0)   # (1, self.size)
            out = out + exp_b   # 与out相加时exp_b的第0维会自动重复N次来和out的形状对齐。
        return out

    def train_forward(self, x):
        # 全连接训练时的前向传播和推理时的前向传播是相同的。多了一个保存操作而已。
        self.input = np.copy(x)      # 保存一下输入，反向传播时会使用到
        out = self.test_forward(x)
        self.output = np.copy(out)   # 保存一下输出，反向传播时会使用到
        return out

    def train_backward(self, grad, optimizer):
        '''
        对本层的权重求偏导，以更新本层的权重。对本层的输入x求偏导，以更新前面的层的权重。
        设本层的权重是w，若loss = f(a, b, c, ...) = a(w)+b(w)+c(w)+...，那么loss对w的偏导数(偏导符号打不出来，用d表示了)
        dloss/dw = dloss/da * da/dw + dloss/db * db/dw + dloss/dc * dc/dw + ...
        a, b, c是中间变量，在这一层，self.output里的所有元素就是这个例子里的a, b, c, ...是中间变量，
        而dloss/da, dloss/db, dloss/dc, ...即grad里的所有元素。所以我们只需要求出
        da/dw, db/dw, dc/dw, ...就可以求出dloss/dw了！ 求loss对本层输入的偏导数，同理。
        '''
        x, out = self.input, self.output   # 获取训练时前向传播的输入输出
        N, in_C = x.shape
        w = self.w   # [in_C, size]
        b = self.b   # [size, ]

        # loss对Bias的偏导数
        if b is not None:  # grad [N, size]      W  [in_C, size]      x [N, in_C]
            dB = np.sum(grad, axis=0)  # 中文记法：等于loss对本层输出的梯度 对0维求和。 多个样本共享了偏移B，所以求和

        # loss对W的偏导数。
        exp_grad = np.expand_dims(grad, 1)   # [N, 1, size]
        exp_X = np.expand_dims(x, 2)         # [N, in_C, 1]
        dW = exp_grad * exp_X                # [N, in_C, size]
        dW = np.sum(dW, axis=(0, ))          # [in_C, size]  多个样本共享了权重W，所以求和

        # loss对输入x的偏导数，用来更新前面的层的权重
        exp_W = np.expand_dims(w, 0)         # [1, in_C, size]
        dX = exp_grad * exp_W                # [N, in_C, size]
        dX = np.sum(dX, axis=(2, ))          # [N, in_C]   把偏移数量那一维求和

        # 更新可训练参数
        if b is not None:
            b = optimizer.update(self.name + '_bias', b, dB, self.b_lr, self.b_decay_type, self.b_decay)
            self.b = b
        w = optimizer.update(self.name+'_weight', w, dW, self.w_lr, self.w_decay_type, self.w_decay)
        self.w = w
        return dX

