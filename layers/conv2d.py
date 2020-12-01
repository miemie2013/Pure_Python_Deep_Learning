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


class Conv2D(Layer):
    def __init__(self,
                 in_C,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 use_bias=False,
                 w_decay_type=None,
                 w_decay=0.,
                 b_decay_type=None,
                 b_decay=0.,
                 w_lr=1.,
                 b_lr=1.):
        super(Conv2D, self).__init__()

        self.in_C = in_C
        self.out_C = num_filters
        self.kH = filter_size
        self.kW = filter_size
        self.stride = stride
        self.padding = padding
        self.num_filters = num_filters
        assert w_decay_type in ['L1Decay', 'L2Decay', None]
        assert b_decay_type in ['L1Decay', 'L2Decay', None]
        self.w_decay_type = w_decay_type
        self.b_decay_type = b_decay_type
        self.w_decay = w_decay
        self.b_decay = b_decay
        self.w_lr = w_lr
        self.b_lr = b_lr

        self.w = np.zeros((self.out_C, self.in_C, self.kH, self.kW), np.float32)
        self.b = None
        if use_bias:
            self.b = np.zeros((self.out_C, ), np.float32)

    def init_weights(self, w):
        self.w = np.copy(w)

    def init_weights(self, w, b):
        self.w = np.copy(w)
        if self.b is not None:
            self.b = np.copy(b)

    def test_forward(self, x):
        # 卷积层推理时的前向传播。
        N, C, H, W = x.shape
        out_C, in_C, kH, kW = self.w.shape
        w = self.w
        b = self.b
        stride = self.stride
        padding = self.padding
        assert (C == in_C), "x.shape[1] must equal in_C."
        out_W = (W+2*padding-(kW-1)) // stride
        out_H = (H+2*padding-(kH-1)) // stride
        out = np.zeros((N, out_C, out_H, out_W), np.float32)

        # 1.先对图片x填充得到填充后的图片pad_x
        pad_x = np.zeros((N, C, H + padding*2, W + padding*2), np.float32)
        pad_x[:, :, padding:padding + H, padding:padding + W] = x

        # 2.卷积核滑动，只会在H和W两个方向上滑动
        for i in range(out_H):   # i是纵坐标
            for j in range(out_W):   # j是横坐标
                ori_x = j*stride   # 卷积核在pad_x中的横坐标，等差数列，公差是stride
                ori_y = i*stride   # 卷积核在pad_x中的纵坐标，等差数列，公差是stride
                part_x = pad_x[:, :, ori_y:ori_y+kH, ori_x:ori_x+kW]   # 截取卷积核所处的位置的像素 [N, in_C, kH, kW]
                exp_part_x = np.expand_dims(part_x, 1)   # 增加1维，[N, 1,     in_C, kH, kW]。
                exp_w = np.expand_dims(w, 0)      # 卷积核也增加1维，[1, out_C, in_C, kH, kW]。
                mul = exp_part_x * exp_w   # 卷积核和exp_part_x相乘，[N, out_C, in_C, kH, kW]。
                mul = np.sum(mul, axis=(2, 3, 4))       # 后3维求和，[N, out_C]。
                if b is not None:
                    mul += b    # 加上偏移，[N, out_C]。
                # 将得到的新像素写进out的对应位置
                out[:, :, i, j] = mul
        return out

    def train_forward(self, x):
        # 卷积层训练时的前向传播和推理时的前向传播是相同的。多了一个保存操作而已。
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
        N, C, H, W = x.shape
        N, out_C, out_H, out_W = out.shape
        out_C, in_C, kH, kW = self.w.shape
        w = self.w
        b = self.b
        stride = self.stride
        padding = self.padding

        # 对图片x填充得到填充后的图片pad_x
        pad_x = np.zeros((N, C, H + padding*2, W + padding*2), np.float32)
        pad_x[:, :, padding:padding + H, padding:padding + W] = x

        # loss对Bias的偏导数
        if b is not None:
            dB = np.sum(grad, axis=(0, 2, 3))  # 中文记法：等于loss对本层输出的梯度 对NHW维求和。

        # loss对W的偏导数。比较复杂，和前向传播时一样，再进行一次卷积核滑动。
        dW = np.zeros(w.shape, np.float32)   # [out_C, in_C, kH, kW]
        # loss对pad_x的偏导数
        dpad_x = np.zeros(pad_x.shape, np.float32)   # [N, C, H + padding*2, W + padding*2]
        # 卷积核滑动，只会在H和W两个方向上滑动
        for i in range(out_H):   # i是纵坐标
            for j in range(out_W):   # j是横坐标
                ori_x = j*stride   # 卷积核在pad_x中的横坐标，等差数列，公差是stride
                ori_y = i*stride   # 卷积核在pad_x中的纵坐标，等差数列，公差是stride
                part_x = pad_x[:, :, ori_y:ori_y+kH, ori_x:ori_x+kW]   # 截取卷积核所处的位置的像素 [N, in_C, kH, kW]

                exp_part_x = np.expand_dims(part_x, 1)   # 增加1维，[N, 1,     in_C, kH, kW]。

                # 获取loss对当前位置输出元素的偏导数dy。
                dy = grad[:, :, i:i+1, j:j+1]  # [N, out_C, 1, 1]  固定前2维为特定下标，就是课件里的标量dloss/dy_nkij了！
                dy = np.expand_dims(dy, 2)   # 增加1维，[N, out_C, 1, 1, 1]。

                # 中文记法：dy乘以当前卷积核覆盖的像素块，再累加，就是loss对W的偏导数了。过了面试的话不用谢我。
                temp = dy * exp_part_x   # [N, out_C, in_C, kH, kW]
                temp = np.sum(temp, axis=(0, ))  # [out_C, in_C, kH, kW]  多个样本共享了权重W，所以求和
                dW += temp      # “多个样本”共享了权重W，所以求和。现在，你是不是觉得我很聪明？哈哈

                # loss对输入x的偏导数
                exp_W = np.expand_dims(w, 0)         # [1, out_C, in_C, kH, kW]
                temp = dy * exp_W                    # [N, out_C, in_C, kH, kW]
                temp = np.sum(temp, axis=(1, ))      # [N, in_C, kH, kW]  全连接层中，是把偏移数量那一维求和掉，卷积层里也是一样，把偏移数量那一维求和掉。
                dpad_x[:, :, ori_y:ori_y+kH, ori_x:ori_x+kW] += temp    # [N, in_C, H + padding*2, W + padding*2]

        # 更新可训练参数
        if b is not None:
            b = optimizer.update(b, dB, self.b_lr, self.b_decay_type, self.b_decay)
            self.b = b
        w = optimizer.update(w, dW, self.w_lr, self.w_decay_type, self.w_decay)
        self.w = w
        # loss对输入x的偏导数，用来更新前面的层的权重
        dx = dpad_x[:, :, padding:padding + H, padding:padding + W]
        return dx



