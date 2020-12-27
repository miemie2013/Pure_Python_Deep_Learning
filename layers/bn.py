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


class BatchNorm(Layer):
    def __init__(self,
                 in_C,
                 momentum=0.9,
                 epsilon=1e-05,
                 w_decay_type=None,
                 w_decay=0.,
                 b_decay_type=None,
                 b_decay=0.,
                 w_lr=1.,
                 b_lr=1.,
                 name=''):
        super(BatchNorm, self).__init__()

        self.in_C = in_C
        self.momentum = momentum
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
        self.mean = np.zeros((self.in_C, ), np.float32)    # 0初始化
        self.var = np.ones((self.in_C, ), np.float32)      # 1初始化

    def init_weights(self, scale, offset):
        self.scale = np.copy(scale)
        self.offset = np.copy(offset)

    def init_means_vars(self, mean, var):
        self.mean = np.copy(mean)
        self.var = np.copy(var)

    def test_forward(self, x):
        # bn层推理时的前向传播和训练时的前向传播是不同的。用的是历史均值和历史方差进行归一化。
        Scale = self.scale    # 可训练参数，用了与batch_norm_op.cc中相同的变量名
        Bias = self.offset    # 可训练参数，用了与batch_norm_op.cc中相同的变量名
        MeanOut = self.mean     # 历史均值，用了与batch_norm_op.cc中相同的变量名
        VarianceOut = self.var  # 历史方差，用了与batch_norm_op.cc中相同的变量名
        epsilon = self.epsilon

        input_shape = x.shape
        assert (len(input_shape) in [2, 4]), "The input of BatchNorm Layer must be 2-D Tensor or 4-D Tensor."
        if len(input_shape) == 4:   # 输入是4维张量。默认是NCHW格式。就变成2维张量处理。
            N, C, H, W = input_shape
            x = x.transpose(0, 2, 3, 1)   # NHWC格式。我们先把通道放到最后一维，这样reshape时才能精准拿到通道属性。
            x = np.reshape(x, (-1, C))
            M = N*H*W   # M是新的批大小
        elif len(input_shape) == 2:   # 输入是2维张量。
            M, C = input_shape

        # bn训练时前向传播的公式
        y = (x - MeanOut) / ((VarianceOut + epsilon)**0.5) * Scale + Bias

        if len(input_shape) == 4:   # 输入是4维张量。就变回去。
            y = np.reshape(y, (N, H, W, C))
            y = y.transpose(0, 3, 1, 2)   # NCHW格式
        return y

    def train_forward(self, x):
        '''
        bn层训练时的前向传播。变量的命名跟随了paddle bn层的源码paddle/fluid/operators/batch_norm_op.cc
        bn层训练时的前向传播，用的是当前批的均值和方差进行归一化，而不是历史均值和历史方差。
        '''
        self.input = np.copy(x)      # 保存一下输入，反向传播时会使用到
        Scale = self.scale    # 可训练参数，用了与batch_norm_op.cc中相同的变量名
        Bias = self.offset    # 可训练参数，用了与batch_norm_op.cc中相同的变量名
        MeanOut = self.mean     # 历史均值，用了与batch_norm_op.cc中相同的变量名
        VarianceOut = self.var  # 历史方差，用了与batch_norm_op.cc中相同的变量名
        momentum = self.momentum
        epsilon = self.epsilon

        input_shape = x.shape
        assert (len(input_shape) in [2, 4]), "The input of BatchNorm Layer must be 2-D Tensor or 4-D Tensor."
        if len(input_shape) == 4:   # 输入是4维张量。默认是NCHW格式。就变成2维张量处理。
            N, C, H, W = input_shape
            x = x.transpose(0, 2, 3, 1)   # NHWC格式。我们先把通道放到最后一维，这样reshape时才能精准拿到通道属性。
            x = np.reshape(x, (-1, C))
            M = N*H*W   # M是新的批大小
        elif len(input_shape) == 2:   # 输入是2维张量。
            M, C = input_shape
        # 求均值。其实不用变成tile_exp_mean也可以直接用于对x归一化，这里为了表现出重复。
        mean = np.mean(x, axis=(0, ))        # [C, ]  不同样本同一属性求均值。注意，这是当前批的均值。
        exp_mean = np.expand_dims(mean, 0)   # [1, C]
        tile_exp_mean = np.tile(exp_mean, (M, 1))   # [M, C]，第0维重复M次，使得形状和x一样
        # 求方差。其实不用变成tile_exp_var也可以直接用于对x归一化，这里为了表现出重复。
        var = np.var(x, axis=(0, ))          # [C, ]  不同样本同一属性求方差。注意，这是当前批的方差。
        exp_var = np.expand_dims(var, 0)     # [1, C]
        tile_exp_var = np.tile(exp_var, (M, 1))     # [M, C]，第0维重复M次，使得形状和x一样

        # gamma和beta同样扩展（重复）成x的形状
        exp_Scale = np.expand_dims(Scale, 0)          # [1, C]
        tile_exp_Scale = np.tile(exp_Scale, (M, 1))   # [M, C]，第0维重复M次，使得形状和x一样
        exp_Bias = np.expand_dims(Bias, 0)            # [1, C]
        tile_exp_Bias = np.tile(exp_Bias, (M, 1))     # [M, C]，第0维重复M次，使得形状和x一样

        # bn训练时前向传播的公式
        y = (x - tile_exp_mean) / ((tile_exp_var + epsilon)**0.5) * tile_exp_Scale + tile_exp_Bias
        if len(input_shape) == 4:   # 输入是4维张量。就变回去。
            y = np.reshape(y, (N, H, W, C))
            y = y.transpose(0, 3, 1, 2)   # NCHW格式

        # 更新历史均值和历史方差
        # MeanOut是历史均值，比重占0.9，mean是这一批的均值，比重是0.1
        MeanOut = MeanOut * momentum + mean * (1. - momentum)
        VarianceOut = VarianceOut * momentum + var * (1. - momentum)
        self.mean = MeanOut
        self.var = VarianceOut
        # 当前批的均值和方差保存一下，反向传播时用的。
        self.cur_tile_exp_mean = tile_exp_mean
        self.cur_tile_exp_var = tile_exp_var
        self.cur_mean = mean
        self.cur_var = var

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
        N, C, H, W = x.shape
        Scale = self.scale    # 可训练参数，用了与batch_norm_op.cc中相同的变量名
        Bias = self.offset    # 可训练参数，用了与batch_norm_op.cc中相同的变量名
        MeanOut = self.mean     # 历史均值，用了与batch_norm_op.cc中相同的变量名
        VarianceOut = self.var  # 历史方差，用了与batch_norm_op.cc中相同的变量名
        momentum = self.momentum
        epsilon = self.epsilon


        input_shape = x.shape
        assert (len(input_shape) in [2, 4]), "The input of BatchNorm Layer must be 2-D Tensor or 4-D Tensor."
        if len(input_shape) == 4:   # 输入是4维张量。默认是NCHW格式。就变成2维张量处理。
            N, C, H, W = input_shape
            x = x.transpose(0, 2, 3, 1)   # NHWC格式
            x = np.reshape(x, (-1, C))
            grad = grad.transpose(0, 2, 3, 1)   # NHWC格式。梯度也要变
            grad = np.reshape(grad, (-1, C))
            M = N*H*W   # M是新的批大小
        elif len(input_shape) == 2:   # 输入是2维张量。
            M, C = input_shape

        # loss对Bias的偏导数
        dBias = np.sum(grad, axis=(0, ))  # 中文记法：等于loss对本层输出的梯度 对NHW维求和。
        # loss对Scale的偏导数。很简单，就是dScale = grad * normX
        dScale = grad * (x-self.cur_tile_exp_mean) / ((self.cur_tile_exp_var + epsilon)**0.5)
        dScale = np.sum(dScale, axis=(0, ))


        # loss对输入x的偏导数，用来更新前面的层的权重。这是最难的一个，建议看 https://zhuanlan.zhihu.com/p/26138673
        dnormX = Scale * grad
        dVar = np.sum(dnormX * (x - self.cur_mean) * -0.5 * (self.cur_var + epsilon) ** -1.5, axis=0)
        dx_ = 1 / np.sqrt(self.cur_var + epsilon)
        dvar_ = 2 * (x - self.cur_mean) / M

        # intermediate for convenient calculation
        di = dnormX * dx_ + dVar * dvar_
        dmean = -1 * np.sum(di, axis=0)
        dmean_ = np.ones_like(x) / M
        dX = di + dmean * dmean_

        if len(input_shape) == 4:   # 输入是4维张量。就变回去。
            dX = np.reshape(dX, (N, H, W, C))
            dX = dX.transpose(0, 3, 1, 2)   # NCHW格式

        # 更新可训练参数
        Scale = optimizer.update(self.name+'_scale', Scale, dScale, self.w_lr, self.w_decay_type, self.w_decay)
        Bias = optimizer.update(self.name+'_offset', Bias, dBias, self.b_lr, self.b_decay_type, self.b_decay)
        self.offset = Bias
        self.scale = Scale
        return dX




