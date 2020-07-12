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
                 epsilon=1e-05):
        super(BatchNorm, self).__init__()

        self.in_C = in_C
        self.momentum = momentum
        self.epsilon = epsilon

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
        # bn层推理时的前向传播和训练时的前向传播是不同的。用的是历史均值和历史方差进行归一化。aaaaaaaaaaaaaaaaaaaaaaaaaa
        return self.train_forward(x)

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
        # 求均值
        N, C, H, W = x.shape
        mean = np.mean(x, axis=(0, 2, 3))       # NHW维求均值，[C, ]。注意，这是当前批的均值。
        exp_mean = np.expand_dims(mean, 0)      # [1, C]
        exp_mean = np.expand_dims(exp_mean, 2)  # [1, C, 1]
        exp_mean = np.expand_dims(exp_mean, 3)  # [1, C, 1, 1]
        tile_exp_mean = np.tile(exp_mean, (N, 1, H, W))   # [N, C, H, W]，第0维重复N次，第2维重复H次，第3维重复W次，使得形状和x一样
        # 求方差
        var = (x - tile_exp_mean) ** 2          # 当前批的方差。
        var = np.mean(var, axis=(0, 2, 3))      # NHW维求均值，[C, ]。注意，这是当前批的方差。
        exp_var = np.expand_dims(var, 0)        # [1, C]
        exp_var = np.expand_dims(exp_var, 2)    # [1, C, 1]
        exp_var = np.expand_dims(exp_var, 3)    # [1, C, 1, 1]
        tile_exp_var = np.tile(exp_var, (N, 1, H, W))     # [N, C, H, W]，第0维重复N次，第2维重复H次，第3维重复W次，使得形状和x一样

        # gamma和beta同样扩展（重复）成x的形状
        exp_Scale = np.expand_dims(Scale, 0)      # [1, C]
        exp_Scale = np.expand_dims(exp_Scale, 2)  # [1, C, 1]
        exp_Scale = np.expand_dims(exp_Scale, 3)  # [1, C, 1, 1]
        tile_exp_Scale = np.tile(exp_Scale, (N, 1, H, W))   # [N, C, H, W]，第0维重复N次，第2维重复H次，第3维重复W次，使得形状和x一样
        exp_Bias = np.expand_dims(Bias, 0)        # [1, C]
        exp_Bias = np.expand_dims(exp_Bias, 2)    # [1, C, 1]
        exp_Bias = np.expand_dims(exp_Bias, 3)    # [1, C, 1, 1]
        tile_exp_Bias = np.tile(exp_Bias, (N, 1, H, W))     # [N, C, H, W]，第0维重复N次，第2维重复H次，第3维重复W次，使得形状和x一样

        # bn训练时前向传播的公式
        y = (x - tile_exp_mean) / ((tile_exp_var + epsilon)**0.5) * tile_exp_Scale + tile_exp_Bias

        # 更新历史均值和历史方差
        # MeanOut是历史均值，比重占0.9，mean是这一批的均值，比重是0.1
        MeanOut = MeanOut * momentum + mean * (1. - momentum)
        VarianceOut = VarianceOut * momentum + var * (1. - momentum)
        self.mean = MeanOut
        self.var = VarianceOut
        # 当前批的均值和方差保存一下，反向传播时用的。
        self.cur_mean = tile_exp_mean
        self.cur_var = tile_exp_var

        self.output = np.copy(y)   # 保存一下输出，反向传播时会使用到
        return y

    def train_backward(self, grad, lr):
        '''
        对本层的权重求偏导，以更新本层的权重。对本层的输入x求偏导，以更新前面的层的权重。
        设本层的权重是w，若loss = f(a, b, c, ...) = a(w)+b(w)+c(w)+...，那么loss对w的偏导数(偏导符号打不出来，用d表示了)
        dloss/dw = dloss/da * da/dw + dloss/db * db/dw + dloss/dc * dc/dw + ...
        a, b, c是中间变量，在这一层，self.output里的所有元素就是这个例子里的a, b, c, ...是中间变量，
        而dloss/da, dloss/db, dloss/dc, ...即grad里的所有元素。所以我们只需要求出
        da/dw, db/dw, dc/dw, ...就可以求出dloss/dw了！ 求loss对本层输入的偏导数，同理。

        参考课件上的公式（课件地址：xxxxxxxxx）：
        y_nchw = (x_nchw - u_c) / std_c * scale_c + bias_c
        (1) 对bias_c求偏导数
        y_nchw对bias_c求偏导，偏导符号打不出来，用d表示了：
        dy_nchw/dbias_c = 1.0
        注意到只有第1维下标是小c的偏导数才是1.0，所以
        dloss/dbias_c =  dloss/dy_0c00 * dy_0c00/dbias_c + dloss/dy_0c01 * dy_0c01/dbias_c + dloss/dy_0c02 * dy_0c02/dbias_c + ...   # 遍历0c0x
                      +  dloss/dy_0c10 * dy_0c10/dbias_c + dloss/dy_0c11 * dy_0c11/dbias_c + dloss/dy_0c12 * dy_0c12/dbias_c + ...   # 遍历0c1x
                      +  dloss/dy_0c20 * dy_0c20/dbias_c + dloss/dy_0c21 * dy_0c21/dbias_c + dloss/dy_0c22 * dy_0c22/dbias_c + ...   # 遍历0c2x
                      + ...
                      +  dloss/dy_1c00 * dy_1c00/dbias_c + dloss/dy_1c01 * dy_1c01/dbias_c + dloss/dy_1c02 * dy_1c02/dbias_c + ...   # 遍历1c0x
                      +  dloss/dy_1c10 * dy_1c10/dbias_c + dloss/dy_1c11 * dy_1c11/dbias_c + dloss/dy_1c12 * dy_1c12/dbias_c + ...   # 遍历1c1x
                      +  dloss/dy_1c20 * dy_1c20/dbias_c + dloss/dy_1c21 * dy_1c21/dbias_c + dloss/dy_1c22 * dy_1c22/dbias_c + ...   # 遍历1c2x
                      + ...
                      +  dloss/dy_2c00 * dy_2c00/dbias_c + dloss/dy_2c01 * dy_2c01/dbias_c + dloss/dy_2c02 * dy_2c02/dbias_c + ...   # 遍历2c0x
                      +  dloss/dy_2c10 * dy_2c10/dbias_c + dloss/dy_2c11 * dy_2c11/dbias_c + dloss/dy_2c12 * dy_2c12/dbias_c + ...   # 遍历2c1x
                      +  dloss/dy_2c20 * dy_2c20/dbias_c + dloss/dy_2c21 * dy_2c21/dbias_c + dloss/dy_2c22 * dy_2c22/dbias_c + ...   # 遍历2c2x
                      + ...                                                                                              # 遍历NHW 3个维，通道固定是小c

                      =  dloss/dy_0c00 *      1.0        + dloss/dy_0c01 *      1.0        + dloss/dy_0c02 *      1.0        + ...
                      +  dloss/dy_0c10 *      1.0        + dloss/dy_0c11 *      1.0        + dloss/dy_0c12 *      1.0        + ...
                      +  dloss/dy_0c20 *      1.0        + dloss/dy_0c21 *      1.0        + dloss/dy_0c22 *      1.0        + ...
                      + ...
                      +  dloss/dy_1c00 *      1.0        + dloss/dy_1c01 *      1.0        + dloss/dy_1c02 *      1.0        + ...
                      +  dloss/dy_1c10 *      1.0        + dloss/dy_1c11 *      1.0        + dloss/dy_1c12 *      1.0        + ...
                      +  dloss/dy_1c20 *      1.0        + dloss/dy_1c21 *      1.0        + dloss/dy_1c22 *      1.0        + ...
                      + ...
                      +  dloss/dy_2c00 *      1.0        + dloss/dy_2c01 *      1.0        + dloss/dy_2c02 *      1.0        + ...
                      +  dloss/dy_2c10 *      1.0        + dloss/dy_2c11 *      1.0        + dloss/dy_2c12 *      1.0        + ...
                      +  dloss/dy_2c20 *      1.0        + dloss/dy_2c21 *      1.0        + dloss/dy_2c22 *      1.0        + ...
                      + ...
                      = np.sum(grad, axis=(0, 2, 3)) [c]    # 观察表达式的规律，直接写出结果
        (2) 对scale_c求偏导数
        y_nchw = (x_nchw - u_c) / std_c * scale_c + bias_c
        y_nchw对scale_c求偏导，偏导符号打不出来，用d表示了：
        dy_nchw/dscale_c = (x_nchw - u_c) / std_c
        注意到只有第1维下标是小c的偏导数才是这个，所以
        dloss/dscale_c =  dloss/dy_0c00 * dy_0c00/dscale_c + dloss/dy_0c01 * dy_0c01/dscale_c + dloss/dy_0c02 * dy_0c02/dscale_c + ...   # 遍历0c0x
                       +  dloss/dy_0c10 * dy_0c10/dscale_c + dloss/dy_0c11 * dy_0c11/dscale_c + dloss/dy_0c12 * dy_0c12/dscale_c + ...   # 遍历0c1x
                       +  dloss/dy_0c20 * dy_0c20/dscale_c + dloss/dy_0c21 * dy_0c21/dscale_c + dloss/dy_0c22 * dy_0c22/dscale_c + ...   # 遍历0c2x
                       + ...
                       +  dloss/dy_1c00 * dy_1c00/dscale_c + dloss/dy_1c01 * dy_1c01/dscale_c + dloss/dy_1c02 * dy_1c02/dscale_c + ...   # 遍历1c0x
                       +  dloss/dy_1c10 * dy_1c10/dscale_c + dloss/dy_1c11 * dy_1c11/dscale_c + dloss/dy_1c12 * dy_1c12/dscale_c + ...   # 遍历1c1x
                       +  dloss/dy_1c20 * dy_1c20/dscale_c + dloss/dy_1c21 * dy_1c21/dscale_c + dloss/dy_1c22 * dy_1c22/dscale_c + ...   # 遍历1c2x
                       + ...
                       +  dloss/dy_2c00 * dy_2c00/dscale_c + dloss/dy_2c01 * dy_2c01/dscale_c + dloss/dy_2c02 * dy_2c02/dscale_c + ...   # 遍历2c0x
                       +  dloss/dy_2c10 * dy_2c10/dscale_c + dloss/dy_2c11 * dy_2c11/dscale_c + dloss/dy_2c12 * dy_2c12/dscale_c + ...   # 遍历2c1x
                       +  dloss/dy_2c20 * dy_2c20/dscale_c + dloss/dy_2c21 * dy_2c21/dscale_c + dloss/dy_2c22 * dy_2c22/dscale_c + ...   # 遍历2c2x
                       + ...                                                                                              # 遍历NHW 3个维，通道固定是小c
        即
        dloss/dscale_c =  dloss/dy_0c00 * (x_0c00-u_c)/std_c + dloss/dy_0c01 * (x_0c01-u_c)/std_c + dloss/dy_0c02 * (x_0c02-u_c)/std_c + ...   # 遍历0c0x
                       +  dloss/dy_0c10 * (x_0c10-u_c)/std_c + dloss/dy_0c11 * (x_0c11-u_c)/std_c + dloss/dy_0c12 * (x_0c12-u_c)/std_c + ...   # 遍历0c1x
                       +  dloss/dy_0c20 * (x_0c20-u_c)/std_c + dloss/dy_0c21 * (x_0c21-u_c)/std_c + dloss/dy_0c22 * (x_0c22-u_c)/std_c + ...   # 遍历0c2x
                       + ...
                       +  dloss/dy_1c00 * (x_1c00-u_c)/std_c + dloss/dy_1c01 * (x_1c01-u_c)/std_c + dloss/dy_1c02 * (x_1c02-u_c)/std_c + ...   # 遍历1c0x
                       +  dloss/dy_1c10 * (x_1c10-u_c)/std_c + dloss/dy_1c11 * (x_1c11-u_c)/std_c + dloss/dy_1c12 * (x_1c12-u_c)/std_c + ...   # 遍历1c1x
                       +  dloss/dy_1c20 * (x_1c20-u_c)/std_c + dloss/dy_1c21 * (x_1c21-u_c)/std_c + dloss/dy_1c22 * (x_1c22-u_c)/std_c + ...   # 遍历1c2x
                       + ...
                       +  dloss/dy_2c00 * (x_2c00-u_c)/std_c + dloss/dy_2c01 * (x_2c01-u_c)/std_c + dloss/dy_2c02 * (x_2c02-u_c)/std_c + ...   # 遍历2c0x
                       +  dloss/dy_2c10 * (x_2c10-u_c)/std_c + dloss/dy_2c11 * (x_2c11-u_c)/std_c + dloss/dy_2c12 * (x_2c12-u_c)/std_c + ...   # 遍历2c1x
                       +  dloss/dy_2c20 * (x_2c20-u_c)/std_c + dloss/dy_2c21 * (x_2c21-u_c)/std_c + dloss/dy_2c22 * (x_2c22-u_c)/std_c + ...   # 遍历2c2x
                       + ...                                                                                              # 遍历NHW 3个维，通道固定是小c
                       = np.sum(grad*(x-u)/std, axis=(0, 2, 3)) [c]    # 观察表达式的规律，直接写出结果。因为表达式里是先运算grad和(x-u)/std逐元素相乘（乘法优先于加法）。
        和高中时推导两个小球碰撞n次速度成等比数列相比，这些都算不了什么。。。
        (3) 对输入求偏导数，用来更新前面的层的权重
        y_nchw = (x_nchw - u_c) / std_c * scale_c + bias_c
        y_nchw对x_nchw求偏导，偏导符号打不出来，用d表示了：
        因为u_c和std_c里是包含x_nchw的，所以问题可没那么简单。
        dy_nchw/dx_nchw = scale_c * [(1 - du_c/dx_nchw)*std_c - (x_nchw - u_c)*dstd_c/dx_nchw] / (std_c^2)       # 分式的导数，还记得吗
        所以我们只要把du_c/dx_nchw和dstd_c/dx_nchw求出来，代入上面就能得到dy_nchw/dx_nchw了，也是用到了复合函数求导法则。
        均值公式u = (x1+x2+...+xn) / n,  方差公式var = (x1^2+x2^2+...+xn^2) / n - u ^ 2, 标准差std = (var+epsilon)**0.5
        所以du_c/dx_nchw = 1.0/(N*H*W),  dvar_c/dx_nchw = 2.0*x_nchw/(N*H*W) - 2*u_c*1.0/(N*H*W) = 2.0*(x_nchw-u_c)/(N*H*W)
        dstd_c/dx_nchw = 1.0 / [2.0*(var_c+epsilon)**0.5] * 2.0*(x_nchw-u_c)/(N*H*W) = (x_nchw-u_c)/[(N*H*W)*(var_c+epsilon)**0.5]
        所以
        dy_nchw/dx_nchw = scale_c * [  (1 - 1.0/(N*H*W))*std_c - (x_nchw - u_c)*(x_nchw-u_c)/[(N*H*W)*(var_c+epsilon)**0.5]   ] / (std_c^2)
        (用std_c = (var_c+epsilon)**0.5把std_c消去)
        = scale_c / (var_c+epsilon) * [  (1 - 1.0/(N*H*W))*(var_c+epsilon)**0.5 - (x_nchw-u_c)^2/[(N*H*W)*(var_c+epsilon)**0.5]   ]
        表达式就是长这样。
        所以（因为前面的Bias和Scale都只有C个，而这里的X是有N*C*H*W个的，所以不能写求和符号哦）
        dloss/dX = grad * Scale / (Var+epsilon) * [  (1 - 1.0/(N*H*W))*(Var+epsilon)**0.5 - (X-u)^2/[(N*H*W)*(Var+epsilon)**0.5]   ]
        注意到只有第1维下标是小c的偏导数才是这个，所以

        '''
        x, y = self.input, self.output   # 获取训练时前向传播的输入输出
        N, C, H, W = x.shape
        Scale = self.scale    # 可训练参数，用了与batch_norm_op.cc中相同的变量名
        Bias = self.offset    # 可训练参数，用了与batch_norm_op.cc中相同的变量名
        MeanOut = self.mean     # 历史均值，用了与batch_norm_op.cc中相同的变量名
        VarianceOut = self.var  # 历史方差，用了与batch_norm_op.cc中相同的变量名
        momentum = self.momentum
        epsilon = self.epsilon

        # loss对Bias的偏导数
        dBias = np.sum(grad, axis=(0, 2, 3))  # 中文记法：等于loss对本层输出的梯度 对NHW维求和。

        # 看飞桨的源码，看看标准差有没有涉及到epsilon？？？
        # loss对Scale的偏导数。这玩意就是bn层里loss对Scale的偏导数了，熟记。过了面试的话不用谢我。
        dScale = grad * (x-self.cur_mean) / ((self.cur_var + epsilon)**0.5)  # 中文记法：等于loss对本层输出的梯度 乘以 本层输入归一化后的值（用的是这一批的均值和方差进行归一化），再对NHW维求和。
        dScale = np.sum(dScale, axis=(0, 2, 3))


        # 看飞桨的源码，看看标准差有没有涉及到epsilon？？？
        # loss对输入x的偏导数，用来更新前面的层的权重
        exp_Scale = np.reshape(Scale, (1, -1, 1, 1))   # [1, C, 1, 1]
        dnormX = grad * exp_Scale                      # [N, C, H, W]
        dX = dnormX

        # 更新可训练参数
        Bias += -1.0 * lr * dBias  # 更新Bias
        Scale += -1.0 * lr * dScale  # 更新Scale
        self.offset = Bias
        self.scale = Scale
        return dX




