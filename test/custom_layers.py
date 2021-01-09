#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date:
#   Description :
#
# ================================================================
import paddle
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as L
import paddle.nn.functional as F
from paddle.fluid import Variable
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers import utils
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant, Normal
from paddle.fluid.regularizer import L2Decay




def deformable_conv(input,
                    offset,
                    mask,
                    num_filters,
                    filter_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=None,
                    deformable_groups=None,
                    im2col_step=None,
                    filter_param=None,
                    bias_attr=None,
                    modulated=True,
                    name=None):

    check_variable_and_dtype(input, "input", ['float32', 'float64'],
                             'deformable_conv')
    check_variable_and_dtype(offset, "offset", ['float32', 'float64'],
                             'deformable_conv')
    check_type(mask, 'mask', (Variable, type(None)), 'deformable_conv')

    num_channels = input.shape[1]
    assert filter_param is not None, "filter_param should not be None here."

    helper = LayerHelper('deformable_conv', **locals())
    dtype = helper.input_dtype()

    if not isinstance(input, Variable):
        raise TypeError("Input of deformable_conv must be Variable")
    if not isinstance(offset, Variable):
        raise TypeError("Input Offset of deformable_conv must be Variable")

    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        num_filter_channels = num_channels // groups

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    input_shape = input.shape
    filter_shape = [num_filters, int(num_filter_channels)] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * num_channels
        std = (2.0 / filter_elem_num)**0.5
        return Normal(0.0, std, 0)

    pre_bias = helper.create_variable_for_type_inference(dtype)

    if modulated:
        helper.append_op(
            type='deformable_conv',
            inputs={
                'Input': input,
                'Filter': filter_param,
                'Offset': offset,
                'Mask': mask,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            })

    else:
        helper.append_op(
            type='deformable_conv_v1',
            inputs={
                'Input': input,
                'Filter': filter_param,
                'Offset': offset,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            })

    output = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    return output



def get_norm(norm_type):
    bn = 0
    gn = 0
    af = 0
    cbn = 0
    if norm_type == 'bn':
        bn = 1
    elif norm_type == 'sync_bn':
        bn = 1
    elif norm_type == 'gn':
        gn = 1
    elif norm_type == 'affine_channel':
        af = 1
    elif norm_type == 'cbn':
        cbn = 1
    return bn, gn, af, cbn



class MyBN(paddle.nn.Layer):
    def __init__(self,
                 num_features,
                 momentum=0.9,
                 epsilon=1e-05,
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW',
                 name=None):
        super(MyBN, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        self.data_format = data_format
        self.name = name

        self.weight = fluid.layers.create_parameter(
            shape=[num_features, ],
            dtype='float32',
            attr=weight_attr,
            default_initializer=fluid.initializer.Constant(1.0))
        self.bias = fluid.layers.create_parameter(
            shape=[num_features, ],
            dtype='float32',
            attr=bias_attr,
            default_initializer=fluid.initializer.Constant(0.0))

        moving_mean_name = None
        moving_variance_name = None

        if name is not None:
            moving_mean_name = name + "_mean"
            moving_variance_name = name + "_variance"

        mattr = ParamAttr(
            name=moving_mean_name,
            initializer=fluid.initializer.Constant(0.0),
            trainable=False)
        vattr = ParamAttr(
            name=moving_variance_name,
            initializer=fluid.initializer.Constant(1.0),
            trainable=False)
        self._mean = fluid.layers.create_parameter(shape=[num_features, ], dtype='float32', attr=mattr)
        self._variance = fluid.layers.create_parameter(shape=[num_features, ], dtype='float32', attr=vattr)
        self.A = None
        self.B = None
        self.special_kernel = None

    def forward(self, x):
        if self.training:
            N, C, H, W = x.shape
            NHW = N*H*W

            # 方案一：用乘法
            # U = fluid.layers.reduce_mean(x, dim=[0, 2, 3], keep_dim=True)  # [1, C, 1, 1]
            # V = fluid.layers.reduce_mean(fluid.layers.square(x - U), dim=[0, 2, 3], keep_dim=True)  # [1, C, 1, 1]
            # normX = (x - U) / L.sqrt(V + self.epsilon)  # [N, C, H, W]
            # scale = L.unsqueeze(self.weight, [0, 2, 3])
            # bias = L.unsqueeze(self.bias, [0, 2, 3])
            # out = normX * scale + bias
            # U = L.reshape(U, (-1, ))
            # V = L.reshape(V, (-1, ))

            # 方案二：用分组卷积代替乘法
            # out = W*(x - U)/s + B     = (W/s) * x + B - (W/s)*U
            U = fluid.layers.reduce_mean(x, dim=[0, 2, 3], keep_dim=False)  # [C, ]
            if self.special_kernel is None:  # 为了快速求(x - U)
                special_kernel = np.ones((self.num_features, 1, 1, 1), np.float32)
                self.special_kernel = paddle.to_tensor(special_kernel)
                self.special_kernel.stop_gradient = True
            V = F.conv2d(x, self.special_kernel, -U, groups=self.num_features)  # 为了快速求(x - U)
            V = fluid.layers.reduce_mean(fluid.layers.square(V), dim=[0, 2, 3], keep_dim=False)  # [C, ]
            std = L.sqrt(V + self.epsilon)  # [C, ]
            A = self.weight / std  # [C, ]
            B = self.bias - U * A  # [C, ]
            A = L.unsqueeze(A, [1, 2, 3])  # [C, 1, 1, 1]
            out = F.conv2d(x, A, B, groups=self.num_features)

            curr_U = U.numpy()
            curr_V = V.numpy()
            state_dict = self.state_dict()
            momentum = self.momentum
            _mean = self._mean.numpy() * momentum + curr_U * (1. - momentum)
            _variance = self._variance.numpy() * momentum + curr_V * (1. - momentum)
            state_dict['_mean'] = _mean.astype(np.float32)
            state_dict['_variance'] = _variance.astype(np.float32)
            self.set_state_dict(state_dict)
            self.A = None
            self.B = None
        else:
            # 方案一：用乘法
            # U = L.unsqueeze(self._mean, [0, 2, 3])  # [1, C, 1, 1]
            # V = L.unsqueeze(self._variance, [0, 2, 3])  # [1, C, 1, 1]
            # normX = (x - U) / L.sqrt(V + self.epsilon)  # [N, C, H, W]
            # scale = L.unsqueeze(self.weight, [0, 2, 3])
            # bias = L.unsqueeze(self.bias, [0, 2, 3])
            # out = normX * scale + bias

            # 方案二：用分组卷积代替乘法
            # out = W*(x - U)/s + B     = (W/s) * x + B - (W/s)*U
            if self.A is None:
                std = L.sqrt(self._variance + self.epsilon)  # [C, ]
                A = self.weight / std  # [C, ]
                B = self.bias - self._mean * A  # [C, ]
                A = L.unsqueeze(A, [1, 2, 3])  # [C, 1, 1, 1]
                self.A = A
                self.B = B
            out = F.conv2d(x, self.A, self.B, groups=self.num_features)
        return out


class conv2d_grad(object):
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 padding=0):
        super(conv2d_grad, self).__init__()
        self.input_dim = input_dim
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.dx_pos = None
        self.special_mask = None
        self.max_len = 0
        self.special_inds = None
        self.special_inds_dw = None

    def set(self, x, y):
        self.x = x
        self.y = y

    def get_grad_w(self, w, b, grad):
        conv_in = self.x
        conv_out = self.y
        N, C, H, W = conv_in.shape
        N, out_C, out_H, out_W = conv_out.shape
        # w  [out_C, in_C, kH, kW]
        out_C, in_C, kH, kW = w.shape
        stride = self.stride
        padding = self.padding
        pad_H = H + padding * 2
        pad_W = W + padding * 2

        # loss对w的偏导数。
        conv_in = paddle.to_tensor(conv_in)
        pad_x = L.pad(conv_in, paddings=[0, 0, 0, 0, padding, padding, padding, padding], pad_value=0.0)  # [N, in_C, pad_H, pad_W]
        pad_x = L.transpose(pad_x, [2, 3, 0, 1])  # [pad_H, pad_W, N, in_C]
        if self.special_inds_dw is None:  # 只会做一次，即初始化。
            self.special_inds_dw = []
            # 卷积核滑动，只会在H和W两个方向上滑动
            for i in range(out_H):  # i是纵坐标
                for j in range(out_W):  # j是横坐标
                    ori_x = j * stride  # 卷积核在pad_x中的横坐标，等差数列，公差是stride
                    ori_y = i * stride  # 卷积核在pad_x中的纵坐标，等差数列，公差是stride
                    for i2 in range(kH):  # i2是纵坐标
                        for j2 in range(kW):  # j2是横坐标
                            point_x = ori_x + j2
                            point_y = ori_y + i2
                            self.special_inds_dw.append([point_y, point_x])
            # self.special_inds_dw.shape == [out_H*out_W*kH*kW, 2]
        special_inds_dw = paddle.to_tensor(self.special_inds_dw)
        special_inds_dw = L.cast(special_inds_dw, 'int32')
        special_inds_dw.stop_gradient = True
        x_in = L.gather_nd(pad_x, special_inds_dw)  # [out_H*out_W*kH*kW, N, in_C]
        x_in = L.reshape(x_in, (out_H, out_W, kH, kW, N, in_C))
        x_in = L.transpose(x_in, [4, 5, 0, 1, 2, 3])                 # [N, in_C, out_H, out_W, kH, kW]
        x_in = L.reshape(x_in, (N, in_C, out_H*out_W, kH, kW))       # [N, in_C, out_H*out_W, kH, kW]
        x_in = L.unsqueeze(x_in, 1)                                  # [N, 1, in_C, out_H*out_W, kH, kW]
        grad_r = L.reshape(grad, (N, out_C, 1, out_H*out_W, 1, 1))   # [N, out_C, 1, out_H*out_W, 1, 1]
        dw = x_in * grad_r                                           # [N, out_C, in_C, out_H*out_W, kH, kW]
        dL_dWeight = L.reduce_sum(dw, dim=[0, 3])                    # [out_C, in_C, kH, kW]
        return dL_dWeight



class CBatchNorm2D(paddle.nn.Layer):
    def __init__(self, num_features, weight_attr, bias_attr, eps=1e-5, momentum=0.9, affine=True,
                 track_running_stats=True,
                 buffer_num=0, rho=1.0,
                 burnin=0, two_stage=True,
                 FROZEN=False, out_p=False, name=None):
        super(CBatchNorm2D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.buffer_num = buffer_num
        self.max_buffer_num = buffer_num
        self.rho = rho
        self.burnin = burnin
        self.two_stage = two_stage
        self.FROZEN = FROZEN
        self.out_p = out_p

        self.iter_count = 0
        self.pre_mu = []
        self.pre_meanx2 = []  # mean(x^2)
        self.pre_dmudw = []
        self.pre_dmeanx2dw = []
        self.pre_weight = []
        self.special_kernel = None

        self.weight = fluid.layers.create_parameter(
            shape=[num_features, ],
            dtype='float32',
            attr=weight_attr,
            default_initializer=fluid.initializer.Constant(1.0))
        self.bias = fluid.layers.create_parameter(
            shape=[num_features, ],
            dtype='float32',
            attr=bias_attr,
            default_initializer=fluid.initializer.Constant(0.0))

        if not self.affine:
            self.weight.stop_gradient = True
            self.bias.stop_gradient = True

        moving_mean_name = None
        moving_variance_name = None

        if name is not None:
            moving_mean_name = name + "_mean"
            moving_variance_name = name + "_variance"

        mattr = ParamAttr(
            name=moving_mean_name,
            initializer=fluid.initializer.Constant(0.0),
            trainable=False)
        vattr = ParamAttr(
            name=moving_variance_name,
            initializer=fluid.initializer.Constant(1.0),
            trainable=False)
        self._mean = fluid.layers.create_parameter(shape=[num_features, ], dtype='float32', attr=mattr)
        self._variance = fluid.layers.create_parameter(shape=[num_features, ], dtype='float32', attr=vattr)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def _update_buffer_num(self):
        if self.two_stage:
            if self.iter_count > self.burnin:
                self.buffer_num = self.max_buffer_num
            else:
                self.buffer_num = 0
        else:
            self.buffer_num = int(self.max_buffer_num * min(self.iter_count / self.burnin, 1.0))

    def forward(self, input, conv, conv_g):
        # deal with wight and grad of self.pre_dxdw!
        self._check_input_dim(input)
        N, C, H, W = input.shape
        NHW = N*H*W
        y = input   # [N, C, H, W]
        weight = conv.weight

        # burnin
        if self.training and self.burnin > 0:
            self.iter_count += 1
            self._update_buffer_num()

        if self.buffer_num > 0 and self.training and (not input.stop_gradient):  # some layers are frozen!
            # cal current batch mu and sigma
            cur_mu = L.reduce_mean(y, dim=[0, 2, 3], keep_dim=False)  # [C, ]
            if self.special_kernel is None:  # 为了快速求(x - cur_mu)
                special_kernel = np.ones((self.num_features, 1, 1, 1), np.float32)
                self.special_kernel = paddle.to_tensor(special_kernel)
                self.special_kernel.stop_gradient = True
            cur_sigma2 = F.conv2d(y, self.special_kernel, -cur_mu, groups=self.num_features)  # 为了快速求(x - cur_mu)
            cur_sigma2 = L.reduce_sum(L.square(cur_sigma2), dim=[0, 2, 3], keep_dim=False) / (NHW-1)  # [C, ]  作者原版实现中使用的是样本方差，所以分母-1

            y2 = L.square(y)
            cur_meanx2 = L.reduce_mean(y2, dim=[0, 2, 3], keep_dim=False)  # [C, ]

            # cal dmu/dw dsigma2/dw
            # dmudw = paddle.grad(outputs=[cur_mu], inputs=[weight], create_graph=False, retain_graph=True)[0]
            # dmeanx2dw = paddle.grad(outputs=[cur_meanx2], inputs=[weight], create_graph=False, retain_graph=True)[0]

            # 自己的求法
            dmudinput = np.zeros(input.shape, np.float32) + 1.0 / NHW
            dmudinput = paddle.to_tensor(dmudinput)
            dmeanx2dinput = input.numpy()
            dmeanx2dinput = paddle.to_tensor(dmeanx2dinput)
            dmeanx2dinput *= 2.0 / NHW
            dmudw = conv_g.get_grad_w(conv.weight, conv.bias, dmudinput)
            dmeanx2dw = conv_g.get_grad_w(conv.weight, conv.bias, dmeanx2dinput)

            # update cur_mu and cur_sigma2 with pres
            weight_data = weight.numpy()
            weight_data = paddle.to_tensor(weight_data)
            weight_data.stop_gradient = True
            # 如果用L.stack()会报错，所以用L.concat()代替。
            mu_all = [cur_mu, ] + [tmp_mu + L.reduce_sum(self.rho * tmp_d * (weight_data - tmp_w), dim=[1, 2, 3]) for
                                   tmp_mu, tmp_d, tmp_w in zip(self.pre_mu, self.pre_dmudw, self.pre_weight)]
            meanx2_all = [cur_meanx2, ] + [tmp_meanx2 + L.reduce_sum(self.rho * tmp_d * (weight_data - tmp_w), dim=[1, 2, 3]) for
                                           tmp_meanx2, tmp_d, tmp_w in zip(self.pre_meanx2, self.pre_dmeanx2dw, self.pre_weight)]
            mu_all = [L.unsqueeze(mu_, 0) for mu_ in mu_all]
            meanx2_all = [L.unsqueeze(meanx2_, 0) for meanx2_ in meanx2_all]
            mu_all = L.concat(mu_all, 0)
            meanx2_all = L.concat(meanx2_all, 0)

            sigma2_all = meanx2_all - L.square(mu_all)

            # with considering count
            re_mu_all = mu_all.clone()
            re_meanx2_all = meanx2_all.clone()
            mask1 = L.cast(sigma2_all >= 0., dtype="float32")
            mask1.stop_gradient = True
            re_mu_all *= mask1
            re_meanx2_all *= mask1
            count = L.reduce_sum(L.cast(sigma2_all >= 0., dtype="float32"), dim=[0, ])
            mu = L.reduce_sum(re_mu_all, dim=[0, ]) / count
            sigma2 = L.reduce_sum(re_meanx2_all, dim=[0, ]) / count - L.square(mu)


            cur_mu_ = cur_mu.numpy()
            cur_mu_ = paddle.to_tensor(cur_mu_)
            cur_mu_.stop_gradient = True
            self.pre_mu = [cur_mu_, ] + self.pre_mu[:(self.buffer_num - 1)]
            cur_meanx2_ = cur_meanx2.numpy()
            cur_meanx2_ = paddle.to_tensor(cur_meanx2_)
            cur_meanx2_.stop_gradient = True
            self.pre_meanx2 = [cur_meanx2_, ] + self.pre_meanx2[:(self.buffer_num - 1)]
            dmudw_ = dmudw.numpy()
            dmudw_ = paddle.to_tensor(dmudw_)
            dmudw_.stop_gradient = True
            self.pre_dmudw = [dmudw_, ] + self.pre_dmudw[:(self.buffer_num - 1)]
            dmeanx2dw_ = dmeanx2dw.numpy()
            dmeanx2dw_ = paddle.to_tensor(dmeanx2dw_)
            dmeanx2dw_.stop_gradient = True
            self.pre_dmeanx2dw = [dmeanx2dw_, ] + self.pre_dmeanx2dw[:(self.buffer_num - 1)]

            tmp_weight = weight.numpy()
            tmp_weight = paddle.to_tensor(tmp_weight)
            tmp_weight.stop_gradient = True
            self.pre_weight = [tmp_weight, ] + self.pre_weight[:(self.buffer_num - 1)]

        else:
            mu = L.reduce_mean(y, dim=[0, 2, 3], keep_dim=False)  # [C, ]
            if self.special_kernel is None:  # 为了快速求(x - mu)
                special_kernel = np.ones((self.num_features, 1, 1, 1), np.float32)
                self.special_kernel = paddle.to_tensor(special_kernel)
                self.special_kernel.stop_gradient = True
            sigma2 = F.conv2d(y, self.special_kernel, -mu, groups=self.num_features)  # 为了快速求(x - mu)
            sigma2 = L.reduce_sum(L.square(sigma2), dim=[0, 2, 3], keep_dim=False) / (NHW-1)  # [C, ]
            cur_mu = mu
            cur_sigma2 = sigma2

        if not self.training or self.FROZEN:   # eval()状态
            U = self._mean
            # TODO: outside **0.5?
            if self.out_p:
                std = L.sqrt(self._variance + self.eps)
            else:
                std = L.sqrt(self._variance) + self.eps

        else:   # train()状态
            if self.track_running_stats is True:
                state_dict = self.state_dict()
                momentum = self.momentum
                _mean = self._mean.numpy() * momentum + cur_mu.numpy() * (1. - momentum)
                _variance = self._variance.numpy() * momentum + cur_sigma2.numpy() * (1. - momentum)
                state_dict['_mean'] = _mean.astype(np.float32)
                state_dict['_variance'] = _variance.astype(np.float32)
                self.set_state_dict(state_dict)
            U = mu
            # TODO: outside **0.5?
            if self.out_p:
                std = L.sqrt(sigma2 + self.eps)
            else:
                std = L.sqrt(sigma2) + self.eps

        A = self.weight / std  # [C, ]
        B = self.bias - U * A  # [C, ]
        A = L.unsqueeze(A, [1, 2, 3])  # [C, 1, 1, 1]
        y = F.conv2d(y, A, B, groups=self.num_features)
        return y

    def forward2(self, input, weight):
        # deal with wight and grad of self.pre_dxdw!
        self._check_input_dim(input)
        y = L.transpose(input, [1, 0, 2, 3])   # [C, N, H, W]
        return_shape = y.shape                 # [C, N, H, W]
        C, N, H, W = return_shape
        NHW = N*H*W
        y = L.reshape(y, (return_shape[0], -1))   # [C, N*H*W]

        # burnin
        if self.training and self.burnin > 0:
            self.iter_count += 1
            self._update_buffer_num()

        if self.buffer_num > 0 and self.training and (not input.stop_gradient):  # some layers are frozen!
            # cal current batch mu and sigma
            _cur_mu = L.reduce_mean(y, dim=[1, ], keep_dim=True)  # [C, 1]
            _cur_sigma2 = L.reduce_sum(L.square(y - _cur_mu), dim=[1, ], keep_dim=True) / (NHW-1)  # [C, 1]  作者原版实现中使用的是样本方差，所以分母-1
            cur_mu = L.reshape(_cur_mu, (-1, ))  # [C, ]
            cur_sigma2 = L.reshape(_cur_sigma2, (-1, ))  # [C, ]
            y2 = L.square(y)
            cur_meanx2 = L.reduce_mean(y2, dim=[1, ], keep_dim=False)  # [C, ]
            # cal dmu/dw dsigma2/dw
            dmudw = paddle.grad(outputs=[cur_mu], inputs=[weight], create_graph=False, retain_graph=True)[0]
            dmeanx2dw = paddle.grad(outputs=[cur_meanx2], inputs=[weight], create_graph=False, retain_graph=True)[0]

            # update cur_mu and cur_sigma2 with pres
            weight_data = weight.numpy()
            weight_data = paddle.to_tensor(weight_data)
            weight_data.stop_gradient = True
            # 如果用L.stack()会报错，所以用L.concat()代替。
            mu_all = [cur_mu, ] + [tmp_mu + L.reduce_sum(self.rho * tmp_d * (weight_data - tmp_w), dim=[1, 2, 3]) for
                                   tmp_mu, tmp_d, tmp_w in zip(self.pre_mu, self.pre_dmudw, self.pre_weight)]
            meanx2_all = [cur_meanx2, ] + [tmp_meanx2 + L.reduce_sum(self.rho * tmp_d * (weight_data - tmp_w), dim=[1, 2, 3]) for
                                           tmp_meanx2, tmp_d, tmp_w in zip(self.pre_meanx2, self.pre_dmeanx2dw, self.pre_weight)]
            mu_all = [L.unsqueeze(mu_, 0) for mu_ in mu_all]
            meanx2_all = [L.unsqueeze(meanx2_, 0) for meanx2_ in meanx2_all]
            mu_all = L.concat(mu_all, 0)
            meanx2_all = L.concat(meanx2_all, 0)

            sigma2_all = meanx2_all - L.square(mu_all)

            # with considering count
            re_mu_all = mu_all.clone()
            re_meanx2_all = meanx2_all.clone()
            mask1 = L.cast(sigma2_all >= 0., dtype="float32")
            mask1.stop_gradient = True
            re_mu_all *= mask1
            re_meanx2_all *= mask1
            count = L.reduce_sum(L.cast(sigma2_all >= 0., dtype="float32"), dim=[0, ])
            mu = L.reduce_sum(re_mu_all, dim=[0, ]) / count
            sigma2 = L.reduce_sum(re_meanx2_all, dim=[0, ]) / count - L.square(mu)


            cur_mu_ = cur_mu.numpy()
            cur_mu_ = paddle.to_tensor(cur_mu_)
            cur_mu_.stop_gradient = True
            self.pre_mu = [cur_mu_, ] + self.pre_mu[:(self.buffer_num - 1)]
            cur_meanx2_ = cur_meanx2.numpy()
            cur_meanx2_ = paddle.to_tensor(cur_meanx2_)
            cur_meanx2_.stop_gradient = True
            self.pre_meanx2 = [cur_meanx2_, ] + self.pre_meanx2[:(self.buffer_num - 1)]
            dmudw_ = dmudw.numpy()
            dmudw_ = paddle.to_tensor(dmudw_)
            dmudw_.stop_gradient = True
            self.pre_dmudw = [dmudw_, ] + self.pre_dmudw[:(self.buffer_num - 1)]
            dmeanx2dw_ = dmeanx2dw.numpy()
            dmeanx2dw_ = paddle.to_tensor(dmeanx2dw_)
            dmeanx2dw_.stop_gradient = True
            self.pre_dmeanx2dw = [dmeanx2dw_, ] + self.pre_dmeanx2dw[:(self.buffer_num - 1)]

            tmp_weight = weight.numpy()
            tmp_weight = paddle.to_tensor(tmp_weight)
            tmp_weight.stop_gradient = True
            self.pre_weight = [tmp_weight, ] + self.pre_weight[:(self.buffer_num - 1)]

        else:
            x = y   # [C, N*H*W]
            mu = L.reduce_mean(x, dim=[1, ], keep_dim=True)  # [C, 1]
            sigma2 = L.reduce_sum(L.square(x - mu), dim=[1, ], keep_dim=True) / (NHW-1)  # [C, 1]  作者原版实现中使用的是样本方差，所以分母-1
            mu = L.reshape(mu, (-1, ))  # [C, ]
            sigma2 = L.reshape(sigma2, (-1, ))  # [C, ]
            cur_mu = mu
            cur_sigma2 = sigma2

        if not self.training or self.FROZEN:
            y = y - L.reshape(self._mean, (-1, 1))
            # TODO: outside **0.5?
            if self.out_p:
                y = y / (L.reshape(self._variance, (-1, 1)) + self.eps) ** .5
            else:
                y = y / (L.reshape(self._variance, (-1, 1)) ** .5 + self.eps)

        else:
            if self.track_running_stats is True:
                state_dict = self.state_dict()
                momentum = self.momentum
                _mean = self._mean.numpy() * momentum + cur_mu.numpy() * (1. - momentum)
                _variance = self._variance.numpy() * momentum + cur_sigma2.numpy() * (1. - momentum)
                state_dict['_mean'] = _mean.astype(np.float32)
                state_dict['_variance'] = _variance.astype(np.float32)
                self.set_state_dict(state_dict)
            y = y - L.reshape(mu, (-1, 1))   # [C, N*H*W]
            # TODO: outside **0.5?
            if self.out_p:
                y = y / (L.reshape(sigma2, (-1, 1)) + self.eps) ** .5
            else:
                y = y / (L.reshape(sigma2, (-1, 1)) ** .5 + self.eps)

        y = L.reshape(self.weight, (-1, 1)) * y + L.reshape(self.bias, (-1, 1))
        y = L.reshape(y, return_shape)
        y = L.transpose(y, [1, 0, 2, 3])   # [N, C, H, W]
        return y







class Conv2dUnit(paddle.nn.Layer):
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 bias_attr=False,
                 norm_type=None,
                 groups=32,
                 act=None,
                 freeze_norm=False,
                 is_test=False,
                 norm_decay=0.,
                 lr=1.,
                 bias_lr=None,
                 weight_init=None,
                 bias_init=None,
                 use_dcn=False,
                 name=''):
        super(Conv2dUnit, self).__init__()
        self.groups = groups
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = (filter_size - 1) // 2
        self.act = act
        self.freeze_norm = freeze_norm
        self.is_test = is_test
        self.norm_decay = norm_decay
        self.use_dcn = use_dcn
        self.name = name

        # conv
        conv_name = name
        self.dcn_param = None
        if use_dcn:
            self.conv = paddle.nn.Conv2D(input_dim,
                                         filter_size * filter_size * 3,
                                         kernel_size=filter_size,
                                         stride=stride,
                                         padding=self.padding,
                                         weight_attr=ParamAttr(initializer=Constant(0.0), name=conv_name + "_conv_offset.w_0"),
                                         bias_attr=ParamAttr(initializer=Constant(0.0), name=conv_name + "_conv_offset.b_0"))
            self.dcn_param = fluid.layers.create_parameter(
                shape=[filters, input_dim, filter_size, filter_size],
                dtype='float32',
                attr=ParamAttr(name=conv_name + "_dcn_weights", learning_rate=lr, initializer=weight_init),
                default_initializer=fluid.initializer.Xavier())
        else:
            conv_battr = False
            if bias_attr:
                blr = lr
                if bias_lr:
                    blr = bias_lr
                conv_battr = ParamAttr(name=conv_name + "_bias",
                                       learning_rate=blr,
                                       initializer=bias_init,
                                       regularizer=L2Decay(0.))   # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            self.conv = paddle.nn.Conv2D(input_dim,
                                         filters,
                                         kernel_size=filter_size,
                                         stride=stride,
                                         padding=self.padding,
                                         weight_attr=ParamAttr(name=conv_name + "_weights", learning_rate=lr, initializer=weight_init),
                                         bias_attr=conv_battr)


        # norm
        assert norm_type in [None, 'cbn', 'bn', 'sync_bn', 'gn', 'affine_channel']
        bn, gn, af, cbn = get_norm(norm_type)
        if conv_name == "conv1":
            norm_name = "bn_" + conv_name
            if gn:
                norm_name = "gn_" + conv_name
            if af:
                norm_name = "af_" + conv_name
        else:
            norm_name = "bn" + conv_name[3:]
            if gn:
                norm_name = "gn" + conv_name[3:]
            if af:
                norm_name = "af" + conv_name[3:]
        norm_lr = 0. if freeze_norm else lr
        pattr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),   # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            name=norm_name + "_scale",
            trainable=False if freeze_norm else True)
        battr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay),   # 不可以加正则化的参数：norm层(比如bn层、affine_channel层、gn层)的scale、offset；卷积层的偏移参数。
            name=norm_name + "_offset",
            trainable=False if freeze_norm else True)
        self.bn = None
        self.gn = None
        self.af = None
        self.cbn = None
        if bn:
            # self.bn = paddle.nn.BatchNorm2D(filters, weight_attr=pattr, bias_attr=battr)
            self.bn = MyBN(filters, weight_attr=pattr, bias_attr=battr)
        if gn:
            self.gn = paddle.nn.GroupNorm(num_groups=groups, num_channels=filters, weight_attr=pattr, bias_attr=battr)
        if af:
            self.af = True
            self.scale = fluid.layers.create_parameter(
                shape=[filters],
                dtype='float32',
                attr=pattr,
                default_initializer=Constant(1.))
            self.offset = fluid.layers.create_parameter(
                shape=[filters],
                dtype='float32',
                attr=battr,
                default_initializer=Constant(0.))
        if cbn:
            self.cbn = CBatchNorm2D(filters, weight_attr=pattr, bias_attr=battr, buffer_num=3, burnin=8, out_p=True)
            # ==================== grad ====================
            self.temp_x = None
            self.conv_g = conv2d_grad(input_dim, filters, filter_size, stride, self.padding)

        # act
        self.act = None
        if act == 'relu':
            self.act = paddle.nn.ReLU()
        elif act == 'leaky':
            self.act = paddle.nn.LeakyReLU(0.1)


    def freeze(self):
        if self.conv is not None:
            if self.conv.weight is not None:
                self.conv.weight.stop_gradient = True
            if self.conv.bias is not None:
                self.conv.bias.stop_gradient = True
        if self.dcn_param is not None:
            self.dcn_param.stop_gradient = True
        if self.bn is not None:
            self.bn.weight.stop_gradient = True
            self.bn.bias.stop_gradient = True
        if self.gn is not None:
            self.gn.weight.stop_gradient = True
            self.gn.bias.stop_gradient = True
        if self.af is not None:
            self.scale.stop_gradient = True
            self.offset.stop_gradient = True

    def forward(self, x):
        if self.use_dcn:
            offset_mask = self.conv(x)
            offset = offset_mask[:, :self.filter_size**2 * 2, :, :]
            mask = offset_mask[:, self.filter_size**2 * 2:, :, :]
            mask = fluid.layers.sigmoid(mask)
            conv_out = deformable_conv(input=x, offset=offset, mask=mask,
                                num_filters=self.filters,
                                filter_size=self.filter_size,
                                stride=self.stride,
                                padding=self.padding,
                                groups=1,
                                deformable_groups=1,
                                im2col_step=1,
                                filter_param=self.dcn_param,
                                bias_attr=False)
        else:
            conv_out = self.conv(x)
            if self.cbn:
                self.conv_g.set(x.numpy(), conv_out.numpy())
        if self.bn:
            norm_out = self.bn(conv_out)
        elif self.gn:
            norm_out = self.gn(conv_out)
        elif self.af:
            norm_out = fluid.layers.affine_channel(conv_out, scale=self.scale, bias=self.offset, act=None)
        elif self.cbn:
            if self.use_dcn:
                norm_out = self.cbn(conv_out, self.dcn_param)
            else:
                norm_out = self.cbn(conv_out, self.conv, self.conv_g)
        else:
            norm_out = conv_out
        if self.act:
            act_out = self.act(norm_out)
        else:
            act_out = norm_out
        return conv_out, norm_out, act_out


class CoordConv(paddle.nn.Layer):
    def __init__(self, coord_conv=True):
        super(CoordConv, self).__init__()
        self.coord_conv = coord_conv

    def __call__(self, input):
        if not self.coord_conv:
            return input
        b = input.shape[0]
        h = input.shape[2]
        w = input.shape[3]
        x_range = L.range(0, w, 1., dtype='float32') / (w - 1) * 2.0 - 1
        y_range = L.range(0, h, 1., dtype='float32') / (h - 1) * 2.0 - 1
        # x_range = paddle.to_tensor(x_range, place=input.place)
        # y_range = paddle.to_tensor(y_range, place=input.place)
        x_range = L.reshape(x_range, (1, 1, 1, -1))  # [1, 1, 1, w]
        y_range = L.reshape(y_range, (1, 1, -1, 1))  # [1, 1, h, 1]
        x_range = L.expand(x_range, [b, 1, h, 1])  # [b, 1, h, w]
        y_range = L.expand(y_range, [b, 1, 1, w])  # [b, 1, h, w]
        offset = L.concat([input, x_range, y_range], axis=1)
        return offset


class SPP(paddle.nn.Layer):
    def __init__(self, seq='asc'):
        super(SPP, self).__init__()
        assert seq in ['desc', 'asc']
        self.seq = seq
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=5, stride=1, padding=2)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=9, stride=1, padding=4)
        self.max_pool3 = paddle.nn.MaxPool2D(kernel_size=13, stride=1, padding=6)

    def __call__(self, x):
        x_1 = x
        x_2 = self.max_pool1(x)
        x_3 = self.max_pool2(x)
        x_4 = self.max_pool3(x)
        if self.seq == 'desc':
            out = L.concat([x_4, x_3, x_2, x_1], axis=1)
        else:
            out = L.concat([x_1, x_2, x_3, x_4], axis=1)
        return out


class DropBlock(paddle.nn.Layer):
    def __init__(self,
                 block_size=3,
                 keep_prob=0.9,
                 is_test=False):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.is_test = is_test

    def __call__(self, input):
        if self.is_test:
            return input

        def CalculateGamma(input, block_size, keep_prob):
            input_shape = fluid.layers.shape(input)
            feat_shape_tmp = fluid.layers.slice(input_shape, [0], [3], [4])
            feat_shape_tmp = fluid.layers.cast(feat_shape_tmp, dtype="float32")
            feat_shape_t = fluid.layers.reshape(feat_shape_tmp, [1, 1, 1, 1])
            feat_area = fluid.layers.pow(feat_shape_t, factor=2)

            block_shape_t = fluid.layers.fill_constant(
                shape=[1, 1, 1, 1], value=block_size, dtype='float32')
            block_area = fluid.layers.pow(block_shape_t, factor=2)

            useful_shape_t = feat_shape_t - block_shape_t + 1
            useful_area = fluid.layers.pow(useful_shape_t, factor=2)

            upper_t = feat_area * (1 - keep_prob)
            bottom_t = block_area * useful_area
            output = upper_t / bottom_t
            return output

        gamma = CalculateGamma(input, block_size=self.block_size, keep_prob=self.keep_prob)
        input_shape = fluid.layers.shape(input)
        p = fluid.layers.expand_as(gamma, input)

        input_shape_tmp = fluid.layers.cast(input_shape, dtype="int64")
        random_matrix = fluid.layers.uniform_random(
            input_shape_tmp, dtype='float32', min=0.0, max=1.0)
        one_zero_m = fluid.layers.less_than(random_matrix, p)
        one_zero_m.stop_gradient = True
        one_zero_m = fluid.layers.cast(one_zero_m, dtype="float32")

        mask_flag = fluid.layers.pool2d(
            one_zero_m,
            pool_size=self.block_size,
            pool_type='max',
            pool_stride=1,
            pool_padding=self.block_size // 2)
        mask = 1.0 - mask_flag

        elem_numel = fluid.layers.reduce_prod(input_shape)
        elem_numel_m = fluid.layers.cast(elem_numel, dtype="float32")
        elem_numel_m.stop_gradient = True

        elem_sum = fluid.layers.reduce_sum(mask)
        elem_sum_m = fluid.layers.cast(elem_sum, dtype="float32")
        elem_sum_m.stop_gradient = True

        output = input * mask * elem_numel_m / elem_sum_m
        return output



