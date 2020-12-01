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


class Optimizer(object):
    pass


class SGD(Optimizer):
    def __init__(self,
                 lr,
                 aaa=0,
                 bbb=0):
        super(SGD, self).__init__()
        self.lr = lr

    def update(self, param_name, param, dparam, param_lr=1.0, decay_type=None, decay_coeff=0.0):
        assert decay_type in ['L1Decay', 'L2Decay', None]
        lr = self.lr   # 优化器的学习率
        lr = param_lr * lr   # 最终的学习率 = 参数的学习率 * 优化器的学习率
        if decay_type is None:
            param = param - lr * dparam
        elif decay_type == 'L2Decay':
            # L2正则化，即在损失函数上加上该参数的平方项：
            # loss_new = loss + 0.5 * decay_coeff * param^2
            # loss_new对param求偏导：
            # dloss_new = dloss + decay_coeff * param    (dloss即本方法的形参dparam)
            #           = dparam + decay_coeff * param
            # 所以权重更新公式为
            # param = param - lr * dloss_new
            #       = param - lr * (dparam + decay_coeff * param)
            #       = param - lr * dparam - lr * decay_coeff * param
            #       = (1.0 - lr * decay_coeff) * param - lr * dparam
            # 推导完成。
            keep = (1.0 - lr * decay_coeff)   # keep通常是0.999...这样的值，即param先乘以一个0.999...的值进行衰减再减去lr * dparam。
            param = keep * param - lr * dparam
        elif decay_type == 'L1Decay':
            # L2正则化，即在损失函数上加上该参数的绝对值项：
            # loss_new = loss + decay_coeff * |param|
            # loss_new对param求偏导：
            # dloss_new = dloss + decay_coeff * sign(param)    (dloss即本方法的形参dparam)
            #           = dparam + decay_coeff * sign(param)
            # 所以权重更新公式为
            # param = param - lr * dloss_new
            #       = param - lr * (dparam + decay_coeff * sign(param))
            #       = param - lr * dparam - lr * decay_coeff * sign(param)
            # 推导完成。相当于不使用正则化时param再加上或减去一个极小的正实数lr * decay_coeff。
            param = param - lr * dparam - lr * decay_coeff * np.sign(param)
        return param



class Momentum(Optimizer):
    def __init__(self,
                 lr,
                 momentum=0.9,
                 use_nesterov=False):
        super(Momentum, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.use_nesterov = use_nesterov
        self.velocities = {}

    def update(self, param_name, param, dparam, param_lr=1.0, decay_type=None, decay_coeff=0.0):
        assert decay_type in ['L1Decay', 'L2Decay', None]
        lr = self.lr   # 优化器的学习率
        lr = param_lr * lr   # 最终的学习率 = 参数的学习率 * 优化器的学习率
        momentum = self.momentum
        if decay_type is None:
            velocity = self.velocities[param_name] if param_name in self.velocities.keys() else np.zeros(dparam.shape)
            velocity = momentum * velocity + dparam
            param = param - lr * velocity
            self.velocities[param_name] = velocity
        elif decay_type == 'L2Decay':
            # L2正则化，即在损失函数上加上该参数的平方项：
            # loss_new = loss + 0.5 * decay_coeff * param^2
            # loss_new对param求偏导：
            # dloss_new = dloss + decay_coeff * param    (dloss即本方法的形参dparam)
            #           = dparam + decay_coeff * param
            # 所以权重更新公式为
            # param = param - lr * dloss_new
            #       = param - lr * (dparam + decay_coeff * param)
            #       = param - lr * dparam - lr * decay_coeff * param
            #       = (1.0 - lr * decay_coeff) * param - lr * dparam
            # 推导完成。
            velocity = self.velocities[param_name] if param_name in self.velocities.keys() else np.zeros(dparam.shape)
            dloss_new = dparam + decay_coeff * param
            velocity = momentum * velocity + dloss_new
            param = param - lr * velocity
            self.velocities[param_name] = velocity
        elif decay_type == 'L1Decay':
            # L2正则化，即在损失函数上加上该参数的绝对值项：
            # loss_new = loss + decay_coeff * |param|
            # loss_new对param求偏导：
            # dloss_new = dloss + decay_coeff * sign(param)    (dloss即本方法的形参dparam)
            #           = dparam + decay_coeff * sign(param)
            # 所以权重更新公式为
            # param = param - lr * dloss_new
            #       = param - lr * (dparam + decay_coeff * sign(param))
            #       = param - lr * dparam - lr * decay_coeff * sign(param)
            # 推导完成。相当于不使用正则化时param再加上或减去一个极小的正实数lr * decay_coeff。
            param = param - lr * dparam - lr * decay_coeff * np.sign(param)
        return param








