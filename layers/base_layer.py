#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-07-05 13:35:27
#   Description : 《纯python实现一个深度学习框架》课程源码
#
# ================================================================

class Layer(object):
    pass

class Layer2(object):
    def init_weights(self, pretrained=None):
        raise NotImplementedError
    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

class ActivationLayer(Layer):
    # 激活层没有可训练权重
    pass

class LossLayer(Layer):
    pass




