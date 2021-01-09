
import os
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as L
import paddle.nn.functional as F
import numpy as np

use_gpu = True
gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()





N = 2
in_C = 3
out_C = 8
H = 28
W = 28
kH = 1
kW = 1
C = out_C


# w = paddle.randn((C, in_C, kH, kW))

x = paddle.randn((N, in_C, H, W))
w = paddle.randn((C, in_C, 1, 1))

y = F.conv2d(x, w)


x_in = L.reshape(x, (N, 1, in_C, H, W))
w_r = L.reshape(w,  (1, C, in_C, 1, 1))
y2 = x_in * w_r   # [N, C, in_C, H, W]
y2 = L.reduce_sum(y2, dim=[2, ])

y = y.numpy()
y2 = y2.numpy()
d = np.sum((y - y2) ** 2)
print(d)


'''
因此，两个形如
(N, 1, in_C, H, W)
(1, C, in_C, 1, 1)
或者说形如
(A, 1, in_C, B, C)
(1, D, in_C, 1, 1)
的张量逐元素相乘，实际上可以转换成1x1卷积来进行快速计算（以及更节省显存）。
某一维度数值相等，那么那一维作为in_C维。
强记：两个5维张量，有一维维大小相等，有一个张量有1个"1"，有一个张量有3个"1"（即"1"的数量分别为1、3）。
简记：有一维维大小相等，且13。
'''





