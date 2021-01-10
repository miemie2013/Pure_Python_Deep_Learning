
import os
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as L
import paddle.nn.functional as F
import numpy as np

use_gpu = True
gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()






print('================= bn的逐元素加减法、1x1分组卷积实现加法 ==================')
N = 2
in_C = 3
H = 28
W = 28
C = in_C


x = paddle.randn((N, C, H, W))
w = paddle.randn((1, C, 1, 1))   # bn归一化时会减去均值，这里的w相当于均值。

y = x - w   # [N, C, H, W]   每个通道C独享一个均值。

# 指数可以让乘法变加法。
x_in = L.exp(x)
w_r = L.transpose(w, [1, 0, 2, 3])   # [C, 1, 1, 1]
w_r = L.exp(-w_r)
y2 = F.conv2d(x_in, w_r, None, groups=C)   # [N, C, H, W]
y2 = L.log(y2)


y = y.numpy()
y2 = y2.numpy()
d = np.sum((y - y2) ** 2)
print(d)

'''
其实有更好的方案：自定义C++ OP。
按着F.conv2d()改，乘法改成加法。但是反向传播会有点不同，偏导数是常数1。
'''






