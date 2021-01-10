
import os
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as L
import paddle.nn.functional as F
import numpy as np

use_gpu = True
gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()





print('================= 逐元素乘、1x1卷积、yolact中的矩阵乘法 ==================')
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


x_in2 = L.transpose(x, [0, 2, 3, 1])   # [N, H, W, in_C]
w_r2 = L.reshape(w,  (C, in_C))
w_r2 = L.transpose(w_r2, [1, 0])   # [in_C, C]
y3 = L.matmul(x_in2, w_r2)      # [N, H, W, C]
y3 = L.transpose(y3, [0, 3, 1, 2])   # [N, C, H, W]


y = y.numpy()
y2 = y2.numpy()
y3 = y3.numpy()
d = np.sum((y - y2) ** 2)
print(d)
d = np.sum((y - y3) ** 2)
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


推广：
两个形如
(A, 1, A1, A2, ..., An, B, C)
(1, D, A1, A2, ..., An, 1, 1)
的张量逐元素相乘，实际上可以将它们reshape成
(A, 1, A1*A2*...*An, B, C)
(1, D, A1*A2*...*An, 1, 1)
再使用1x1卷积计算。

简记：“13”。


另外，yolact中的矩阵乘法可转换成1x1卷积。
[N, H, W, in_C] x [in_C, C] = [N, H, W, C]
或者说
[N*H*W, in_C] x [in_C, C] = [N*H*W, C]
[N, in_C] x [in_C, C] = [N, C]   （若H==W==1）
即二维矩阵的矩阵乘法相当于在1x1图像上做1x1卷积。
（此时图像每个像素和输出结果是全连接的。在1x1图像上做1x1卷积相当于全连接层。）
'''



print('================= bn的逐元素乘、1x1分组卷积、平均池化 ==================')
N = 2
in_C = 3
H = 28
W = 28
C = in_C


x = paddle.randn((N, C, H, W))
w = paddle.randn((1, C, 1, 1))   # bn归一化时会乘以标准差的倒数，这里的w相当于标准差的倒数。

y = x * w   # [N, C, H, W]   每个通道C独享一个标准差。

w_r = L.transpose(w, [1, 0, 2, 3])   # [C, 1, 1, 1]
y2 = F.conv2d(x, w_r, None, groups=C)   # [N, C, H, W]


y = y.numpy()
y2 = y2.numpy()
d = np.sum((y - y2) ** 2)
print(d)


'''
因此，两个形如
(N, C, H, W)
(1, C, 1, 1)
或者说形如
(R, C, S, T)
(1, C, 1, 1)
的张量逐元素相乘，实际上可以转换成1x1分组卷积来进行快速计算（以及更节省显存）。
某一维度数值相等，那么那一维作为C维。
强记：两个4维张量，有一维维大小相等，有一个张量有3个"1"（即"1"的数量分别为0、3）。

推广：
两个形如
(R, A1, A2, ..., An, S, T)
(1, A1, A2, ..., An, 1, 1)
的张量逐元素相乘，实际上可以将它们reshape成
(R, A1*A2*...*An, S, T)
(1, A1*A2*...*An, 1, 1)
再使用1x1卷积计算。

考虑到它们“1”的个数，简记为"03"。

另外，平均池化是每个通道各自进行，因此可转换成分组卷积。
'''



print('================= in的逐元素乘、1x1分组卷积 ==================')
N = 2
in_C = 3
H = 28
W = 28
C = in_C


x = paddle.randn((N, C, H, W))
w = paddle.randn((N, C, 1, 1))   # in归一化时会乘以标准差的倒数，这里的w相当于标准差的倒数。

y = x * w   # [N, C, H, W]   每个样本N每个通道C独享一个标准差。

# 如果x保持着形状[N, C, H, W]，是无法转换成等价的1x1分组卷积。这是因为卷积操作本来就是每张图片共享了卷积核，
# 而in要求每张图片有各自的C个标准差。所以要转换一下。
# x如果reshape成[1, N*C, H, W]，恰好变成了1张图片下bn的情况。

x_in = L.reshape(x, (1, N*C, H, W))
w_r = L.reshape(w, (N*C, 1, 1, 1))
y2 = F.conv2d(x_in, w_r, None, groups=N*C)   # [1, N*C, H, W]
y2 = L.reshape(y2, (N, C, H, W))


y = y.numpy()
y2 = y2.numpy()
d = np.sum((y - y2) ** 2)
print(d)


'''
因此，两个形如
(N, C, H, W)
(N, C, 1, 1)
的张量逐元素相乘，实际上可以转换成1x1分组卷积来进行快速计算（以及更节省显存）。
我们把相等的维N、C合并成新的通道维N*C，转换成了bn下的情况。

推广：
两个形如
(A1, A2, ..., An, B, C)
(A1, A2, ..., An, 1, 1)
的张量逐元素相乘，实际上可以将它们reshape成
(1, A1*A2*...*An, B, C)
(A1*A2*...*An, 1, 1, 1)
再使用1x1分组卷积计算。

考虑到它们“1”的个数，简记为"02"。
"02"可以看做"03"R=1时的情况。两个张量只要能reshape成"02"、"03"、"13"等就可以转换成卷积。


'''





