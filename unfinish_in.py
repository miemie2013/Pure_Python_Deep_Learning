
import os
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as L
import paddle.nn.functional as F
import numpy as np

use_gpu = True
gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()


epsilon = 1e-5
N = 2
C = 2
H = 2
W = 2
HW = H*W


x = paddle.randn((N, C, H, W))
x.stop_gradient = False

U = fluid.layers.reduce_mean(x, dim=[2, 3], keep_dim=True)  # [N, C, 1, 1]
V = fluid.layers.reduce_mean(fluid.layers.square(x - U), dim=[2, 3], keep_dim=True)  # [N, C, 1, 1]
normX = (x - U) / L.sqrt(V + epsilon)
Var1 = (x - U)
Var2 = 1.0 / L.sqrt(V + epsilon)
Var3 = (x - U) * 1.0 / L.sqrt(V + epsilon)


dUdx = paddle.grad(
    outputs=[U],
    inputs=[x],
    create_graph=True,
    retain_graph=True)[0]
dVdx = paddle.grad(
    outputs=[V],
    inputs=[x],
    create_graph=True,
    retain_graph=True)[0]
dnormXdx = paddle.grad(
    outputs=[normX],
    inputs=[x],
    create_graph=True,
    retain_graph=True)[0]
dVar1dx = paddle.grad(
    outputs=[Var1],
    inputs=[x],
    create_graph=True,
    retain_graph=True)[0]
dVar2dx = paddle.grad(
    outputs=[Var2],
    inputs=[x],
    create_graph=True,
    retain_graph=True)[0]
dVar3dx = paddle.grad(
    outputs=[Var3],
    inputs=[x],
    create_graph=True,
    retain_graph=True)[0]

dUdx = dUdx.numpy()
dVdx = dVdx.numpy()
dnormXdx = dnormXdx.numpy()
# dVar1dx全是0。看paddle.grad()的文档：对于每个inputs，计算所有outputs相对于其的梯度和。这样算的话，确实全是0.
# 我写的dVar1_dx则是outputs和inputs里相同位置的元素求偏导，所以里面每个元素是 1-1/M。所以dnormX_dx到底该怎么求？
dVar1dx = dVar1dx.numpy()
dVar2dx = dVar2dx.numpy()
dVar3dx = dVar3dx.numpy()



x = x.numpy()
V = V.numpy()
U = U.numpy()
dU_dx = np.zeros((N, C, H, W), np.float32) + (1.0 / HW)
dV_dx = np.sum(x - U, axis=(2, 3), keepdims=True) * -2 / HW / HW + (x - U) * 2 / HW
dnormX_dx = (1.0 - dU_dx) / ((V + epsilon) ** 0.5) \
            - 0.5 * (x - U) * dV_dx / ((V + epsilon) ** 1.5)
dVar1_dx = (1.0 - dU_dx)
dVar2_dx = -0.5 * dV_dx / ((V + epsilon) ** 1.5)
# 特意使用了paddle.grad()求出的dVar1dx和dVar2dx来表示dVar3_dx，然而仍然与dVar3dx不等。
dVar3_dx = dVar1dx * 1.0 / np.sqrt(V + epsilon) + (x - U) * dVar2dx


d = np.sum((dU_dx - dUdx) ** 2)   # dU_dx应该是正确的。
print(d)
d = np.sum((dV_dx - dVdx) ** 2)   # dV_dx应该是正确的。
print(d)
d = np.sum((dnormX_dx - dnormXdx) ** 2)
print(d)
d = np.sum((dVar1_dx - dVar1dx) ** 2)
print(d)
d = np.sum((dVar2_dx - dVar2dx) ** 2)   # dVar2_dx应该是正确的。
print(d)
d = np.sum((dVar3_dx - dVar3dx) ** 2)
print(d)






