

import cv2
import numpy as np

'''
打印装置卷积中，输出每一个元素yij对W的梯度dyij/dW。

损失对W的梯度为：
西格玛（求和符号）dloss/dyij * dyij/dW

'''


# 默认padding都是0


stride = 2
kH = 3
out_H = 13
in_H = 6
# out_H = 5
# in_H = 2
kW = kH
out_W = out_H
in_W = in_H


# 每一个元素yij对W的梯度dyij/dW。

for i in range(out_H):
    print('===================== line %d =====================' % i)
    for j in range(out_W):

        # 卷积核滑动，只会在H和W两个方向上滑动
        X_y = 0
        aaaaaaa = np.zeros((kH, kW, 2)) - 1
        for start_y in range(0, out_H - kH + 1, stride):
            X_x = 0
            for start_x in range(0, out_W - kW + 1, stride):
                if start_y <= i and i < start_y + kH and start_x <= j and j < start_x + kW:
                    if i == 0 and j == 6:
                        print()
                    aaaaaaa[i - start_y][j - start_x][0] = X_y
                    aaaaaaa[i - start_y][j - start_x][1] = X_x
                X_x += 1
            X_y += 1
        print('dY[%d, %d]_dW ='%(i, j))
        for r in range(kH):
            str = ''
            for s in range(kW):
                x = aaaaaaa[r][s][0]
                y = aaaaaaa[r][s][1]
                if x < 0 and y < 0:
                    str += '0000000\t'
                else:
                    str += 'X[%d, %d]\t'%(x, y)
            print(str)
        print()




