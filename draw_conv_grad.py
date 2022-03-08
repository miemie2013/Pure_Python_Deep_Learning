

import cv2
import numpy as np


C = 6
groups = 2
c = C // groups
H = 4
W = 4
out_C = 4
out_H = 2
out_W = 2
dY_dX = np.zeros((1, C, H, W))

image = np.zeros((720, 300, 3), dtype=np.uint8) + 240

left_up = [120, 20]
w = 150

tri_w = int(w // 1.414)
tri_h = tri_w
ddd = tri_h + 10

featuremap_points = []
# 画出C个4x4的斜视视角的格子图
for i in range(C):
    points = []
    x0 = left_up[0]
    y0 = left_up[1]
    x1 = left_up[0] + w
    y1 = left_up[1]
    points.append([x0, y0])
    points.append([x1, y1])
    # cv2.line(image, (x0, y0), (x1, y1), (0, 0, 0), thickness=1)
    x1 = left_up[0] - tri_w
    y1 = left_up[1] + tri_h
    # cv2.line(image, (x0, y0), (x1, y1), (0, 0, 0), thickness=1)
    x0 = x1
    y0 = y1
    x1 = x0 + w
    y1 = y0
    points.append([x0, y0])
    points.append([x1, y1])
    # cv2.line(image, (x0, y0), (x1, y1), (0, 0, 0), thickness=1)
    x0 = x1
    y0 = y1
    x1 = left_up[0] + w
    y1 = left_up[1]
    # cv2.line(image, (x0, y0), (x1, y1), (0, 0, 0), thickness=1)
    featuremap_points.append(points)

    # 画出通道的颜色，每组使用不同颜色。
    group_id = i // c
    poly = [points[0], points[1], points[3], points[2]]
    poly = np.array(poly)
    if group_id == 0:
        cv2.fillConvexPoly(image, poly, (70, 62, 234))
    elif group_id == 1:
        cv2.fillConvexPoly(image, poly, (39, 138, 104))
    elif group_id == 2:
        cv2.fillConvexPoly(image, poly, (86, 205, 255))

    # 画出所有格子图边界
    x0 = points[0][0]
    y0 = points[0][1]
    x1 = points[1][0]
    y1 = points[1][1]
    x2 = points[2][0]
    y2 = points[2][1]
    x3 = points[3][0]
    y3 = points[3][1]
    cv2.line(image, (x0, y0), (x1, y1), (0, 0, 0), thickness=1)
    cv2.line(image, (x0, y0), (x2, y2), (0, 0, 0), thickness=1)
    cv2.line(image, (x3, y3), (x1, y1), (0, 0, 0), thickness=1)
    cv2.line(image, (x3, y3), (x2, y2), (0, 0, 0), thickness=1)


    # 画出所有格子，画横线
    for j in range(1, H, 1):
        x_0 = int(x0 + j * (x2 - x0) / H)
        y_0 = int(y0 + j * (y2 - y0) / H)
        cv2.line(image, (x_0, y_0), (x_0 + w, y_0), (0, 0, 0), thickness=1)

    # 画出所有格子，画竖线
    for j in range(1, W, 1):
        x_0 = int(x0 + j * (x1 - x0) / W)
        y_0 = y0
        cv2.line(image, (x_0, y_0), (x_0 - tri_w, y_0 + tri_h), (0, 0, 0), thickness=1)


    left_up[1] += ddd
    # if (i + 1) % c == 0:
    #     left_up[1] += 30



image2 = np.copy(image)

# 填入梯度的值

for ii in range(out_C):
    for jj in range(out_H):
        for kk in range(out_W):
            image = np.copy(image2)
            grad = 'W[%d]'%ii
            target_group_id = ii // 2
            j_lower = jj
            k_lower = kk
            j_upper = j_lower + 2
            k_upper = k_lower + 2

            for i in range(C):
                tri_w = int(w // 1.414)
                tri_h = tri_w
                points = featuremap_points[i]
                x0 = points[0][0]
                y0 = points[0][1]
                x1 = points[1][0]
                y1 = points[1][1]
                x2 = points[2][0]
                y2 = points[2][1]
                x3 = points[3][0]
                y3 = points[3][1]
                group_id = i // c

                for j in range(H):
                    x_0 = int(x0 + j * (x2 - x0) / H)
                    y_0 = int(y0 + j * (y2 - y0) / H)
                    for k in range(W):
                        x_ = int(x_0 + k * (x1 - x0) / W)
                        y_ = y_0
                        if group_id == target_group_id and j_lower <= j and j <= j_upper and k_lower <= k and k <= k_upper:
                            cv2.putText(image, grad, (x_ - 5, y_ + 15), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.3, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                        else:
                            cv2.putText(image, '0', (x_, y_ + 15), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            save_name = "conv_dY%d%d%d%d_dX.png"%(0, ii, jj, kk)
            cv2.imwrite(save_name, image)







