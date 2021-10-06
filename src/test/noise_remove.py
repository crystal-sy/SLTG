# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 22:15:45 2021

@author: styra
"""

import cv2 # 仅用于读取图像矩阵

img = cv2.imread("demo.bmp", 0)  
step_med = 6
step_mea = 7

def median_filter(x, y):
    sum_s = []
    for k in range(-int(step_med/2), int(step_med/2)+1):
        for m in range(-int(step_med/2), int(step_med/2)+1):
            sum_s.append(img[x+k][y+m])
    sum_s.sort()
    return sum_s[(int(step_med*step_med/2)+1)]

def mean_filter(x, y):
    sum_s = 0
    for k in range(-int(step_mea/2), int(step_mea/2)+1):
        for m in range(-int(step_mea/2), int(step_mea/2)+1):
            sum_s += img[x+k][y+m] / (step_mea*step_mea)
    return sum_s

if __name__ == '__main__':
    # 读取原灰度图
    img_copy_med = cv2.imread("demo.bmp", 0) 
    img_copy_mea = cv2.imread("demo.bmp", 0) 
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            img_copy_med[i][j] = img[i][j]
            img_copy_mea[i][j] = img[i][j]

    # 中值滤波
    for i in range(int(step_med/2), img.shape[0]-int(step_med/2)):
        for j in range(int(step_med/2), img.shape[1]-int(step_med/2)):
            img_copy_med[i][j] = median_filter(i, j)
    cv2.imwrite("median_filter.bmp", img_copy_med)  
    
    # 均值滤波
    for i in range(int(step_mea/2), img.shape[0]-int(step_mea/2)):
        for j in range(int(step_mea/2), img.shape[1]-int(step_mea/2)):
            img_copy_mea[i][j] = mean_filter(i, j)
    cv2.imwrite("mean_filter.bmp", img_copy_mea)

    
