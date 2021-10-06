# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 22:45:42 2021

@author: styra
"""

import cv2 # 仅用于读取图像矩阵
import matplotlib.pyplot as plt  
import numpy as np

gray_level = 256  # 灰度级

def pixel_probability(img):
    """
    计算像素值出现概率
    """
    assert isinstance(img, np.ndarray)
    prob = np.zeros(shape=(256))

    for rv in img:
        for cv in rv:
            prob[cv] += 1

    r, c = img.shape
    prob = prob / (r * c)
    return prob

def probability_to_histogram(img, prob):
    """
    根据像素概率将原始图像直方图均衡化
    """
    prob = np.cumsum(prob)  # 累计概率
    img_map = [int(i * prob[i]) for i in range(256)]  # 像素值映射

   # 像素值替换
    assert isinstance(img, np.ndarray)
    r, c = img.shape
    for ri in range(r):
        for ci in range(c):
            img[ri, ci] = img_map[img[ri, ci]]
    return img

def plot(y, name):
    """
    画直方图，len(y)==gray_level
    """
    plt.figure(num=name)
    plt.bar([i for i in range(gray_level)], y, width=1)
    plt.show()

if __name__ == '__main__':

    img = cv2.imread("test.bmp", 0)  # 读取灰度图
    prob = pixel_probability(img)
    plot(prob, "原图直方图")

    # 直方图均衡化
    img = probability_to_histogram(img, prob)
    cv2.imwrite("test_hist.bmp", img)  # 保存图像

    prob = pixel_probability(img)
    plot(prob, "直方图均衡化结果")
    