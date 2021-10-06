# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 22:45:42 2021

@author: styra
"""

import cv2
from matplotlib import pyplot as plt

def whole_hist(image):
	'''
	绘制整幅图像的直方图
	'''
	plt.hist(image.ravel(), 256, [0, 256]) #numpy的ravel函数功能是将多维数组降为一维数组
	plt.show()

def channel_hist(image):
	'''
	画三通道图像的直方图
	'''
	color = ('b', 'g', 'r')   #这里画笔颜色的值可以为大写或小写或只写首字母或大小写混合
	for i , color in enumerate(color):
		hist = cv2.calcHist([image], [i], None, [256], [0, 256])  #计算直方图
		plt.plot(hist, color)
		plt.xlim([0, 256])
	plt.show()
    
def Histogram_Equalization(image):
    # 直方图均衡化
    (b, g, r) = cv2.split(image)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    imageHE = cv2.merge((bH, gH, rH))
    cv2.imwrite('testHE.bmp', imageHE)
    return imageHE

image = cv2.imread('test.bmp')
cv2.imshow('image', image)
cv2.waitKey(0)
whole_hist(image)
channel_hist(image)
imageHE = Histogram_Equalization(image)
whole_hist(imageHE)