'''

editor: Jones
date: 20200220
content: 
base function

1 門檻值
2.二值化
3.非零的壓力值陣列
4.一維 array 重心位置
5.一維 array 重心位置
'''


import numpy as np
import matplotlib.pyplot as plt
# import cv2
import pandas as pd 
import math
import seaborn as sns


# 門檻值 22
def thresholding(raw_data_array):

	threshold = 22
	raw_data_array[raw_data_array <= threshold] = 0
	return raw_data_array


# 二值化
def binarization(binarized_array):

	binarized = 0
	binarized_array[binarized_array <= binarized] = 0
	binarized_array[binarized_array > binarized] = 1023

	return binarized_array


# 非零的壓力值陣列
def nonzero_pressure_value(threshold_array):

	nonzero_array = threshold_array[np.nonzero(threshold_array > 0)]
	return nonzero_array


# 一維 array 重心位置
def center_of_mass(one_dimensional_array):

	weight_sum = 0
	for index, value in enumerate(one_dimensional_array):
		weight_sum = weight_sum + index * value

	mass_center = weight_sum/np.sum(one_dimensional_array)
	return mass_center


# 二維 array 重心位置
def two_dimension_center_of_mass(two_dimensional_array):

	# 垂直轴阵列
	vertical_axis_array = np.sum(two_dimensional_array, axis = 1)

	# 水平軸阵列
	horizontal_axis_array = np.sum(two_dimensional_array, axis = 0)

	y_axis_mc = center_of_mass(vertical_axis_array)
	x_axis_mc = center_of_mass(horizontal_axis_array)

	return x_axis_mc, y_axis_mc


# def printHello():
# 	print("Hello")