'''
editor: Jones
date: 20200219
content: 
頭和脚區分
第四種方法：
1. 將身體切分三等份，分為頭部，中部，腳部
2. 分別計算頭部array，腳部array 的垂直軸的重心，然後再計算其變異數 
3. 分別計算頭部array，腳部array 的水平軸的重心，然後再計算其變異數 
4. 比較變異數的大小
'''

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import leastsq
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy.interpolate import interp1d
import pandas as pd
import cv2
from PIL import Image
from base_function_package.base_function import thresholding, center_of_mass, two_dimension_center_of_mass


# 頭部、中部、腳部區分
def head_foot_split(my_array):

	up = 0
	down = 0
	col = my_array.shape[0]
	# print(col)

	if(np.max(my_array[0]) == 0):
		up = 1
		if(np.max(my_array[1]) == 0):
			up = 2
			if(np.max(my_array[2]) == 0):
				up = 3
	else:
		up = 0

	if(np.max(my_array[-1]) == 0):
		down = -1
		if(np.max(my_array[-2]) == 0):
			down = -2
			if(np.max(my_array[2]) == 0):
				down = -3

	my_array = my_array[up: col+down]
	# print(my_array)

	head  = 0
	foot = 0

	col_2 = my_array.shape[0]
	if(col_2 % 3 == 1):
		head = int(col_2 / 3) 
		foot = int(col_2 / 3)
	elif(col_2 % 3 == 2):
		head = int(col_2 / 3 + 1)
		foot = int(col_2 / 3 + 1) 
	else:
		head = int(col_2 / 3) 
		foot = int(col_2 / 3) 

	return head, foot


# 頭腳區分
def head_foot_diff(my_array):

	head, foot = head_foot_split(my_array)

	head_array = my_array[:head]
	foot_array = my_array[-foot:]

	y_axis_head_array = np.sum(head_array, axis = 1)
	y_axis_foot_array = np.sum(foot_array, axis = 1)
	x_axis_head_array = np.sum(head_array, axis = 0)
	x_axis_foot_array = np.sum(foot_array, axis = 0)

	y_axis_head_array_mean = np.mean(y_axis_head_array)
	y_axis_foot_array_mean = np.mean(y_axis_foot_array)
	x_axis_head_array_mean = np.mean(x_axis_head_array)
	x_axis_foot_array_mean = np.mean(x_axis_foot_array)

	y_axis_head_array_var = np.var(y_axis_head_array)
	y_axis_foot_array_var = np.var(y_axis_foot_array)
	x_axis_head_array_var = np.var(x_axis_head_array)
	x_axis_foot_array_var = np.var(x_axis_foot_array)

	if y_axis_head_array_mean > y_axis_foot_array_mean and x_axis_head_array_mean:
		return my_array
	else:
		# print('睡床尾')
		mat180 = np.rot90(my_array, 2)
		return mat180


# 獲取旋轉角度
def rotation_angle(my_array):

	xg, yg = two_dimension_center_of_mass(my_array)

	head ,foot = head_foot_split(my_array)
	head_array = my_array[:head]
	# print(head_array)
	middle_array = my_array[head:-foot]
	# print(middle_array)
	foot_array = my_array[-foot:]
	# print(foot_array)
	head_xg, head_yg = two_dimension_center_of_mass(head_array)
	middle_xg, middle_yg = two_dimension_center_of_mass(middle_array)
	middle_yg = middle_yg + head

	a = np.array([middle_xg - head_xg, middle_yg - head_yg])
	b = np.array([[0],
				[middle_yg - head_yg]])

	# print(np.vdot(a,b))
	# print((math.sqrt(a[0] **2 + a[1] ** 2) * math.sqrt(b[0][0] **2 + b[1][0] ** 2)))

	cos_angle = np.vdot(a,b) / (math.sqrt(a[0] **2 + a[1] ** 2) * math.sqrt(b[0][0] **2 + b[1][0] ** 2))
	angle = math.acos(cos_angle) * 180 /math.pi
	return angle

	# return angle, head_xg, head_yg, middle_xg, middle_yg


# 影像旋轉
def image_rotation(my_array, angle):

	(h, w) = my_array.shape
	center = (w // 2, h // 2)
	# print('center =', center)

	M = cv2.getRotationMatrix2D(center, -angle, 1.0)
	# print(M)
	my_array = (my_array/4).astype(np.uint8)
	# print(my_array)
	rotated = cv2.warpAffine(my_array, M, (w, h))
	# print(rotated)

	return rotated

	# image_ratate_array = np.zeros((20,11))
	# # print(image_ratate_array)

	# rotated_index_list = []

	# for row_index, row_element in enumerate(my_array):
	# 	for col_index, col_element in enumerate(row_element):
	# 		b = np.array([[col_index], [row_index], [1]])
	# 		a = np.dot(M, b)
	# 		# print(a)
	# 		rotated_index_list.append(a)

	# rotated_index_array = (np.array(rotated_index_list)).reshape(20,22)
	# print(rotated_index_array)

	

	# b = np.array([[2], [5], [1]]) 
	# print(b)
	# a = np.dot(M, b)
	# print(a)
	# return rotated



##主函數從此處開始###
def main():

	# open file 
	raw_data = pd.read_csv('data_csv/allnightData/Jones1224_allnight.csv')
	raw_data = raw_data.to_numpy()

	row = 20
	col = 11
	index = 10000

	original_array = thresholding(raw_data[index][:220].reshape(20,11))
	print(original_array)

	head_foot_array = head_foot_diff(original_array)
	# print(head_foot_array)
	angle, head_xg, head_yg, middle_xg, middle_yg= rotation_angle(head_foot_array)
	print(angle)
	print(head_xg)
	print(head_yg)
	print(middle_xg)
	print(middle_yg)

	# new_array = cv2.resize(head_foot_array/4, (110, 200), interpolation=cv2.INTER_LINEAR)

	# rotated_array = image_rotation(head_foot_array,angle)
	# print(rotated_array)

	# # 顯示圖形
	plt.figure()
	plt.subplot() # 第一行的左图
	plt.title('Face Left Posture')
	plt.imshow(head_foot_array , cmap=plt.cm.jet)
	plt.yticks(np.arange(-0.5,20.5,1.0))
	plt.xticks(np.arange(-0.5,11.5,1.0))
	plt.colorbar()
	# plt.scatter([head_xg,], [head_yg,], marker='.', color = 'red',linewidths  = 6)
	# plt.scatter([middle_xg,], [middle_yg,], marker='.', color = 'green', linewidths  = 6)
	# plt.plot([head_xg, middle_xg], [head_yg, middle_yg], color = 'white', linewidth = 3)
	# plt.plot([head_xg, head_xg], [-0.5, 19.5], color = 'black', linewidth = 3)
	# plt.plot([-0.5, 10.5], [5.5, 5.5], color = 'red', linewidth = 3)
	# plt.plot([-0.5, 10.5], [12.5, 12.5], color = 'red', linewidth = 3)
	plt.show()

	# plt.subplot(2,2,2) # 第一行的左图
	# plt.title('Sleeping Positions')
	# plt.imshow(original_array[::-1], cmap=plt.cm.jet)
	# plt.ylim((20.5, -0.5))
	# plt.yticks(np.arange(-0.5,20.5,1.0))


	# plt.subplot(2,2,3) # 第一行的左图
	# plt.plot(range(len(np.sum(original_array, axis = 1))), np.sum(original_array, axis = 1), '--o') # 只有點
	# plt.xlim((-0.5, 19.5))
	# plt.xticks(np.arange(0,20,1.0))
	# plt.scatter(yg_positive, yg_value_positive, marker='^', color = 'red',linewidths  = 3)
	# plt.scatter(upper_positive, yg_upper_value_positive, marker='^', color = 'green',linewidths  = 3)
	# plt.scatter((down_positive+yg_positive), yg_down_value_positive, marker='^', color = 'orange',linewidths  = 3)

	# plt.subplot(2,2,4) # 第一行的左图
	# plt.plot(range(len(np.sum(original_array[::-1], axis = 1))), np.sum(original_array[::-1], axis = 1), '--o') # 只有點
	# plt.xlim((-0.5, 19.5))
	# plt.xticks(np.arange(0,20,1.0))
	# plt.scatter(yg_negative, yg_value_negative, marker='^', color = 'red',linewidths  = 3)
	# plt.scatter(upper_negative, yg_upper_value_negative, marker='^', color = 'green',linewidths  = 3)
	# plt.scatter((down_negative+yg_negative), yg_down_value_negative, marker='^', color = 'orange',linewidths  = 3)
	# plt.legend()



# 程式起點
if __name__ == '__main__':
	main()