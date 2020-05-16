'''

editor: Jones
date: 2019/07/29
content: 
1. txt to csv

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# 門檻值
def min_thresholding(threshold_array):

	min_threshold = 50
	# print(min_threshold)
	threshold_array[threshold_array <= min_threshold] = 0

	return threshold_array

def center_of_gravity(my_array):
	
	sum_i = 0
	sum_j = 0

	for y, i in enumerate(my_array):
		for x, j in enumerate(i):
			b = sum_i
			a = sum_j

			sum_i = y * j
			sum_i = sum_i + b

			sum_j = x * j
			sum_j = sum_j + a

	yg = sum_i/np.sum(my_array)
	xg = sum_j/np.sum(my_array)

	return xg, yg

def head_foot_diff(my_array):

	up = 0
	down = 0
	col = my_array.shape[0]
	print(col)

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
	print(my_array)

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

	print(head)
	print(foot)

	head_array = my_array[:head]
	foot_array = my_array[-foot:]

	print(head_array)
	print(foot_array)
	print(np.sum(head_array, axis = 1))
	print(np.sum(foot_array, axis = 1))


	# head_array 垂直軸變異數
	head_vertical_var = np.var(np.sum(head_array, axis = 0))
	# foot_array 垂直軸變異數
	foot_vertical_var = np.var(np.sum(foot_array, axis = 0))

	error_1 = head_vertical_var / foot_vertical_var


	# head_array 水平軸變異數
	head_horizontal_var = np.var(np.sum(head_array, axis = 1))
	# foot_array 水平軸變異數
	foot_horizontal_var = np.var(np.sum(foot_array, axis = 1))

	error_2 = head_horizontal_var / foot_horizontal_var

	print(error_1, error_2)
	print(head_vertical_var)
	print(foot_vertical_var)
	print(head_horizontal_var)
	print(foot_horizontal_var)

	if(np.sum(my_array)>1023):
		if(head_vertical_var > foot_vertical_var and head_horizontal_var > foot_horizontal_var):
			print('睡在床头')
		else:
			print('睡床尾')

	else:
		print('離床')
	# return head_vertical_var, foot_vertical_var


def head_middle(my_array):

	head_array = my_array[:7]
	middle_array = my_array[7:13]
	foot_array = my_array[14:]

	print('head_array =', head_array)
	print('middle_array =', middle_array)
	print('foot_array =', foot_array)

	distance = 7

	xg1, yg1 = center_of_gravity(head_array)
	xg2, yg2 = center_of_gravity(middle_array)
	xg3, yg3 = center_of_gravity(foot_array)
	yg2 = yg2 + distance
	yg3 = yg3 + 13

	print('xg1 = %f, yg1 = %f' % (xg1, yg1))
	print('xg2 = %f, yg2 = %f' % (xg2, yg2))
	print('xg3 = %f, yg3 = %f' % (xg3, yg3))

	return xg1, yg1, xg2, yg2, xg3, yg3


def head_foot(my_array):

	head_array = my_array[:10]
	foot_array = my_array[10:]

	print('head_array =', head_array)
	print('foot_array =', foot_array)

	xg_head, yg_head = center_of_gravity(head_array)
	xg_foot, yg_foot = center_of_gravity(foot_array)
	yg_foot = yg_foot + 10

	print('xg_head = %f, yg_head = %f' % (xg_head, yg_head))
	print('xg_foot = %f, yg_foot = %f' % (xg_foot, yg_foot))

	return xg_head, yg_head, xg_foot, yg_foot


# def rotate_angle():

# 	x=np.array([xg1-xg2,yg2-yg1])
# 	y=np.array([xg3-xg2,yg2-yg3])

def odd_number_list(my_list):

	count = 0

	new_my_list = []

	while count < 200:
		new_my_list.append(my_list[count])
		count += 2
	print('len =', len(new_my_list))

	return new_my_list


# 主程序
def main():
	# open file 
	f = open('textdata/培健1210_lying.txt', 'r',encoding = 'utf-8')

	raw_data_list = []
	for x in f:
		raw_data_list.append(x)
	print(len(raw_data_list))

	# countA = 0
	# countB = 0
	# countC = 0
	# countD = 0

	# # 正躺
	face_up_list = []
	# # 右侧躺
	face_right_list = []
	# # 左侧躺
	face_left_list = []
	# # 俯卧
	face_down_list = []


	for item in raw_data_list[1:]:
		if item[-2] == '0':
			face_up_list.append(item)
		elif item[-2] == '1':
			face_right_list.append(item)
		elif item[-2] == '2':
			face_left_list.append(item)
		elif item[-2] == '3':
			face_down_list.append(item)

	print('countA =', len(face_up_list))
	print('countB =', len(face_right_list))
	print('countC =', len(face_left_list))
	print('countD =', len(face_down_list))

	original_array = np.array(face_left_list[0].split(),dtype = int)[:220].reshape(20,11)
	xg, yg = center_of_gravity(original_array)
	head_foot_diff(original_array)
	print('xg =', xg)
	print('yg =', yg)

	xg1, yg1, xg2, yg2, xg3, yg3 = head_middle(original_array)

	# 顯示圖形
	fig, ax = plt.subplots()
	plt.title('Sleeping Positions')
	plt.imshow(original_array, cmap=plt.cm.jet)
	plt.yticks(np.arange(0,20,1.0))
	plt.xticks(np.arange(0,11,1.0))
	plt.scatter([xg,],[yg,], marker = '*', color = 'green', s = 80)
	plt.scatter([xg1,],[yg1,], marker = '*', color = 'green', s = 80)
	plt.scatter([xg2,],[yg2,], marker = '*', color = 'orange', s = 80)
	plt.scatter([xg3,],[yg3,], marker = '*', color = 'pink', s = 80)

	# plt.scatter([xg_head,],[19-yg_head,], marker = 'o', color = 'w', s = 50)
	# plt.scatter([xg_foot,],[19-yg_foot,], marker = 'o', color = 'black', s = 50)
	# plt.colorbar()
	plt.show()

	index = -200

	# 400 frame 子暘，佳芸， 阿杜，建宇，思琪, 竑量，培健，皓菘，沛忱，沛臻，嘉宏，燕鴻，翊嘉，秉渝
	face_up_list = face_up_list[index:]
	face_right_list = face_right_list[index:]
	face_left_list = face_left_list[index:]
	face_down_list = face_down_list[index:]

	# face_up_list = odd_number_list(face_up_list)
	# face_right_list = odd_number_list(face_right_list)
	# face_left_list = odd_number_list(face_left_list)
	# face_down_list = odd_number_list(face_down_list)

	# 400 frame 郅博
	# face_up_list = face_up_list[index:]
	# face_right_list = face_right_list[index:]
	# face_left_list = face_left_list[2:-10]
	# face_down_list = face_down_list[index:]

	# face_up_list = odd_number_list(face_up_list)
	# face_right_list = odd_number_list(face_right_list)
	# face_left_list = odd_number_list(face_left_list)
	# face_down_list = odd_number_list(face_down_list)

	# 400 frame 育誠
	# face_up_list = face_up_list[index:]
	# new_face_right_list = face_right_list[18:218]
	# face_left_list = face_right_list[252:452]
	# face_down_list = face_right_list[index:]

	# 400 frame 顯郡
	# face_up_list = face_up_list[index:]
	# face_right_list = face_right_list[index:]
	# face_left_list = face_left_list[-210:-10]
	# face_down_list = face_down_list[-204:-4]

	# 400 frame 鵬翔
	# new_face_up_list = face_up_list[78:278]
	# face_right_list = face_up_list[300:500]
	# face_left_list = face_left_list[index:]
	# face_down_list = face_down_list[index:]


	# new_face_up_list = odd_number_list(new_face_up_list)
	# face_right_list = odd_number_list(face_right_list)
	# face_left_list = odd_number_list(face_left_list)
	# face_down_list = odd_number_list(face_down_list)

	# 400 frame Jones
	# face_up_list = raw_data_list[1310:1510]
	# face_right_list = raw_data_list[1600:1800]
	# face_left_list = raw_data_list[2000:2200]
	# face_down_list = raw_data_list[2320:2520]


	# face_up_list = odd_number_list(face_up_list)
	# face_right_list = odd_number_list(face_right_list)
	# face_left_list = odd_number_list(face_left_list)
	# face_down_list = odd_number_list(face_down_list)



	all_list  = face_up_list + face_right_list + face_left_list + face_down_list
	print(len(all_list))

	all_array = []

	for i in all_list:
		j = np.array(i.split(),dtype = int)[:220]
		all_array.append(j)

	new_array = np.array(all_array)
	# print(new_array[-1])

	columns = []
	i = 0
	while i < 220:
		col = 'pixel%d' %i
		i = i + 1
		columns.append(col)

	df = pd.DataFrame(new_array, columns = columns)

	# # label 0, 正躺 face up
	# # label 1, 右侧躺 face right
	# # label 2, 左侧躺 face left
	# # lable 3, 俯卧 face down

	label_0 = np.zeros(100, dtype=int)
	label_1 = np.ones(100, dtype=int)
	label_2 = np.ones(100, dtype=int) * 2
	label_3 = np.ones(100, dtype=int) * 3

	label = np.hstack((label_0, label_1, label_2, label_3))
	df.insert(220, "label", label, True)
	print(df.head())
	# df.to_csv ("csvdata/1214Jones_lying.csv" , encoding = "utf-8")



	# # 顯示圖形
	# fig, ax = plt.subplots()
	# plt.title('Positions')
	# plt.imshow(rotated, cmap = plt.cm.jet)
	# plt.xlim((-0.5, 10.5))
	# plt.ylim((-0.5, 19.5))
	# plt.xticks(np.arange(-0.5,10.5,1.0))
	# plt.yticks(np.arange(-0.5,19.5,1.0))
	# plt.show()

		# xg1, yg1, xg2, yg2, xg3, yg3 = head_middle(original_array)


	# print('np.dot = ', np.dot((xg1,yg1), (xg2, yg2)))

	# x=np.array([0,yg2-yg1])
	# y=np.array([xg2-xg1,yg1-yg2])

	# print('x =', x)
	# print('y =', y)

	# # 两个向量
	# Lx=np.sqrt(x.dot(x))
	# Ly=np.sqrt(y.dot(y))

	# #相当于勾股定理，求得斜线的长度
	# cos_angle=x.dot(y)/(Lx*Ly)
	# print('cos_angle =', cos_angle)

	# angle=np.arccos(cos_angle)
	# angle2=angle*360/2/np.pi
	# #变为角度
	# print('angle = ', angle2)

	# (h, w) = original_array.shape
	# center = (w // 2, h // 2)
	# print(center)
	# cv2.waitKey(0)


	# M = cv2.getRotationMatrix2D(center, 180-angle2, 1.0)
	# print(M)
	# rotated = cv2.warpAffine(original_array/4, M, (0,0))
	# rotated = rotated*4 
	# rotated = rotated.astype(int)
	# print(rotated)




# 程式起點
if __name__ == '__main__':
	main()


