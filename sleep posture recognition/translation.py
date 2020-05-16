'''

editor: Jones
date: 2019/07/29
content: 
1. 平移

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd


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

def head_middle(my_array):

	head_array = my_array[:7]
	middle_array = my_array[7:13]
	foot_array = my_array[14:]

	# print('head_array =', head_array)
	# print('middle_array =', middle_array)
	# print('foot_array =', foot_array)

	distance = 7

	xg1, yg1 = center_of_gravity(head_array)
	xg2, yg2 = center_of_gravity(middle_array)
	xg3, yg3 = center_of_gravity(foot_array)
	yg2 = yg2 + distance
	yg3 = yg3 + 13

	# print('xg1 = %f, yg1 = %f' % (xg1, yg1))
	# print('xg2 = %f, yg2 = %f' % (xg2, yg2))
	# print('xg3 = %f, yg3 = %f' % (xg3, yg3))

	return xg1, yg1, xg2, yg2, xg3, yg3

# 向左平移1格
def left_roll_1(my_array):

	row = 20
	col = 11
	left_roll_1_array = my_array[::,1:]
	left_roll_1_array = np.insert(left_roll_1_array, col-1, 0, axis=1)
	return left_roll_1_array

# 向左平移2格
def left_roll_2(my_array):

	row = 20
	col = 11
	left_roll_2_array = my_array[::,2:]
	left_roll_2_array = np.insert(left_roll_2_array, col-2, 0, axis=1)
	left_roll_2_array = np.insert(left_roll_2_array, col-1, 0, axis=1)
	return left_roll_2_array

# 向右平移1格
def right_roll_1(my_array):

	row = 20
	col = 11
	right_roll_1_array = my_array[::,:-1]
	right_roll_1_array = np.insert(right_roll_1_array, 0, 0, axis=1)
	return right_roll_1_array

# 向右平移2格
def right_roll_2(my_array):

	row = 20
	col = 11
	right_roll_2_array = my_array[::,:-2]
	right_roll_2_array = np.insert(right_roll_2_array, 0, 0, axis=1)
	right_roll_2_array = np.insert(right_roll_2_array, 0, 0, axis=1)
	return right_roll_2_array

def translat_X_axis_center(my_array):

	# 寻找重心
	xg, yg = center_of_gravity(my_array) 
	# print('xg =', xg)
	# print('yg =', yg)

	if(round(xg) == 2):
		print('xg = 2')
	elif(round(xg) == 8):
		print('xg = 8')

	# 若 xg = 5, 則維持不變
	if(round(xg) == 5):
		# print(my_array)
		return my_array
	# 若 xg = 4, 则右移1格
	elif(round(xg) == 4):
		right1_array = np.roll(my_array, 1, axis = 1)
		# print(right1_array)
		return right1_array
	# 若 xg = 3, 则右移1格
	elif(round(xg) == 3):
		right2_array = np.roll(my_array, 2, axis = 1)
		# print(right2_array)
		return right2_array
	# 若 xg = 6, 则左移1格
	elif(round(xg) == 6):
		left1_array = np.roll(my_array, -1, axis = 1)
		# print(left1_array)
		return left1_array
	# 若 xg = 7, 则左移2格
	elif(round(xg) == 7):
		left2_array = np.roll(my_array, -2, axis = 1)
		# print(left2_array)
		return left2_array

def image_translation(my_array):

	# translation_array = []

	left_roll_1_array = left_roll_1(my_array).ravel()
	left_roll_2_array = left_roll_2(my_array).ravel()
	right_roll_1_array = right_roll_1(my_array).ravel()
	right_roll_2_array= right_roll_2(my_array).ravel()

	translation_array = np.vstack((my_array.ravel(), left_roll_1_array, left_roll_2_array, right_roll_1_array, right_roll_2_array))
	return translation_array

def label_lying_data(my_array):

	columns = []
	i = 0
	while i < 220:
		col = 'pixel%d' %i
		i = i + 1
		columns.append(col)

	df = pd.DataFrame(my_array, columns = columns)

	# # label_0 正躺
	# label_0 = np.zeros(500, dtype=int)
	# # label_` 右側躺
	# label_1 = np.ones(500, dtype=int)
	# # label_2 左側躺
	# label_2 = np.ones(500, dtype=int) * 2
	# # label_3 趴睡
	# label_3 = np.ones(500, dtype=int) * 3

	# label = np.hstack((label_0, label_1, label_2, label_3))
	# print('label =', label)
	# df.insert(220, "label", label, True)
	print(df.head())

	return df


# open file 
# 400 frame 子暘，育誠，佳芸，建宇,思琪，竑量，皓菘，耀堂，沛忱，沛臻，嘉宏，燕鴻，顯郡，郅博，翊嘉，秉渝，鵬翔,Jones
lying_data = pd.read_csv("csvData/rawData/Jane_face_left.csv")
lying_data = lying_data.to_numpy()

row = 20
col = 11
# face_up = lying_data[:100]
# face_right = lying_data[100:200]
# face_left = lying_data[200:300]
# face_down = lying_data[300:]
# original_array = face_down[-1][:220].reshape(row, col)
# print(original_array)
# xg, yg = center_of_gravity(original_array)
# print('xg = %s yg = %s'%(xg, yg))
# xg1, yg1, xg2, yg2, xg3, yg3 = head_middle(original_array)

translation_array = []

for item in lying_data[::,:-1]:
	item_array = item.reshape(row, col)
	translat_X_axis_center_item = translat_X_axis_center(item_array)
	translation_item = image_translation(translat_X_axis_center_item)
	for j in range(5):
		translation_array.append(translation_item[j])

translation_array = np.array(translation_array)
print('len =', len(translation_array))
df_translation = label_lying_data(translation_array)
df_translation.to_csv ("csvdata/translateData/Jane_face_left_tran.csv" , index = False, encoding = "utf-8")



# 顯示圖形
# fig, ax = plt.subplots()
# plt.title('Sleeping Positions')
# plt.imshow(original_array, cmap=plt.cm.jet)
# plt.yticks(np.arange(0,20,1.0))
# plt.xticks(np.arange(0,11,1.0))
# plt.scatter([xg,],[yg,], marker = '*', color = 'green', s = 80)
# plt.scatter([xg1,],[yg1,], marker = '*', color = 'green', s = 50)
# plt.scatter([xg2,],[yg2,], marker = '*', color = 'orange', s = 50)
# plt.scatter([xg3,],[yg3,], marker = '*', color = 'pink', s = 50)
# plt.show()