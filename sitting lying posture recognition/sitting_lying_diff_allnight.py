'''
editor: Jones
date: 20200216
content: 
1. 找特徵
2. 4個特徵：
3. Y軸的變異數，
4. 平均壓力值，
5. 找重心，加半徑畫圓，圓内點數/總點數
6. 將特徵放到SVM去訓練，用來區分坐姿和睡姿
新增： 將壓力影像轉直方圖，找直方圖的變異數，將其作為一個特徵值
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd 
import math
import seaborn as sns
from scipy.stats import norm
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import csv

from sitting_lying_diff import sitting_lying_feature_extraction, support_vector_machine
from image_rotation import head_foot_diff, image_rotation, rotation_angle
from sleep_position_recognition import predict_classification_results



# 整理分類結果，返回各種睡姿的睡覺時間
def organize_classification_results(results):

	face_up_count = 0
	face_right_count = 0
	face_left_count = 0
	face_down_count = 0
	unkown = 0

	for result in results:
		if result == 0:
			face_up_count = face_up_count + 1
		elif result == 1:
			face_right_count = face_right_count + 1 
		elif result == 2:
			face_left_count = face_left_count + 1
		elif result == 3: 
			face_down_count = face_down_count + 1
		else:
			print('unkown')
			unkown = unkown + 1

	print('face up time =', face_up_count)
	print('face right time =', face_right_count)
	print('face left time =', face_left_count)
	print('face_down_count =', face_down_count)

	# return face_up_count, face_right_count, face_left_count, face_down_count


# 整理分類結果，返回各種睡姿的睡覺時間
def organize_results(my_array, results):

	new_list = []

	for element in my_array:
		new_list.append(element.flatten())

	new_array = np.array(new_list)

	columns = []
	i = 0
	while i < 220:
		col = 'pixel%d' %i
		i = i + 1
		columns.append(col)

	df = pd.DataFrame(new_array, columns = columns)
	print(df.head)

	label = results.flatten()
	df.insert(220, "label", label, True)
	print(df.head)

	# df.to_csv("rotated array csv/Jones1224_rotated_array_label.csv" , encoding = "utf-8")

	# return df
	
def sleep_quality_index(results):

	# 臥床時間
	lying_time = len(results)
	# 整晚翻身次數
	number_of_changes = 0
	change_sleep_position_time = 0

	count = 1
	while count < len(results):
		if results[count-1] != results[count]:
			number_of_changes = number_of_changes + 1
		elif results[count-1] == results[count] and results[count-1] == 0:
			change_sleep_position_time + 1

		count = count + 1

	print('lying_time =', lying_time)
	print('number_of_changes =', number_of_changes)

	return lying_time, number_of_changes


# 主程序
def main():

	# test data
	test_data = pd.read_csv('data_csv/allnightData/Jane0411_allnight.csv')
	test_array = test_data.to_numpy()
	print(len(test_array))

	# 非坐姿和躺姿index 0001 0001 
	count17 = 0
	# 坐姿和躺姿index 0000 0001 
	count1 = 0
	# 非坐姿和躺姿 index list
	count17_index_list = []
	# 坐姿和躺姿  index list
	sitting_lying_index_list = []
	# 坐姿和躺姿 list
	sitting_lying_data_list = []

	for index, element in enumerate(test_array):
		if np.sum(element) <= 1023:
			count17 += 1
			count17_index_list.append(index)
		elif np.sum(element) > 1023:
			count1 += 1
			sitting_lying_index_list.append(index)
			sitting_lying_data_list.append(element)

	print('count17 =', count17)

	# 坐臥姿態特徵提取
	test_feature_data = sitting_lying_feature_extraction(sitting_lying_data_list)
	# 將上述的特徵值放進SVM test, return lying list
	lying_index_list = support_vector_machine(test_feature_data)
	print(lying_index_list[0])

	# head_foot_list = []

	# rotated_list = []

	# item_zeros_array = np.zeros((20,11))

	# index = 0
	# while index < len(lying_index_list):
	# 	item_array = sitting_lying_data_list[lying_index_list[index]]
	# 	item_lying_array = item_array.reshape(20,11)
	# 	head_foot_item_array = head_foot_diff(item_lying_array)
	# 	head_foot_list.append(head_foot_item_array)
	# 	angle = rotation_angle(head_foot_item_array)
	# 	# print('angle =', angle)
	# 	item_rotated_array = image_rotation(head_foot_item_array, angle)
	# 	rotated_list.append(item_rotated_array)
	# 	index += 1

	# head_foot_array = np.array(head_foot_list)
	# rotated_array = np.array(rotated_list)

	# # 預測分類結果
	# predict_results = predict_classification_results(rotated_array)
	# print(predict_results[0])
	# organize_classification_results(predict_results)
	# lying_time, number_of_changes = sleep_quality_index(predict_results)

	# print(head_foot_array[0])
	
	# # 分類結果整理匯集 csv
	# organize_results(head_foot_array, predict_results)

	# label_list = []

	# for result in predict_results:
	# 	if result == 0:
	# 		label_list.append('Face Up')
	# 	elif result == 1:
	# 		label_list.append('Face Right')
	# 	elif result == 2:
	# 		label_list.append('Face Left')
	# 	elif result == 3: 
	# 		label_list.append('Face Down')

	


	# Jones1224_allnight
	# 0-297 sum = 0 無人在床
	# 298-667 坐姿
	# 668-680 由坐轉躺過程
	# 681-6494 躺姿
	# 6495-6706 離床
	# 6707-6715 準備入床
	# 6716- 躺姿

	# print(sitting_lying_index_list[0])
	# print((test_array[299] == sitting_lying_data_list[1]).all())

	# yes = 0
	# no = 0

	# for index, element in enumerate(sitting_lying_data_list[:469]):
	# 	if ((element == test_array[index + 298]).all()):
	# 		yes += 1
	# 	else:
	# 		no += 1

	# print('yes =', yes)
	# print('no =', no)


	# sitting_lying_data_list sitting 0-469, 6091-6189
	# original_array = sitting_lying_data_list[6190].reshape(20,11)
	# print(np.sum(original_array))
	# print(original_array)

	# 顯示圖形
	# fig, ax = plt.subplots()
	# plt.title('Positions')
	# plt.plot(range(len(label_list)), label_list)
	# plt.show()



# 程式起點
if __name__ == '__main__':
	main()  





