'''

editor: Jones
date: 20200220
content: 5個特徵值
1.計算非零壓力值的加權平均值
2.計算非零壓力值的變異數
3.找垂直轴重心，垂直軸的變異數
4.以重心為圓心，半徑為1.0，以此遞增，所圍成的圓，計算圓中壓中的壓力值*個數與壓力總和的比例，找合適半徑
5.以重心為圓心，半徑為1.0，以此遞增，所圍成的圓，計算圓中壓中的壓力值點數與總的壓力值點數的比例
6.将上述特徵值放進支撐向量機進行訓練，用於區分坐臥姿態

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import math
import seaborn as sns
from scipy.stats import norm
from base_function_package.base_function import thresholding, binarization, nonzero_pressure_value, center_of_mass, two_dimension_center_of_mass
from sklearn.model_selection import train_test_split

from sklearn import datasets
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC



# 計算非零壓力值的加權平均值
def nonzero_average_pressure_value(nonzero_array):

	j = 0
	average_pres_val = 0
	size = len(nonzero_array)
	while j < size:
		average_pres_val = average_pres_val + (1/size * nonzero_array[j])
		j = j + 1
	return average_pres_val


# 計算非零壓力值的變異數
def nonzero_variance_pressure_value(nonzero_array, average_pres_val):

	variance_1D = 0
	size = len(nonzero_array)
	for k in nonzero_array:
		variance_1D = variance_1D + (1/size * (k - average_pres_val) ** 2)
	return variance_1D


# 水平軸的變異數
def variance_x_axis(my_array):

	# 垂直軸總和
	x_axis_array = np.sum(my_array, axis = 0)
	# print(x_axis_array)
	weight_x_axis_sum = 0
	for x, x_axis_val in enumerate(x_axis_array):
		weight_x_axis_sum = weight_x_axis_sum + x * x_axis_val

	xg = weight_x_axis_sum/np.sum(x_axis_array)

	var_x_axis = 0
	for x, x_axis_val in enumerate(x_axis_array):
		var_x_axis = var_x_axis + (x_axis_val / np.sum(x_axis_array) * (x - xg) ** 2)
	return var_x_axis


# 垂直軸的變異數
def variance_y_axis(my_array):

	# 水平軸總和
	y_axis_array = np.sum(my_array, axis = 1)
	# print(y_axis_array)

	weight_y_axis_sum = 0
	for y, y_axis_val in enumerate(y_axis_array):
		weight_y_axis_sum = weight_y_axis_sum + y * y_axis_val

	yg = weight_y_axis_sum/np.sum(y_axis_array)

	var_y_axis = 0
	for y, y_axis_val in enumerate(y_axis_array):
		var_y_axis = var_y_axis + (y_axis_val / np.sum(y_axis_array) * (y - yg) ** 2)
	return var_y_axis


# 以重心為圓心，半徑為1.0，以此遞增，所圍成的圓，計算圓中壓中的壓力值點數與總的壓力值點數的比例
def ratio_of_points(my_array):

	# print(my_array)

	xg, yg =  two_dimension_center_of_mass(my_array)
	# print('xg1 =', xg)
	# print('yg1 =', yg)
	binary_array = binarization(my_array)
	# print(binary_array)

	distance_list = []
	row = 20
	col = 11
	radius_1 = 1.0

	for row_index, row_element in enumerate(binary_array):
		for col_index, col_element in enumerate(row_element):
			if(col_element == 1023):
				distance = np.sqrt((xg - col_index) ** 2 + (yg - row_index) ** 2)
				# print('distance =', distance)
				distance_list.append(distance)

	distance_array = np.sort(np.array(distance_list))
	ratio_list = []
	count = 0

	while count <= math.ceil(max(distance_list)):
		ratio = np.size(distance_array[distance_array < radius_1 * count])/np.size(distance_array)
		ratio_list.append(round(ratio, 3))
		count += 1

	return ratio_list


# 以重心為圓心，半徑為1.0，以此遞增，所圍成的圓，計算圓中壓中的壓力值*個數與壓力總和的比例，找合適半徑
def ratio_of_pressure_value(my_array):

	xg, yg =  two_dimension_center_of_mass(my_array)
	# print('xg2 =', xg)
	# print('yg2 =', yg)

	distance_list = []
	pressure_value_list = []
	row = 20
	col = 11
	radius_1 = 1.0
	count = 0

	for row_index, row_element in enumerate(my_array):
		for col_index, col_element in enumerate(row_element):
			if(col_element > 0):
				distance = np.sqrt((xg - col_index) ** 2 + (yg - row_index) ** 2)
				distance_list.append(distance)
				pressure_value_list.append(col_element)

	# print('distance_list =', distance_list)

	pres_val_ratio_list = []
	pres_val_ratio = 0
	# print('pressure distance max =', math.ceil(max(distance_list)))
	# print('pressure max =', max(distance_list))

	while count <= math.ceil(max(distance_list)):
		i = 0
		while i < len(distance_list):
			if (distance_list[i] > radius_1 * (count-1)) and (distance_list[i] < radius_1 * count):
				pres_val_ratio = pressure_value_list[i] + pres_val_ratio
			i = i+1
		value_ratio = pres_val_ratio / np.sum(my_array)
		pres_val_ratio_list.append(round(value_ratio, 3))
		count = count + 1

	return pres_val_ratio_list


def center_of_gravity_pressure_value(my_array):

	xg, yg =  two_dimension_center_of_mass(my_array)
	xg_bottom = int(xg)
	xg_top = math.ceil(xg)
	yg_bottom = int(yg)
	yg_top = math.ceil(yg)

	print('xg_bottom =', xg_bottom)
	print('xg_top =', xg_top)
	print('yg_bottom =', yg_bottom)
	print('yg_top =', yg_top)

	f_Q11 = my_array[yg_bottom][xg_bottom]
	f_Q12 = my_array[yg_bottom][xg_top]
	f_Q21 = my_array[yg_top][xg_bottom]
	f_Q22 = my_array[yg_top][xg_top]
	print(f_Q11)
	print(f_Q12)
	print(f_Q21)
	print(f_Q22)


	A = np.array([xg_top - xg, xg - xg_bottom])
	B = np.array([[f_Q11, f_Q12],
				[f_Q21, f_Q22]])
	C = np.array([[yg_top - yg],
				[yg - yg_bottom]])
	print(A)
	print(B)
	print(C)

	# center_of_gravity_pressure_value
	cg_pres_val = A.dot(B)
	print(cg_pres_val)
	cg_pres_val = cg_pres_val.dot(C)
	print(cg_pres_val)
	cg_pres_val = round(cg_pres_val[0])

	return cg_pres_val


def ratio_index(ratio_list):

	ratio0_list = []
	ratio1_list = []
	ratio2_list = []
	ratio3_list = []

	# print(ratio_list[500:600])

	ratio = 0
	count = 0
	while count < len(ratio_list):
		ratio0_list.append(ratio_list[count][0])
		ratio1_list.append(ratio_list[count][1])
		ratio2_list.append(ratio_list[count][2])
		# ratio3_list.append(ratio_list[count][3])
		count += 1 

	# print(np.mean(np.array(ratio0_list)))
	# print('variance =', np.var(np.array(ratio0_list)))
	# print(np.mean(np.array(ratio1_list)))
	# print('variance =', np.var(np.array(ratio1_list)))
	# print(np.mean(np.array(ratio2_list)))
	# print('variance =', np.var(np.array(ratio2_list)))
	# print(np.mean(np.array(ratio3_list)))
	# print('variance =', np.var(np.array(ratio3_list)))

# 以重心為圓心，畫橢圓，定半徑

# 特徵值提取
def sitting_lying_feature_extraction(file_array):

	row = 20
	col = 11
	# 非零壓力值的加權平均值list
	average_pres_val_list = []
	# 非零壓力值變異數list
	variance_1D_list = []
	# 垂直軸變異數 list
	variance_y_axis_list = []
	# 定圓心，半徑內壓力總和/整張影像壓力總和 list
	ratio_of_pressure_value_list = []
	# 定圓心，半徑內壓力點數/整張影像壓力點數 list
	ratio_of_points_list = []

	# 提取特徵值
	for index, element in enumerate(file_array):
		item_array = np.array(element)[:220]
		item_array = item_array.reshape(row, col)
		threshold_array = thresholding(item_array)
		nonzero_item_array = nonzero_pressure_value(item_array)
		average_pres_val = nonzero_average_pressure_value(nonzero_item_array)
		variance_1D = nonzero_variance_pressure_value(nonzero_item_array, average_pres_val)
		var_y_axis = variance_y_axis(item_array)
		pres_val_ratio_list = ratio_of_pressure_value(item_array)
		ratio_list = ratio_of_points(item_array)

		average_pres_val_list.append(average_pres_val)
		variance_1D_list.append(variance_1D)
		variance_y_axis_list.append(var_y_axis)
		ratio_of_pressure_value_list.append(pres_val_ratio_list)
		ratio_of_points_list.append(ratio_list)


	# # 取半径为3
	pressure_value_index_list = []
	for index in ratio_of_pressure_value_list:
		# print('len =', len(index))
		if len(index) > 3:
			pressure_value_index_list.append(index[3])
		else:
			pressure_value_index_list.append(index[-1])

	# # # # 取半径为3
	point_index_list = []
	for points in ratio_of_points_list:
		# print('point =', len(points))
		if len(points) > 3:
			point_index_list.append(points[3])
		else:
			point_index_list.append(points[-1])

	d = {'average value': np.array(average_pres_val_list), 'variance 1D': np.array(variance_1D_list), 
			'y axis variance': variance_y_axis_list, 'pressure value ratio as 3': np.array(pressure_value_index_list), 
			'points ratio as 3': np.array(point_index_list)}

	df = pd.DataFrame(data = d)
	return df


# support vector machine
def support_vector_machine(test_feature_data):

	train_feature_data = pd.read_csv('input/20200203whole_feature_data_0213.csv')

	X = train_feature_data[['average value', 'variance 1D', 'y axis variance', 'pressure value ratio as 3', 'points ratio as 3']]
	y = train_feature_data[['target']]

	# 切分 train:test = 2:1 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
	Z = test_feature_data[['average value', 'variance 1D', 'y axis variance', 'pressure value ratio as 3', 'points ratio as 3']]

	# Normatlization
	sc = StandardScaler()
	sc.fit(X_train)
	X_train_std = sc.transform(X_train)
	X_test_std = sc.transform(X_test)
	Z_test_std = sc.transform(Z)

	# svm
	svm = SVC()
	clf = svm.fit(X_train_std, y_train)

	# 預測
	predict_result = svm.predict(X_test_std)
	really_result = y_test['target'].values
	print('clf:', clf.score(X_test_std, y_test))

	# 錯誤統計
	error = 0
	for i, v in enumerate(predict_result):
		if v != really_result[i]:
			error += 1
	print('error =',error)

	predict_result2 = svm.predict(Z_test_std)

	# print('clf:', clf.score(X_test_std, y_test))
	count0 = 0
	count1 = 0
	sitting_index_list = []
	lying_index_list = []
	sitting_lying_list = []
	# predict_result
	for index, element in enumerate(predict_result2):
		if element == 0:
			count0 += 1
			sitting_index_list.append(index)
			sitting_lying_list.append(0)
		elif element == 1:
			count1 += 1
			lying_index_list.append(index)
			sitting_lying_list.append(1)

	print('sitting time =',count0)
	print('lying time =', count1)
	print('lying_index_list size =', len(lying_index_list))
	# print('sitting_lying_list =', sitting_lying_list)

	return predict_result2, sitting_lying_list, lying_index_list

def leave_bed_state(raw_data):

	leave_bed_list = pd.DataFrame(raw_data, columns= ['leave bed']).to_numpy().flatten()
	print(leave_bed_list)

	leave_bed_list_diff = np.diff(leave_bed_list)

	row = 20
	col = 11

	leave_bed = 1
	no_leave_bed = 0
	number_of_time_left_bed = 0
	leave_bed_time_list = []

	for index, element in enumerate(leave_bed_list_diff):
		if element == 1:
			# print(index)
			leave_bed_start_time = index
		if element == -1:
			# print(index)
			leave_bed_stop_time = index

			leave_bed_time = leave_bed_stop_time - leave_bed_start_time
			# print('leave_bed_time =', leave_bed_time)
			leave_bed_time_list.append(leave_bed_time)
			number_of_time_left_bed = number_of_time_left_bed + 1
			# print('number_of_time_left_bed =', number_of_time_left_bed)

	return leave_bed_time_list, number_of_time_left_bed

# # 主程序
def main():
	# open file 
	sitting_lying_raw_data_df = pd.read_csv('data_csv/allnightData/Jane0411_allnight_sitting_lying.csv')
	# print(sitting_lying_raw_data_df.head)

	really_result = sitting_lying_raw_data_df['sitting lying posture'].to_numpy()
	sitting_lying_raw_data_df = sitting_lying_raw_data_df.to_numpy()
	sittting_lying_feature_test_df = sitting_lying_feature_extraction(sitting_lying_raw_data_df)
	predict_result, sitting_lying_list, lying_index_list = support_vector_machine(sittting_lying_feature_test_df)

	# 錯誤統計
	error = 0
	for i, v in enumerate(predict_result):
		if v != really_result[i]:
			error += 1
	print('error =',error)














# 程式起點
if __name__ == '__main__':
	main()





