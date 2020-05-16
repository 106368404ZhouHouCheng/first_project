'''
editor: Jones
date:2020/04/19
content:
1. read allnight sleep data
2. data save to csv file type
3. allnight data label
4. leave bed, 
5. sitting lying difference,
6. sleep posture  
'''

import numpy as np 
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt 


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



file_name = 'allnightData/Jones0430.csv'
data_frame = pd.read_csv(file_name)
# print(data_frame.head())
raw_data = data_frame.to_numpy()
# print(raw_data)
leave_bed_list = []

for row_index, row_element in enumerate(raw_data):
	if np.sum(row_element) < 1024:
		# print(row_index)
		leave_bed_label = 1
		leave_bed_list.append(leave_bed_label)
	else:
		leave_bed_label = 0
		leave_bed_list.append(leave_bed_label)

# print(leave_bed_list)

# for i in leave_bed_list:
# 	print(i)

leave_bed_array = raw_data[508][:220].reshape(20,11)
print(leave_bed_array)
print(np.sum(leave_bed_array))

x_axis_mc, y_axis_mc = two_dimension_center_of_mass(leave_bed_array)
print(x_axis_mc)
print(y_axis_mc)

x_axis_mc_list = []
y_axis_mc_list = []

for index, element in enumerate(raw_data[246:20884]):
	element_array = element[:220].reshape(20,11)
	x_axis_mc, y_axis_mc = two_dimension_center_of_mass(element_array)
	x_axis_mc_list.append(x_axis_mc)
	y_axis_mc_list.append(y_axis_mc)



# print(mc_list)
# mc_array = np.array(mc_list)
# print(mc_array)


plt.figure()
plt.imshow(leave_bed_array, cmap='jet')
plt.scatter(x_axis_mc, y_axis_mc, marker='^', color = 'red',linewidths  = 3)
plt.show()


# file_name = 'allnightData/Jane0429.csv'
# data_frame = pd.read_csv(file_name)
# print(data_frame.head())
# raw_data = data_frame[['sleep posture']].to_numpy()
# # print(raw_data)
# state_change_index_list = []
# state_change_list = []
# state_change_list2 = []

# count = 1

# while count < len(raw_data):
# 	diff = raw_data[count][0] - raw_data[count-1][0]
# 	if diff != 0:
# 		print(count)
# 		state_change_index_list.append(count)
# 		state_change_list.append(raw_data[count-1][0])
# 		state_change_list2.append(raw_data[count][0])
# 		print(raw_data[count-1])
# 		print(raw_data[count])
# 	count = count + 1

# print(state_change_index_list)
# print(range(len(state_change_list)))
# print(state_change_list)
# print(state_change_list2)

# print(np.diff(np.array(state_change_index_list)))

