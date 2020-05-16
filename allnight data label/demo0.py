'''
editor: Jones
date:2020/05/07
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


def read_csv_func(file_name):

	data_frame = pd.read_csv(file_name)

	sleep_posture = data_frame[['sleep posture']]
	sleep_posture_label_list = sleep_posture.to_numpy().ravel()

	# print(sleep_posture_label_list)
	label0 = 0
	label1 = 0
	label2 = 0

	for i in sleep_posture_label_list:
		if i == 0:
			label0 = label0 + 1
		elif i == 1:
			label1 = label1 + 1
		elif i == 2:
			label2 = label2 + 1

	print('label0 =', label0)
	print('label1 =', label1)
	print('label2 =', label2)

	face_left_list = []
	face_right_list = []
	face_up_list = []


	raw_data = data_frame.to_numpy()

	for i in raw_data:
		if i[-1] == 1:
			face_right_list.append(i)
		elif i[-1] == 2:
			face_left_list.append(i)
		elif i[-1] == 0:
			face_up_list.append(i)
		else:
			pass

	return np.array(face_left_list)

	# new_face_left_list = []
	# step = 0
	# while step < len(face_left_list):
	# 	new_face_left_list.append(step)
	# 	step = step + 3
	# print('left =', len(new_face_left_list))

	# new_face_right_list = []

	# count = 0
	# while count < len(face_right_list):
	# 	new_face_right_list.append(count)
	# 	count = count + 3
	# print('right =', len(new_face_right_list))


	# return raw_data

# file_name_1 = 'allnightData/Jones0420.csv'
# file_name_2 = 'allnightData/Jones0421.csv'
# file_name_3 = 'allnightData/Jones0430.csv'
# print('Hello world.')
# Jones0420_face_up_list = read_csv_func(file_name_1)
# Jones0421_face_up_list = read_csv_func(file_name_2)
# Jones0430_face_up_list = read_csv_func(file_name_3)

# Jones0420_df = pd.DataFrame(Jones0420_face_up_list)
# Jones0421_df = pd.DataFrame(Jones0421_face_up_list)
# Jones0430_df = pd.DataFrame(Jones0430_face_up_list)

# face_up_df = Jones0420_df.append(Jones0421_df, ignore_index = True)
# face_up_df = face_up_df.append(Jones0430_df, ignore_index = True)
# print(face_up_df.head)

# face_up_df.to_csv(r'allnightData\Jones_face_left.csv', index = False)



# print(raw_data)
# leave_bed_list = []

# for row_index, row_element in enumerate(raw_data):
# 	if np.sum(row_element) < 1024:
# 		# print(row_index)
# 		leave_bed_label = 1
# 		leave_bed_list.append(leave_bed_label)
# 	else:
# 		leave_bed_label = 0
# 		leave_bed_list.append(leave_bed_label)


# file_name_1 = 'allnightData/Jones_sleep_posture/Jones_face_left.csv'
# data_frame = pd.read_csv(file_name_1)
# raw_data = data_frame.to_numpy()
# print(len(raw_data))

# new_face_left_list = []
# count = 0
# while count < len(raw_data):
# 	new_face_left_list.append(raw_data[count])
# 	count = count + 3

# step = 1
# while step < len(raw_data):
# 	new_face_left_list.append(raw_data[step])
# 	step = step + 3

# size = 2
# while size < len(raw_data):
# 	new_face_left_list.append(raw_data[size])
	# size = size + 3


# print(len(new_face_left_list))
# face_left_df = pd.DataFrame(new_face_left_list[:9000])
# print(face_left_df)
# face_left_df.to_csv(r'Jones_face_left.csv', index = False)

file_name_1 = 'Jones_face_up.csv'
file_name_2 = 'Jones_face_right.csv'
file_name_3 = 'Jones_face_left.csv'

face_up_data_frame = pd.read_csv(file_name_1)
face_right_data_frame = pd.read_csv(file_name_2)
face_left_data_frame = pd.read_csv(file_name_3)

Jones_lying_df = face_up_data_frame.append(face_right_data_frame, ignore_index = True)
Jones_lying_df = Jones_lying_df.append(face_left_data_frame, ignore_index = True)
print(Jones_lying_df)
Jones_lying_df.to_csv(r'Jones_lying.csv', index = False)