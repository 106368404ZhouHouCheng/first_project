'''
editor:Jones
date:2020/3/2
content:
'''
import pandas as pd
import numpy as np


# 主程序
def main():

	sitting_data_path = 'csvdata/20200113 csvData/20200113whole_sitting.csv'
	sitting_data =  pd.read_csv(sitting_data_path).to_numpy()

	face_up_data_path = 'csvdata/20200113 csvData/20200113wholeFaceUp.csv'
	face_up_data = pd.read_csv(face_up_data_path).to_numpy()

	face_right_data_path = 'csvdata/20200113 csvData/20200113wholeFaceRight.csv'
	face_right_data = pd.read_csv(face_right_data_path).to_numpy()

	face_left_data_path = 'csvdata/20200113 csvData/20200113wholeFaceLeft.csv'
	face_left_data = pd.read_csv(face_left_data_path).to_numpy()

	face_down_data_path = 'csvdata/20200113 csvData/20200113wholeFaceDown.csv'
	face_down_data = pd.read_csv(face_down_data_path).to_numpy()


	# sitting data
	sitting_threshold_list = []

	for i in sitting_data:
		item_array = (i[:-1]).reshape(20,11)
		threshold = np.sum(item_array)
		sitting_threshold_list.append(threshold)

	sitting_threshold_array = np.array(sitting_threshold_list)
	print(np.min(sitting_threshold_array))

	# face up data
	face_up_threshold_list = []

	for i in face_up_data:
		item_array = (i[:-1]).reshape(20,11)
		threshold = np.sum(item_array)
		face_up_threshold_list.append(threshold)

	face_up_threshold_array = np.array(face_up_threshold_list)
	print(np.min(face_up_threshold_array))

	#  face right data
	face_right_threshold_list = []

	for i in face_right_data:
		item_array = (i[:-1]).reshape(20,11)
		threshold = np.sum(item_array)
		face_right_threshold_list.append(threshold)

	face_right_threshold_array = np.array(face_right_threshold_list)
	print(np.min(face_right_threshold_array))

	# face left data
	face_left_threshold_list = []

	for i in face_left_data:
		item_array = (i[:-1]).reshape(20,11)
		threshold = np.sum(item_array)
		face_left_threshold_list.append(threshold)

	face_left_threshold_array = np.array(face_left_threshold_list)
	print(np.min(face_left_threshold_array))

	# face down data
	face_down_threshold_list = []

	for i in face_down_data:
		item_array = (i[:-1]).reshape(20,11)
		threshold = np.sum(item_array)
		face_down_threshold_list.append(threshold)

	face_down_threshold_array = np.array(face_down_threshold_list)
	print(np.min(face_down_threshold_array))



if __name__ == '__main__':
	main()

