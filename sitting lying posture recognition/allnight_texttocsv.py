'''
editor: Jones
date: 2019/07/29
content: 
1. txt to csv
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import cv2

# 主程序
def main():
	# open file 
	f = open('data_txt/allnight data/Jane0411.txt', 'r',encoding = 'utf-8')

	raw_data_list = []
	for x in f:
		raw_data_list.append(x)
	# print(len(raw_data_list))

	allnight_list = []
	for item in raw_data_list[1:]:
		j = np.array(item.split(),dtype = int)[:220]
		allnight_list.append(j)

	# allnight_array = np.array(allnight_list)
	# print(len(allnight_array))

	count0 = 0
	count1 = 0
	allnight_array = []
	row = 20
	col = 11

	while count0 < len(allnight_list):
		item_array = allnight_list[count0]
		item_array = item_array.reshape(20,11)
		mat90 = np.rot90(item_array, 1)
		mat180 = np.rot90(mat90, 1)
		rotate_item_array = mat180.flatten()

		allnight_array.append(rotate_item_array)
		count0 = count0 + 4

	print(len(allnight_array))


	# split_list = []

	# for row_index, row_element in enumerate(allnight_array):
	# 	for col_index, col_element in enumerate(row_element):
	# 		if col_element > 1023:
	# 			print('row_index =', row_index)
	# 			split_list.append(row_index)
	# 			break


	# print(split_list)
	# print(len(allnight_array))
	# top_allnight_array = allnight_array[0:split_list[0]]
	# down_allnight_array = allnight_array[split_list[-1]:]
	# allnight_array = np.append(top_allnight_array, down_allnight_array, axis=0)
	# print('allnight_array =', len(allnight_array))

	columns = []
	i = 0
	while i < 220:
		col = 'pixel%d' %i
		i = i + 1
		columns.append(col)

	df = pd.DataFrame(allnight_array, columns = columns)
	# print(df.head)
	df.to_csv("data_csv/allnightData/Jane0411_allnight.csv" , encoding = "utf-8")

	# row = 20
	# col = 11
	# original_array = allnight_array[487].reshape(row, col)
	# original_array2 = allnight_array[488].reshape(row, col)
	# print('sum = ', np.sum(original_array))
	# print(original_array)
	# print(original_array2)

	# 0-298 離床
	# 298-668 坐着
	# 668-680 離床
	# 681-707 坐著
	# 708-720 離床
	# 754-773 坐著
	# 6495-6706 離床

	# 6795-6706 離床


	# 顯示圖形
	# fig, ax = plt.subplots()
	# plt.title('Positions')
	# plt.imshow(original_array, cmap = plt.cm.jet)
	# plt.yticks(np.arange(-0.5,20.5,1.0))
	# plt.xticks(np.arange(-0.5,11.5,1.0))
	# plt.show()


	# fig, ax = plt.subplots()
	# plt.title('resize')
	# plt.imshow(img_200, cmap = plt.cm.jet)
	# # plt.colorbar()
	# plt.show()



# 程式起點
if __name__ == '__main__':
	main()
