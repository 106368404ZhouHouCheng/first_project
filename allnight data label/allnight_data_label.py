import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # 主程序
def main():
	# open file 
	raw_data_df = pd.read_csv('Jones0430.csv')
	raw_data_array = raw_data_df.to_numpy()
	# print(raw_data_array)

	row = 20
	col = 11
	stop_index = 220
	fig_number = 15
	count = 186
	start = 0
	stop = 15

	start = start + count * fig_number
	stop = stop + count * fig_number

	print('start =', start)
	print('stop =', stop)

	start_stop_array = raw_data_array[start:stop]

	count0 = 0
	count1 = 1
	count2 = 2
	count3 = 3
	count4 = 4
	count5 = 5
	count6 = 6 
	count7 = 7
	count8 = 8
	count9 = 9
	count10 = 10 
	count11 = 11
	count12 = 12
	count13 = 13
	count14 = 14

	original_array0 = start_stop_array[count0][:stop_index].reshape(row, col)
	original_array1 = start_stop_array[count1][:stop_index].reshape(row, col)
	original_array2 = start_stop_array[count2][:stop_index].reshape(row, col)
	original_array3 = start_stop_array[count3][:stop_index].reshape(row, col)
	original_array4 = start_stop_array[count4][:stop_index].reshape(row, col)
	original_array5 = start_stop_array[count5][:stop_index].reshape(row, col)
	original_array6 = start_stop_array[count6][:stop_index].reshape(row, col)
	original_array7 = start_stop_array[count7][:stop_index].reshape(row, col)
	original_array8 = start_stop_array[count8][:stop_index].reshape(row, col)
	original_array9 = start_stop_array[count9][:stop_index].reshape(row, col)
	original_array10 = start_stop_array[count10][:stop_index].reshape(row, col)
	original_array11 = start_stop_array[count11][:stop_index].reshape(row, col)
	original_array12 = start_stop_array[count12][:stop_index].reshape(row, col)
	original_array13 = start_stop_array[count13][:stop_index].reshape(row, col)
	original_array14 = start_stop_array[count14][:stop_index].reshape(row, col)
	print(original_array12)
	print(original_array13)
	print(original_array14)
	# print(original_array4)
	# print(original_array6)

	# # 顯示圖形
	plt.figure()
	plt.subplot(3,5,1) # 第一行的左圖
	plt.title(start)
	plt.imshow(original_array0 , cmap=plt.cm.jet)

	plt.subplot(3,5,2) # 第一行的右圖
	plt.title(start+1)
	plt.imshow(original_array1 , cmap=plt.cm.jet)

	plt.subplot(3,5,3) # 第二行的左圖
	plt.title(start+2)
	plt.imshow(original_array2 , cmap=plt.cm.jet)

	plt.subplot(3,5,4) # 第二行的右圖
	plt.title(start+3)
	plt.imshow(original_array3 , cmap=plt.cm.jet)

	plt.subplot(3,5,5) # 第二行的右圖
	plt.title(start+4)
	plt.imshow(original_array4 , cmap=plt.cm.jet)

	plt.subplot(3,5,6) # 第二行的右圖
	plt.title(start+5)
	plt.imshow(original_array5 , cmap=plt.cm.jet)

	plt.subplot(3,5,7) # 第二行的右圖
	plt.title(start+6)
	plt.imshow(original_array6 , cmap=plt.cm.jet)

	plt.subplot(3,5,8) # 第二行的右圖
	plt.title(start+7)
	plt.imshow(original_array7 , cmap=plt.cm.jet)

	plt.subplot(3,5,9) # 第二行的右圖
	plt.title(start+8)
	plt.imshow(original_array8 , cmap=plt.cm.jet)

	plt.subplot(3,5,10) # 第二行的右圖
	plt.title(start+9)
	plt.imshow(original_array9 , cmap=plt.cm.jet)

	plt.subplot(3,5,11) # 第二行的右圖
	plt.title(start+10)
	plt.imshow(original_array10 , cmap=plt.cm.jet)

	plt.subplot(3,5,12) # 第二行的右圖
	plt.title(start+11)
	plt.imshow(original_array11 , cmap=plt.cm.jet)

	plt.subplot(3,5,13) # 第二行的右圖
	plt.title(start+12)
	plt.imshow(original_array12 , cmap=plt.cm.jet)

	plt.subplot(3,5,14) # 第二行的右圖
	plt.title(start+13)
	plt.imshow(original_array13 , cmap=plt.cm.jet)

	plt.subplot(3,5,15) # 第二行的右圖
	plt.title(start+14)
	plt.imshow(original_array14 , cmap=plt.cm.jet)

	plt.show()


	# f = open('textData/Jane0411.txt', 'r',encoding = 'utf-8')

	# raw_data_list = []
	# for x in f:
	# 	raw_data_list.append(x)
	# # print(len(raw_data_list))

	# allnight_list = []
	# for item in raw_data_list[1:]:
	# 	j = np.array(item.split(),dtype = int)[:220]
	# 	allnight_list.append(j)

	# allnight_array = np.array(allnight_list)
	# print(len(allnight_array))

	# original_array = allnight_array[24000].reshape(20, 11)
	# print(original_array)

	# mat90 = np.rot90(original_array, 2)
	# print(mat90)



# 程式起點
if __name__ == '__main__':
	main()