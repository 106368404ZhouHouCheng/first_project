'''
editor: Jones
date: 2019/12/25
content: 
1.尋找合適的threshold
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd 
import math
import pandas as pd


def nonzero_array(my_array):

	my_array = my_array[np.nonzero(my_array)]
	return my_array


# 手動設定門檻值為50
def thresholding_50(threshold_array):

	min_threshold = 50
	# print(min_threshold)
	threshold_array[threshold_array <= min_threshold] = 0

	return threshold_array

# 計算非零壓中點數
def pressure_points(point_array):

	point = 0
	threshold = 2

	for i in point_array:
		for j in i:
			if(threshold < j):
				point = point + 1

	print('point =', point)

	return point

# 計算非零的平均壓力值
def nonzero_average_pressure(my_array):

	sum_ave = np.sum(my_array)
	print(sum_ave)

	nonzero_ave_val = np.sum(my_array) / pressure_points(my_array)
	return nonzero_ave_val


# 以所有的壓力值的加權平均值作為門檻值，低於門檻值則為0，高於門檻值則保持原本的數值
def weight_average_thresholding(threshold_array):

	hist, bin_edges = np.histogram(threshold_array.ravel(), bins = range(1024+1))
	print('hist =', len(hist))

	weight_sum = 0
	count = 0
	while count < hist.size:
		weight_sum += count * hist[count]
		count = count + 1
	weight_pre_average = weight_sum / threshold_array.size

	threshold_array[threshold_array < weight_pre_average] = 0

	return threshold_array

# # 高斯濾波
# def Gaussian_smoothing():
	

# otsu 
def otsu_threshold(threshold_array):

	hist, bin_edges = np.histogram(threshold_array.ravel(), bins = range(1024))
	print('hist =', hist)
	print('bin_edges =', bin_edges)

	print('len =', len(hist))
	print(type(bin_edges))

	weight_sum = 0
	count = 1
	while count < hist.size:
		weight_sum += count * hist[count]
		count = count + 1
	print('weight_sum =', weight_sum)

	weightB_sum = 0.0
	weightB = 0.0
	weightF = 0.0
	mB = 0.0
	mF = 0.0
	between = 0.0
	threshold1 = 0.0
	threshold2 = 0.0
	max_a = 0
	total = np.size(threshold_array)
	total = total-hist[0]

	i=1
	between_list = []

	while i < 1023:
		print('i =', i)
		print('element =', hist[i])
		weightB = weightB + hist[i]
		print('weightB =', weightB)

		if weightB == 0:
			continue
		weightF = total - weightB
		print('weightF =', weightF)

		if weightF == 0:
			break

		weightB_sum = weightB_sum + i * hist[i]
		print('weightB_sum =', weightB_sum)
		mB = weightB_sum / weightB
		mF = (weight_sum - weightB_sum) / weightF

		print('mB =', mB)
		print('mF =', mF)

		between = weightB * weightF * (mB-mF) * (mB-mF)

		print('between =', between)

		if(between >= max_a):
			threshold1 = i
			print('threshold1 =', threshold1)
			if(between > max_a):
				threshold2 = i
				print('threshold2 =', threshold2)

			max_a = between
			print('max_a =', max_a)
		i += 1
		between_list.append(between)

	print('threshold1 =', threshold1)
	print('threshold2 =', threshold2)

	print('(threshold1+threshold2)/2.0 =', (threshold1+threshold2)/2.0)
	print('between_list =', between_list)
	print(len(between_list))

	return (threshold1+threshold2)/2.0


# triangle algotighm
def triangle_threshold(my_array):

	my_array = my_array[np.nonzero(my_array > 0)]
	# 計算影像灰階直方圖
	hist, bin_edges = np.histogram(my_array.ravel(), bins = range(1,1025))
	print(hist[0])
	print(type(hist))
	print(len(hist))
	# 尋找直方圖中兩側邊界
	# 左邊界
	left_round = 0
	right_round = 0
	isflipped = False;
	i = 0
	j = 1022
	N = 1023

	while i < N:
		if hist[i] > 0:
			left_round = i
			break
		i = i + 1
	
	# 右邊界	
	while j < N:
		if hist[j] > 0:
			right_round = j
			break
		j = j - 1

	print('left_round =', left_round)
	print('right_round =', right_round)
	# 位置再移動一個步長，即為最左側位置
	if(left_round > 0):
		left_round -= 1

	# 位置再移動一個步長，即為最右側位置
	if(right_round < 1023):
		right_round += 1

	print('left_round =', left_round)
	print('right_round =', right_round)

	# 尋找直方圖中的最大值
	max_a = np.max(hist)
	print('max_a =', max_a)
	max_index = np.argmax(hist)
	print(max_index)

	# 檢測是否最大波峰在亮的一側，否則翻轉
	if abs(max_index - left_round) < abs(max_index - right_round):

		isflipped = True
		hist_array = np.flip(hist)
		print('hist =', hist_array)

		left_round = N - 1 - right_round
		max_index = N - 1 - max_index

		print('left_round =', left_round)
		print(max_index)

	# 計算閾值得到閾值T，如果翻轉則1023-T

	thresh = left_round
	dist = 0
	b = left_round - max_index

	left_round_1 = left_round + 1

	while left_round_1 <= max_index:
		tempdist = max_a * left_round_1 + b * hist[left_round_1]
		print('tempdist = ', tempdist)
		if(tempdist > dist):
			dist = tempdist
			thresh = left_round_1

		print('left_round_1 =', left_round_1)

		left_round_1 += 1

	thresh = thresh - 1
	print(thresh)

	if isflipped:
		thresh = N-1-thresh
	print('thresh =', thresh)


# Balanced histogram thresholding
def bht(hist, min_count = 1):

	n_bins = len(hist)
	print('n_bins =', n_bins)
	h_s = 0

	while hist[h_s] < min_count:
		h_s = h_s + 1
	print('h_s =', h_s)

	h_e = n_bins - 1
	while hist[h_e] < min_count:
		h_e = h_e - 1

	print('h_e =', h_e)
	# use mean intensity of histogram as center; alternatively: (h_s + h_e) / 2)
	hist_center = int(round(np.average(np.linspace(1, 1024, n_bins), weights = hist)))
	# weight in the left part
	w_left = np.sum(hist[h_s:hist_center]) 
	# weight in the right part
	w_right = np.sum(hist[hist_center:h_e+1])

	print('hist_center =', hist_center)
	print('w_left =', w_left)
	print('w_right =', w_right)

	print('--------------------------')

	while h_s < h_e:
		print(h_s)
		print('w_left_1 =', w_left)
		print('w_right_1 =', w_right)

		# left part became beavier
		if w_left > w_right:
			w_left = w_left - hist[h_s]
			h_s += 1
		else:
			w_right = w_right - hist[h_e]
			h_e -= 1

		print('w_left_1 =', w_left)
		print('w_right_1 =', w_right)

		print('h_s =', h_s)
		print('h_e =', h_e)
		# re-center the weighing scale
		new_center = int(round((h_e + h_s) / 2))
		print('new_center =', new_center)

		if new_center < hist_center:
			w_left = w_left - hist[hist_center]
			w_right = w_right + hist[hist_center]
		elif new_center > hist_center:
			w_left = w_left + hist[hist_center]
			w_right = w_right - hist[hist_center]

		print('w_left_2 =', w_left)
		print('w_right_2 =', w_right)

		hist_center = new_center
		print('hist_center=', hist_center)

		print('**************************')

	print('hist_center =', hist_center + 1)


# 直方圖等化 直接使用OpenCv equalizeHist
def cv2_equalizeHist(my_array):

	print(my_array)
	my_array = my_array[my_array > 0]
	my_array = my_array/4
	# print(my_array)
	input_array = my_array.astype(np.uint8)
	input_array = input_array[input_array > 0]
	print(input_array)
	print('mean =', np.mean(input_array))
	equ = cv2.equalizeHist(input_array)
	print(equ)

	hist, bin_edges = np.histogram(equ, bins = range(0,255))

	print('hist =', hist)

	print('mena_1 =', np.mean(equ))
	print(np.var(equ))
	print(np.std(equ))
	# print('bin_edges =',bin_edges)

def histogram_equlization(my_array):

	my_array = my_array.flatten()
	print(my_array)
	print(np.sort(my_array))

	my_array = my_array[my_array > 0]

	lut = np.zeros(1024, dtype = np.dtype)#创建空的查找表
	print(my_array)
	hist, bins = np.histogram(my_array,1024,[1,1024])
	print('hist =', hist)

	cdf = hist.cumsum()
	print('cdf =', cdf)
	print(len(cdf))
	print('cdf.min =', cdf.min())

	for i, v in enumerate(lut):
		lut[i] = round(1023 * (cdf[i] - cdf.min())/(cdf.max() - cdf.min()))
		# print(lut[i])
	print('lut =', lut)

	new_list = []

	j = 0
	while j < len(my_array):
		# print('j1 =', j)
		k = 0
		while k < 1024:
			# print('k =', k)
			if my_array[j] == k:
				# print('my_array[j] =', my_array[j])
				new_value = lut[k]
				# print('new_value =', new_value)
				new_list.append(new_value)
			k = k + 1
		j = j + 1
		

	print(new_list)
	# new_array = np.array(new_list, dtype = int).reshape(20,11)
	new_array = np.array(new_list, dtype = int)

	hist, bin_edges = np.histogram(new_array, bins = range(0, 1024))
	print('hist =', hist)

	print(np.mean(new_array))
	print(np.var(new_array))
	print(np.std(new_array))

	return new_array


# 權重比例
def weighting_sum_ratio(my_array):

	# my_array = my_array[my_array > 0]
	# print(np.sort(my_array))
	hist, bin_edges = np.histogram(my_array, bins = range(0, 1025))
	# print('sum =',np.sum(my_array))

	i = 0
	weight_sum = 0
	while i < len(hist):
		weight_sum = hist[i] * i + weight_sum
		i += 1
	# print('weight_sum =', weight_sum)

	threshold_ratio = 0.997
	add_value = 0.01
	threshold_list = []

	while threshold_ratio > 0.89:
		weight_small = weight_sum * threshold_ratio
		j = 1023
		threshold = 0
		weight = 0
		# print('weight_small =', weight_small)

		while j > 0:
			weight = hist[j]*j + weight
			# print('weight =', weight)
			if weight >= weight_small:
				# print('weight2 =', weight)
				# print('j =', j)
				threshold = j
				break
			j = j - 1

		# print('threshold =',threshold)
		threshold_list.append(threshold)
		threshold_ratio = threshold_ratio - add_value

	return np.array(threshold_list)


# 主程序
def main():

	# open file 
	# f = open('textdata/1204Jones_sitting.txt', 'r',encoding = 'utf-8')
	# raw_data_list = []
	# for x in f:
	# 	raw_data_list.append(x)
	# print(len(raw_data_list[1:]) / 3600)

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
		threshold = weighting_sum_ratio(i[:220])
		# print(threshold)
		sitting_threshold_list.append(threshold)

	sitting_threshold_array = np.array(sitting_threshold_list)
	print(sitting_threshold_array[:10])

	# face up data
	face_up_threshold_list = []

	for i in face_up_data:
		threshold = weighting_sum_ratio(i[:220])
		# print(threshold)
		face_up_threshold_list.append(threshold)

	face_up_threshold_array = np.array(face_up_threshold_list)
	print(face_up_threshold_array)

	#  face right data
	face_right_threshold_list = []

	for i in face_right_data:
		threshold = weighting_sum_ratio(i[:220])
		# print(threshold)
		face_right_threshold_list.append(threshold)

	face_right_threshold_array = np.array(face_right_threshold_list)
	print(face_right_threshold_array)

	# face left data
	face_left_threshold_list = []

	for i in face_left_data:
		threshold = weighting_sum_ratio(i[:220])
		# print(threshold)
		face_left_threshold_list.append(threshold)

	face_left_threshold_array = np.array(face_left_threshold_list)
	print(face_left_threshold_array)

	# face down data
	face_down_threshold_list = []

	for i in face_down_data:
		threshold = weighting_sum_ratio(i[:220])
		# print(threshold)
		face_down_threshold_list.append(threshold)

	face_down_threshold_array = np.array(face_down_threshold_list)
	# print(face_down_threshold_array)

	sitting_df = pd.DataFrame(sitting_threshold_array)
	face_up_df = pd.DataFrame(face_up_threshold_array)
	face_right_df = pd.DataFrame(face_right_threshold_array)
	face_left_df = pd.DataFrame(face_left_threshold_array)
	face_down_df = pd.DataFrame(face_down_threshold_array)

	print(sitting_df)
	print(face_up_df)
	print(face_right_df)
	print(face_left_df)
	print(face_down_df)

	# sitting_df.to_excel("csvData/20200114wholeThreshold csv V1.1/20200114wholeSittingThreshold.xlsx")
	# face_up_df.to_excel("csvData/20200114wholeThreshold csv V1.1/20200114wholeFaceUpThreshold.xlsx")
	# face_right_df.to_excel("csvdata/20200114wholeThreshold csv V1.1/20200114wholeFaceRightThreshold.xlsx")
	# face_left_df.to_excel("csvdata/20200114wholeThreshold csv V1.1/20200114wholeFaceLeftThreshold.xlsx")
	# face_down_df.to_excel("csvdata/20200114wholeThreshold csv V1.1/20200114wholeFaceDownThreshold.xlsx")




	# original_array2 = sitting_data[99][:220].reshape(20,11)
	# print(original_array1.dtype)
	# original_array1 = original_array1.astype(np.uint8)
	# print(original_array1)

	# cv2_equalizeHist(original_array1)
	# new_array = histogram_equlization(original_array1)


	# print(original_array1.dtype)

	# equ = cv2.equalizeHist(original_array1)
	# print(equ)

	# otsu_threshold(original_array1)

	# triangle_threshold(original_array1)
	# print(len(original_array1))

	# original_array2 = original_array2[original_array2 > 3]
	# print(original_array2)

	# nonzero_original_array = nonzero_array(original_array1)
	# nonzero_original_array = original_array1[original_array1 > 0]
	# print('nonzero_original_array =', nonzero_original_array)

	# original_array2 = original_array1[original_array1>3]
	# print(original_array2)

	# print(np.sum(nonzero_original_array))
	# print(np.mean(nonzero_original_array))
	# print(np.var(nonzero_original_array))

	# print(len(nonzero_original_array))
	# print(np.sort(nonzero_original_array))

	# print('median = ', np.median(nonzero_original_array))

	# np_sort_array = np.sort(nonzero_original_array)


	# print(np_sort_array)

	# print(np.mean(np_sort_array))


	# hist, bin_edges = np.histogram(nonzero_original_array, bins = range(1,1024))

	# print('hist =', hist)
	# print('bin_edges =',bin_edges)

	# hist, bin_edges = np.histogram(original_array1, bins = range(0, 1024))
	# print('hist =', hist)
	# print(len(hist))
	# print('bin_edges =',bin_edges)

	# cdf = hist.cumsum()
	# print('cdf =', cdf)

	# cdf = (cdf - cdf[0] * 255 / (cdf[-1] - 1))
	# print(cdf)

	# img2 = np.zeros((20, 11, 1))
	# img2 = cdf[original_array1]

	# print(img2)

	# hist2, bins2 = np.histogram(img2, 1024)
	# print(hist2)

	# cdf2 = hist2.cumsum()
	# print(cdf2)

	# lut = np.zeros(256, dtype = original_array1.dtype)
	# print(lut)

	# bht(hist)

	# nonzero_count = 0

	# for i in hist:
	# 	if(i != 0):
	# 		print(i)
	# 		nonzero_count = i + nonzero_count

	# print(nonzero_count)



	# original_array1 = np.array(raw_data_list[0].split(),dtype = int)[:220].reshape(20,11)
	# original_array2 = np.array(raw_data_list[0].split(),dtype = int)[:220].reshape(20,11)

	# nonzero_ave_val = nonzero_average_pressure(original_array1)
	# print('nonzeor ave val =', nonzero_ave_val)

	# threshold_array = weight_average_thresholding(original_array1)
	# print('weight_pre_ave =', weight_pre_ave)

	# print(np.mean(original_array2))
	# triangle_threshold(original_array1)

	# otsu_threshold(original_array1)

	# print(np.sum(original_array))
	# print(np.mean(original_array))
	# weight_average_thresholding(original_array2)

	# print(np.sort(new_array.flatten()))

	# hist, bin_edges = np.histogram(new_array.flatten(), bins = np.arange(1024))
	# print(hist)
	# print(bin_edges)

	# fig, ax = plt.subplots()
	# plt.title("Sleeping Positions's Histogram")
	# plt.imshow(original_array1, cmap=plt.cm.jet)


	# 顯示圖形
	# fig, ax = plt.subplots()
	# plt.title("Sleeping Positions's Histogram")
	# plt.imshow(new_array, cmap=plt.cm.jet)
	# plt.yticks(np.arange(0.0, 40, 2.0))
	# plt.xticks(np.arange(-0.5, 10.5, 1.0))
	# plt.colorbar()
	fig, ax = plt.subplots()
	plt.xlabel('Pressure Value')
	plt.ylabel('Point')
	# plt.hist(np.sort(new_array.flatten()), bins = bin_edges)
	# plt.show()

if __name__ == '__main__':
	main()