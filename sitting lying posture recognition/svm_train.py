'''
editor: Jones
date: 2020/02/27
content: 用SVM去區分坐姿與睡姿

'''

from sklearn import datasets
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.externals import joblib


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
	# predict_result
	for index, element in enumerate(predict_result2):
		if element == 0:
			count0 += 1
			sitting_index_list.append(index)
		elif element == 1:
			count1 += 1
			lying_index_list.append(index)

	# print('count0_index_list =',count0_index_list)
	# print('len =', len(count0_index_list))
	print('lying_index_list size =', len(lying_index_list))

	return lying_index_list















