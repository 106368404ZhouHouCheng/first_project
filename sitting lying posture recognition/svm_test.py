'''
editor: Jones
date: 20190920
content: 
1. 找特徵
2. 4個特徵：
3. X軸，Y軸的變異數，
4. 平均壓力值，
5. 找重心，加半徑畫圓，圓内點數/總點數
6. 將特徵放到SVM去訓練，用來區分坐姿和睡姿
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


data = pd.read_csv('input/allnight/Jones1224_feature_data_0213.csv')

X = data[['average value', 'variance 1D', 'y axis variance', 'pressure value ratio as 3', 'points ratio as 3']]

sc = StandardScaler()
sc.fit(X)
print(sc.mean_)
print(sc.var_)
X_test_std = sc.transform(X)
print(X_test_std[:10])

# # read model
clf= joblib.load('model/20200213svm.pki')
print('clf =', clf)

# 預測
predict_result = clf.predict(X)
# really_result = y['target'].values
# print('really_result =', really_result)

# 錯誤統計
# error = 0
# for i, v in enumerate(predict_result):
# 	if v != really_result[i]:
# 		print(i)
# 		error += 1
# print('error =',error)

count0 = 0
count1 = 0
count0_index_list = []
count1_index_list = []
# predict_result
for index, element in enumerate(predict_result):
	if element ==0:
		count0 += 1
		count0_index_list.append(index)
	elif element == 1:
		count1 += 1
		count1_index_list.append(index)

# print('count0_index_list =',count0_index_list)
print(len(count1_index_list))
print(len(count0_index_list))








