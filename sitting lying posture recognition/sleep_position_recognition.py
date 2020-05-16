'''
editor: Jones
date: 20200221
content: 
1.睡姿辨識
2.
'''

from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.optimizers import Nadam
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import tensorflow as tf 



# Load testing data from a file and return X, y
def load_testing_data(file_name):

	col = 11
	row = 20
	# open file and read it
	# split label
	# label = lying_data[['label']].to_numpy()
	# split data 
	train_data = lying_data.to_numpy()
	print('train_data[0] =', train_data[0])
	# split train data 
	train_data = train_data[::,:-1]
	train_data_reshape = []

	for i in train_data:
		i_reshape = i.reshape(row, col)
		train_data_reshape.append(i_reshape)
	train_data_reshape = np.array(train_data_reshape)

	return train_data_reshape

# 正規化
def normalization_data(data):

	col = 11
	row = 20
	# 多加一個顏色的維度 
	data_norm = data.reshape(data.shape[0], row, col, 1).astype('float32')
	print('data.shape[0] =', data.shape[0])
	# 正規化
	data_norm = data_norm/225

	return data_norm


def predict_classification_results(test_data):

	# 從 HDF5 檔案中載入模型
	model = tf.keras.models.load_model('model/20200221sleep_recognition.h5')

	# load_testing_data = load_testing_data(test_data)

	print("Loading testing data and label.")
	test_data_norm = normalization_data(test_data)

	# 預測分類機率分佈結果
	predict_test = model.predict(test_data_norm)
	predict = np.argmax(predict_test,axis=1)
	print('predict_test =', predict_test)

	# 預測分類結果
	predict_result = model.predict_classes(test_data_norm).astype('int')

	return predict_result

# # 評估模型準確率
# scores = model.evaluate(test_data_norm, label_norm)
# print(scores)
# print(scores[1]*100.0)

# really_result = label

# print('predict_result =', predict_result)
# print('really_result =', really_result.ravel())

# # 錯誤統計
# error = 0
# for i, v in enumerate(predict_result):
# 	if v != really_result[i]:
# 		print('predict_result =', v)
# 		print('count =', i)
# 		print('really_result = ', really_result[i])
# 		error += 1
# print('error =',error)