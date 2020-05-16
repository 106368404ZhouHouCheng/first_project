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
from image_rotation import head_foot_split, head_foot_diff, rotation_angle, image_rotation, translat_X_axis_center



# Load training data from a file and return X, y
def load_training_data(file_name):

	row = 20
	col = 11
	# open file and read it
	lying_data = pd.read_csv(file_name)
	# split label
	raw_data = lying_data.to_numpy()
	# print(raw_data)
	# split data 
	train_data_reshape = []
	label = []

	for i in raw_data:
		if i[-1] == 0 or i[-1] == 1 or i[-1] == 2 or i[-1] == 3:
			i_reshape = i[:220].reshape(row, col)
			train_data_reshape.append(i_reshape)
			label.append(i[-1])
	train_data_reshape = np.array(train_data_reshape)
	label = np.array(label)
	# print(train_data_reshape)
	# print(label)

	return train_data_reshape, label

def image_rotated_array(raw_data):

	rotated_data_array = []
	for i in raw_data:
		angle = rotation_angle(i)
		i_trans = translat_X_axis_center(i)

		i_rotated = image_rotation(i_trans, angle)
		rotated_data_array.append(i_rotated)

	rotated_data_array = np.array(rotated_data_array)

	return rotated_data_array

def normalization_data(data):

	row = 20
	col = 11
	# 多加一個顏色的維度 
	data_norm = data.reshape(data.shape[0], row, col, 1).astype('float32')
	# print('data.shape[0] = ', data.shape[0])
	# 正規化
	data_norm = data_norm/1023

	return data_norm



# 從 HDF5 檔案中載入模型
model = tf.keras.models.load_model('model/20200511_1546.h5')
# 顯示模型結構
model.summary()
# 檔案
test_file = 'csvData/allnightData/Jane0411.csv'

print("Loading testing data and label.")
test_data, label = load_training_data(test_file)
rotated_test_data = image_rotated_array(test_data)
# print('rotated_train_data =', rotated_test_data[0])

# rotated_data_array = []
# for i in test_data:
# 	angle = rotation_angle(i)
# 	i_rotated = image_rotation(i,angle)
# 	rotated_data_array.append(i_rotated)

# rotated_data_array = np.array(rotated_data_array)


# print('test_data[0] =', test_data[0:10])
test_data_norm = normalization_data(rotated_test_data)
# print(test_data_norm[0])
label_norm = np_utils.to_categorical(label)
# print(label_norm)


# 評估模型準確率
# scores = model.evaluate(test_data_norm, label_norm)
# print(scores)
# print(scores[1]*100.0)


# 預測分類機率分佈結果
predict_test = model.predict(test_data_norm)
predict = np.argmax(predict_test,axis=1)

# # print(predict_test[:100])

# # count = 0
# # for col_index, col_element in enumerate(predict_test[:100]):
# # 	if max(col_element) <= 0.25:
# # 		print(col_index)
# # 		count += 1

# # print('count =', count)

# # # print(predict_test[:10])
# # # print(predict[:10])
# # # print(inverted)

# # # 預測分類結果
predict_result = model.predict_classes(test_data_norm).astype('int')
# print(predict_result[:10]) 

really_result = label

print('predict_result =', predict_result)
print('really_result =', really_result.ravel())


error_list = []
# 錯誤統計
error = 0
for i, v in enumerate(predict_result):
	if v != really_result[i]:
		print('predict_result =', v)
		print('count =', i)
		# error_list.append(test_data[i])
		print('really_result = ', really_result[i])
		error += 1
print('error =',error)
# print(error_list)