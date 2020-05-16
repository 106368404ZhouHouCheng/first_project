'''

editor: Jones
date: 2019/12/23
content: 
1.將四種睡姿放進CNN中訓練，分類出正躺，右側躺，左側躺，俯臥
'''

from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.optimizers import Nadam, Adam
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os
import tensorflow as tf 
from image_rotation import head_foot_split, head_foot_diff, rotation_angle, image_rotation, translat_X_axis_center

# from keras.callbacksimport ModelCheckpoint 

# Load training data from a file and return X, y
def load_training_data(file_name):

	# open file and read it
	lying_data = pd.read_csv(file_name)
	# split label
	label = lying_data[['label']].to_numpy()
	# split data 
	train_data = lying_data.to_numpy()
	# print('train_data[0] =', train_data[0])
	# split train data 
	train_data = train_data[::,:-1]
	train_data_reshape = []

	for i in train_data:
		i_reshape = i.reshape(row, col)
		train_data_reshape.append(i_reshape)
	train_data_reshape = np.array(train_data_reshape)
	print(train_data_reshape[0])
	print(train_data_reshape[0][::-1])

	return train_data_reshape, label

# 正規化
def normalization_data(data):

	# 多加一個顏色的維度 
	data_norm = data.reshape(data.shape[0], row, col, 1).astype('float32')
	print(data_norm.shape)
	# 正規化
	data_norm = data_norm/1023

	return data_norm

def image_rotated_array(raw_data):

	rotated_data_array = []
	for i in raw_data:
		angle = rotation_angle(i)
		i_trans = translat_X_axis_center(i)
		i_rotated = image_rotation(i_trans, angle)
		rotated_data_array.append(i_rotated)

	rotated_data_array = np.array(rotated_data_array)

	return rotated_data_array


# 創建模型
def create_model():

	model = Sequential()
	# create CNN layer 1
	model.add(Conv2D(filters=12, kernel_size=5, padding='same', activation='relu', input_shape=(row, col, 1)))
	model.add(MaxPooling2D(pool_size = 2))
	# create CNN layer 2
	model.add(Conv2D(filters=24, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size = 2))
	# add dropout layer
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(4, activation='softmax'))
	# 顯示模型結構
	model.summary()
	# optimizer = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
	model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
	return model

# def isDisplayAvl():  
#     return 'DISPLAY' in os.environ.keys()  

def plot_image(image):  
    fig = plt.gcf()  
    fig.set_size_inches(2,2)  
    plt.imshow(image, cmap='binary')  
    plt.show()  
  
def plot_images_labels_predict(images, labels, prediction, idx, num=10):  
    fig = plt.gcf()  
    fig.set_size_inches(12, 14)  
    if num > 25: num = 25  
    for i in range(0, num):  
        ax=plt.subplot(5,5, 1+i)  
        ax.imshow(images[idx], cmap='binary')  
        title = "l=" + str(labels[idx])  
        if len(prediction) > 0:  
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))  
        else:  
            title = "l={}".format(str(labels[idx]))  
        ax.set_title(title, fontsize=10)  
        ax.set_xticks([]); ax.set_yticks([])  
        idx+=1  
    plt.show()  


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# Train data
train_file = 'input/20200508whole_lying_train.csv'

col = 11
row = 20

print("Loading training data and label.")
train_data, label = load_training_data(train_file)
rotated_train_data = image_rotated_array(train_data)
print('rotated_train_data =', rotated_train_data[0])
X_train, X_test, y_train, y_test = train_test_split(rotated_train_data, label, test_size=0.1, random_state = 0)
X_train_norm = normalization_data(X_train)
X_test_norm = normalization_data(X_test)
# Label Onehot-encoding  
y_train_norm = np_utils.to_categorical(y_train)
y_test_norm = np_utils.to_categorical(y_test)
print('y_test_norm =', y_test_norm)


 # 創建一個基本的模型實例
model = create_model()

# 開始訓練
train_history = model.fit(X_train_norm, y_train_norm, validation_data=(X_test_norm, y_test_norm) , batch_size = 270, epochs = 30, verbose = 1, shuffle = True)

# 将整个模型保存为HDF5文件
model.save("./model/20200511_1546.h5")


# tf.keras.experimental.export_saved_model(model, saved_model_path)

# 評估模型準確率
scores = model.evaluate(X_test_norm, y_test_norm)
print(scores)
print(scores[1]*100.0)


# 預測分類機率分佈結果
predict_test = model.predict(X_test_norm)
predict = np.argmax(predict_test,axis=1)
# inverted = encoder.inverse_transform([predict])
print(predict_test[:10])
print(predict[:10])
# print(inverted)

count = 0
for col_index, col_element in enumerate(predict_test):
	if max(col_element) <= 0.25:
		print(col_index)
		count += 1

print('count =', count)

# 預測分類結果
predict_result = model.predict_classes(X_test_norm).astype('int')
print(predict_result[:10]) 

really_result = y_test

print('predict_result =', predict_result)
print('really_result =', really_result)

# 錯誤統計
error = 0
for i, v in enumerate(predict_result):
	if v != really_result[i]:
		# print(i)
		error += 1
print('error =',error)

show_train_history(train_history, 'acc', 'val_acc')  
show_train_history(train_history, 'loss', 'val_loss') 






