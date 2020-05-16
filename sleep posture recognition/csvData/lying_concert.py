'''

editor: Jones
date: 2019/12/20
content: 将各自的csv 组合成 一整个 train csv

'''
import pandas as pd 
import csv
import numpy as np

# lying data
# jones_lying_tran_data = pd.read_csv('rawData/1204Jones_lying.csv')
# print(len(jones_lying_tran_data))
# ziyang_lying_tran_data = pd.read_csv('rawData/1210子暘_lying.csv')
# print('len = ', len(ziyang_lying_tran_data))
# yucheng_lying_tran_data = pd.read_csv('rawData/1210育誠_lying.csv')
# print('len = ', len(yucheng_lying_tran_data))
# jiayun_lying_tran_data = pd.read_csv('rawData/1210佳芸_lying.csv')
# print('len = ', len(jiayun_lying_tran_data))
# jianyu_lying_tran_data = pd.read_csv('rawData/1210建宇_lying.csv')
# print('len = ', len(jianyu_lying_tran_data))
# siqi_lying_tran_data = pd.read_csv('rawData/1210思琪_lying.csv')
# print('len = ', len(siqi_lying_tran_data))
# hongliang_lying_tran_data = pd.read_csv('rawData/1210竑量_lying.csv')
# print('len = ', len(hongliang_lying_tran_data))
# haosong_lying_tran_data = pd.read_csv('rawData/1210皓菘_lying.csv')
# print('len = ', len(haosong_lying_tran_data))
# pengxiang_lying_tran_data = pd.read_csv('rawData/1210鵬翔_lying.csv')
# print('len = ', len(pengxiang_lying_tran_data))
# yaotang_lying_tran_data = pd.read_csv('rawData/1210耀堂_lying.csv')
# print('len = ', len(yaotang_lying_tran_data))
# peichen_lying_tran_data = pd.read_csv('rawData/1211沛忱_lying.csv')
# print('len = ', len(peichen_lying_tran_data))
# peizhen_lying_tran_data = pd.read_csv('rawData/1211沛臻_lying.csv')
# print('len = ', len(peizhen_lying_tran_data))
# jiahong_lying_tran_data = pd.read_csv('rawData/1211嘉宏_lying.csv')
# print('len = ', len(jiahong_lying_tran_data))
# yanhong_lying_tran_data = pd.read_csv('rawData/1211燕鴻_lying.csv')
# print('len = ', len(yanhong_lying_tran_data))
# xianjun_lying_tran_data = pd.read_csv('rawData/1211顯郡_lying.csv')
# print('len = ', len(xianjun_lying_tran_data))
# zhibo_lying_tran_data = pd.read_csv('rawData/1212郅博_lying.csv')
# print('len = ', len(zhibo_lying_tran_data))
# yijia_lying_tran_data = pd.read_csv('rawData/1212翊嘉_lying.csv')
# print('len = ', len(yijia_lying_tran_data))
# bingyu_lying_tran_data = pd.read_csv('rawData/1217秉渝_lying.csv')
# print('len = ', len(bingyu_lying_tran_data))


# threshold1 = 100
# threshold2 = 200
# threshold3 = 300
# jones_face_up_data = jones_lying_tran_data[0:threshold1]
# ziyang_face_up_data = ziyang_lying_tran_data[0:threshold1]
# yucheng_face_up_data = yucheng_lying_tran_data[0:threshold1]
# jiayun_face_up_data = jiayun_lying_tran_data[0:threshold1]
# jianyu_face_up_data = jianyu_lying_tran_data[0:threshold1]
# siqi_face_up_data = siqi_lying_tran_data[0:threshold1]
# hongliang_face_up_data = hongliang_lying_tran_data[0:threshold1]
# haosong_face_up_data = haosong_lying_tran_data[0:threshold1]
# pengxiang_face_up_data = pengxiang_lying_tran_data[0:threshold1]
# yaotang_face_up_data = yaotang_lying_tran_data[0:threshold1]
# peichen_face_up_data = peichen_lying_tran_data[0:threshold1]
# peizhen_face_up_data = peizhen_lying_tran_data[0:threshold1]
# jiahong_face_up_data = jiahong_lying_tran_data[0:threshold1]
# yanhong_face_up_data = yanhong_lying_tran_data[0:threshold1]
# xianjun_face_up_data = xianjun_lying_tran_data[0:threshold1]
# zhibo_face_up_data = zhibo_lying_tran_data[0:threshold1]
# yijia_face_up_data = yijia_lying_tran_data[0:threshold1]
# bingyu_face_up_array = bingyu_lying_tran_data[0:threshold1]


# face_up_data = jones_face_up_data.append(ziyang_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(yucheng_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(jiayun_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(jianyu_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(siqi_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(hongliang_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(haosong_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(pengxiang_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(yaotang_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(peichen_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(peizhen_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(jiahong_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(yanhong_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(xianjun_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(zhibo_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(yijia_face_up_data, ignore_index=True)
# face_up_data = face_up_data.append(bingyu_face_up_array, ignore_index=True)
# print(face_up_data.head)

# face_up_data.to_csv("20200113wholeFaceUp.csv" , encoding = "utf-8")


# jones_face_right_data = jones_lying_tran_data[threshold1:threshold2]
# ziyang_face_right_data = ziyang_lying_tran_data[threshold1:threshold2]
# yucheng_face_right_data = yucheng_lying_tran_data[threshold1:threshold2]
# jiayun_face_right_data = jiayun_lying_tran_data[threshold1:threshold2]
# jianyu_face_right_data = jianyu_lying_tran_data[threshold1:threshold2]
# siqi_face_right_data = siqi_lying_tran_data[threshold1:threshold2]
# hongliang_face_right_data = hongliang_lying_tran_data[threshold1:threshold2]
# haosong_face_right_data = haosong_lying_tran_data[threshold1:threshold2]
# pengxiang_face_right_data = pengxiang_lying_tran_data[threshold1:threshold2]
# yaotang_face_right_data = yaotang_lying_tran_data[threshold1:threshold2]
# peichen_face_right_data = peichen_lying_tran_data[threshold1:threshold2]
# peizhen_face_right_data = peizhen_lying_tran_data[threshold1:threshold2]
# jiahong_face_right_data = jiahong_lying_tran_data[threshold1:threshold2]
# yanhong_face_right_data = yanhong_lying_tran_data[threshold1:threshold2]
# xianjun_face_right_data = xianjun_lying_tran_data[threshold1:threshold2]
# zhibo_face_right_data = zhibo_lying_tran_data[threshold1:threshold2]
# yijia_face_right_data = yijia_lying_tran_data[threshold1:threshold2]
# bingyu_face_right_array = bingyu_lying_tran_data[threshold1:threshold2]


# face_right_data = jones_face_right_data.append(ziyang_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(yucheng_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(jiayun_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(jianyu_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(siqi_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(hongliang_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(haosong_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(pengxiang_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(yaotang_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(peichen_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(peizhen_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(jiahong_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(yanhong_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(xianjun_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(zhibo_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(yijia_face_right_data, ignore_index=True)
# face_right_data = face_right_data.append(bingyu_face_right_array, ignore_index=True)
# print(face_right_data.head)

# face_right_data.to_csv("20200113wholeFaceRight.csv" , encoding = "utf-8")

# jones_face_left_data = jones_lying_tran_data[threshold2:threshold3]
# ziyang_face_left_data = ziyang_lying_tran_data[threshold2:threshold3]
# yucheng_face_left_data = yucheng_lying_tran_data[threshold2:threshold3]
# jiayun_face_left_data = jiayun_lying_tran_data[threshold2:threshold3]
# jianyu_face_left_data = jianyu_lying_tran_data[threshold2:threshold3]
# siqi_face_left_data = siqi_lying_tran_data[threshold2:threshold3]
# hongliang_face_left_data = hongliang_lying_tran_data[threshold2:threshold3]
# haosong_face_left_data = haosong_lying_tran_data[threshold2:threshold3]
# pengxiang_face_left_data = pengxiang_lying_tran_data[threshold2:threshold3]
# yaotang_face_left_data = yaotang_lying_tran_data[threshold2:threshold3]
# peichen_face_left_data = peichen_lying_tran_data[threshold2:threshold3]
# peizhen_face_left_data = peizhen_lying_tran_data[threshold2:threshold3]
# jiahong_face_left_data = jiahong_lying_tran_data[threshold2:threshold3]
# yanhong_face_left_data = yanhong_lying_tran_data[threshold2:threshold3]
# xianjun_face_left_data = xianjun_lying_tran_data[threshold2:threshold3]
# zhibo_face_left_data = zhibo_lying_tran_data[threshold2:threshold3]
# yijia_face_left_data = yijia_lying_tran_data[threshold2:threshold3]
# bingyu_face_left_array = bingyu_lying_tran_data[threshold2:threshold3]


# face_left_data = jones_face_left_data.append(ziyang_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(yucheng_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(jiayun_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(jianyu_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(siqi_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(hongliang_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(haosong_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(pengxiang_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(yaotang_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(peichen_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(peizhen_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(jiahong_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(yanhong_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(xianjun_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(zhibo_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(yijia_face_left_data, ignore_index=True)
# face_left_data = face_left_data.append(bingyu_face_left_array, ignore_index=True)
# print(face_left_data.head)

# face_left_data.to_csv("20200113wholeFaceLeft.csv" , encoding = "utf-8")

# jones_face_down_data = jones_lying_tran_data[threshold3:]
# ziyang_face_down_data = ziyang_lying_tran_data[threshold3:]
# yucheng_face_down_data = yucheng_lying_tran_data[threshold3:]
# jiayun_face_down_data = jiayun_lying_tran_data[threshold3:]
# jianyu_face_down_data = jianyu_lying_tran_data[threshold3:]
# siqi_face_down_data = siqi_lying_tran_data[threshold3:]
# hongliang_face_down_data = hongliang_lying_tran_data[threshold3:]
# haosong_face_down_data = haosong_lying_tran_data[threshold3:]
# pengxiang_face_down_data = pengxiang_lying_tran_data[threshold3:]
# yaotang_face_down_data = yaotang_lying_tran_data[threshold3:]
# peichen_face_down_data = peichen_lying_tran_data[threshold3:]
# peizhen_face_down_data = peizhen_lying_tran_data[threshold3:]
# jiahong_face_down_data = jiahong_lying_tran_data[threshold3:]
# yanhong_face_down_data = yanhong_lying_tran_data[threshold3:]
# xianjun_face_down_data = xianjun_lying_tran_data[threshold3:]
# zhibo_face_down_data = zhibo_lying_tran_data[threshold3:]
# yijia_face_down_data = yijia_lying_tran_data[threshold3:]
# bingyu_face_down_array = bingyu_lying_tran_data[threshold3:]


# face_down_data = jones_face_down_data.append(ziyang_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(yucheng_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(jiayun_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(jianyu_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(siqi_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(hongliang_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(haosong_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(pengxiang_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(yaotang_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(peichen_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(peizhen_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(jiahong_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(yanhong_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(xianjun_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(zhibo_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(yijia_face_down_data, ignore_index=True)
# face_down_data = face_down_data.append(bingyu_face_down_array, ignore_index=True)
# print(face_down_data.head)

# face_down_data.to_csv("20200113wholeFaceDown.csv" , encoding = "utf-8")

# # lying_data
# lying_data = jones_lying_tran_data.append(ziyang_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(yucheng_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(jiayun_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(jiayun_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(siqi_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(hongliang_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(haosong_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(pengxiang_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(yaotang_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(peichen_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(peizhen_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(jiahong_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(yanhong_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(xianjun_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(zhibo_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(yijia_lying_tran_data, ignore_index=True)
# lying_data = lying_data.append(bingyu_lying_tran_data, ignore_index=True)
# print(lying_data.head)

# lying_data.to_csv("1216whole_lying_tran.csv" , encoding = "utf-8")


# sitting_data = pd.read_csv('1216whole_sitting.csv')
# lying_data = pd.read_csv('1216whole_lying.csv')

# whole_train_data = sitting_data.append(lying_data, ignore_index=True)

# print(whole_train_data)

# whole_train_data.to_csv("1216whole_train_data.csv" , encoding = "utf-8")

# jones_lying_tran_data = pd.read_csv('rawData/1204Jones_lying.csv')


















