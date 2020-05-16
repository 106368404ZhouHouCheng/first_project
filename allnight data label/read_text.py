'''
editor: Jones
date:2020/04/19
content:
1. read allnight sleep data
2. data save to csv file type
3. allnight data label
4. leave bed, 
5. sitting lying difference,
6. sleep posture  
'''

# print("hello world!")
import numpy as np 
import pandas as pd 

file_name = 'textData/Jones0430.txt'
file = open(file_name, "r")
# print(type(file))
file_list = []
for x in file:
	file_list.append(x)


allnight_data_list = [] 

for frame in file_list[1:]:
	frame_item = frame.split( )
	frame_list = [] 
	for item in frame_item[:220]:
		frame_list.append(int(item))

	frame_array = np.array(frame_list).reshape(20,11)
	mat180_array = np.rot90(frame_array, 2).flatten()
	# print(mat180_array)
	allnight_data_list.append(mat180_array)

print(len(allnight_data_list[0]))
count = 0
columns = []
while count < len(allnight_data_list[0]):
	column_name = 'pixel' + str(count)
	columns.append(column_name)
	count += 1

print(columns)
data_frame = pd.DataFrame(allnight_data_list, columns = columns)
print(data_frame.head)

data_frame.to_csv(r'Jones0430.csv', index = False)





