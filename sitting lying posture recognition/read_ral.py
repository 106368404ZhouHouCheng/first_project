'''

editor:Jones
date:2020/03/02
content:read rotated_array_label csv

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
	test_data = pd.read_csv('rotated array csv/Jones1224_rotated_array_label.csv')
	test_array = test_data.to_numpy()

	rotated_array = test_array[0][:-1]
	label = test_array[0][-1]
	row = 20
	col = 11

	face_up_label_list = []

	for index, elements in enumerate(test_array):
		label = elements[-1]
		if label == 0:
			face_up_label_list.append(label)

	# 1-20
	fig, axs = plt.subplots(4,5)
	plt.suptitle("Sleep Postures")

	ax_row = 0
	size = 901
	count = 0 + 20 * size

	while ax_row < 4:
		ax_col = 0
		while ax_col < 5:
			item_array = test_array[count][:-1].reshape(row, col)
			label = test_array[count][-1]
			if label == 0:
				axs[ax_row][ax_col].set_title('Face Up')
				print('Face Up')
			elif label == 1:
				axs[ax_row][ax_col].set_title('Face Right')
				print('Face Right')
			elif label == 2:
				axs[ax_row][ax_col].set_title('Face Left')
				print('Face Left')
			elif label == 3:
				axs[ax_row][ax_col].set_title('Face Down')
				print('Face Down')

			axs[ax_row][ax_col].imshow(item_array, cmap = plt.cm.jet)
			count += 1
			ax_col += 1
		ax_row += 1
	plt.show()


# 程式起點
if __name__ == '__main__':
	main()