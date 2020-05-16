'''

editor: Jones
date: 2020/04/14
content: 
1. read data csv

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

raw_data_df = pd.read_csv('csvData/20200113wholeFaceLeft.csv')
raw_data_array = raw_data_df.to_numpy()

original_array = raw_data_array[1200][:220].reshape(20, 11)

# 顯示圖形
fig, ax = plt.subplots()
plt.title('Sleeping Positions')
plt.imshow(original_array, cmap=plt.cm.jet)
# plt.yticks(np.arange(0,20,1.0))
# plt.xticks(np.arange(0,11,1.0))
plt.show()

