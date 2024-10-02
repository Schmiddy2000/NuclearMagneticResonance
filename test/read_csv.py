import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tools import averager
from fit_peaks import fit_skew_gaussian

file_index = 9

file_path = f'../data/oscilloscope_data/NewFile{file_index}.csv'

df = pd.read_csv(file_path)

channel_df = df.drop([0], axis=0)
channel_df = channel_df[['X', 'CH1', 'CH2']]
channel_df.columns = ['x', 'ch_one', 'ch_two']
channel_df = channel_df.astype(np.float64)

x = channel_df.x.to_numpy()

fit_skew_gaussian(averager(channel_df.ch_one.to_numpy()), 47, 56, True, True)

# plt.figure(figsize=(12, 5))
#
# plt.plot(x, averager(channel_df.ch_one.to_numpy()), c='y', label='CH1')
# plt.plot(x, averager(channel_df.ch_two.to_numpy())* 100, c='b', label='CH2')
#
# plt.legend()
# plt.show()
