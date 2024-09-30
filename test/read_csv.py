import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

file_index = 9

file_path = f'../data/test/NewFile{file_index}.csv'

df = pd.read_csv(file_path)

channel_df = df.drop([0], axis=0)
channel_df = channel_df[['X', 'CH1', 'CH2']]
channel_df.columns = ['x', 'ch_one', 'ch_two']
channel_df = channel_df.astype(np.float64)

x = channel_df.x.to_numpy()

plt.plot(x, channel_df.ch_one.to_numpy(), c='y', label='CH1')

plt.plot(x, channel_df.ch_two.to_numpy() * 100, c='b', label='CH2')

plt.legend()
plt.show()
