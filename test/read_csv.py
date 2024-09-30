import pandas as pd
import numpy as np
from tools import averager

from typing import Tuple

from matplotlib import pyplot as plt
from fit_peaks import fit_skew_gaussian


def get_data(file_index: int, use_average: bool = False) -> Tuple[np.array, np.array, np.array]:
    file_path = f'../data/oscilloscope_data/NewFile{file_index}.csv'
    df = pd.read_csv(file_path)

    channel_df = df.drop([0], axis=0)
    channel_df = channel_df[['X', 'CH1', 'CH2']]
    channel_df.columns = ['x', 'ch_one', 'ch_two']
    channel_df = channel_df.astype(np.float64)

    x = channel_df.x.to_numpy()
    ch_1 = channel_df.ch_one.to_numpy()
    ch_2 = channel_df.ch_two.to_numpy()

    if use_average:
        ch_1 = averager(ch_1)
        ch_2 = averager(ch_2)

    return x, ch_1, ch_2


# fit_skew_gaussian(averager(channel_df.ch_one.to_numpy()), 47, 56, True, True)

# plt.figure(figsize=(12, 5))
#
# plt.plot(x, averager(channel_df.ch_one.to_numpy()), c='y', label='CH1')
# plt.plot(x, averager(channel_df.ch_two.to_numpy())* 100, c='b', label='CH2')
#
# plt.legend()
# plt.show()
