import numpy as np
import pandas as pd
from copy import copy
from matplotlib import pyplot as plt

from typing import Tuple


def get_csv_data(file_index: int, use_average: bool = False) -> Tuple[np.array, np.array, np.array]:
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


def show_basic_csv_plot(file_index: int) -> None:
    x, ch_1, ch_2 = get_csv_data(file_index)

    plt.figure(figsize=(12, 5))

    plt.plot(x, ch_1, label='CH1', c='y')
    plt.plot(x, ch_2, label='CH2', c='b')

    plt.legend()
    plt.show()

    return None


def averager(counts_array: np.array) -> np.array:
    averaged_counts_array = copy(counts_array)
    last_true_value = counts_array[0]

    for i in range(len(counts_array) - 2):
        new_last_true_value = averaged_counts_array[i + 1]
        averaged_counts_array[i + 1] = np.mean([last_true_value, averaged_counts_array[i + 1],
                                                averaged_counts_array[i + 2]])
        last_true_value = new_last_true_value

    return averaged_counts_array
