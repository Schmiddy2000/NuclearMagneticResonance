import numpy as np
import pandas as pd
from copy import copy
from matplotlib import pyplot as plt

from typing import Tuple, List, Union
from numpy.typing import NDArray


def get_csv_data(file_index: int,
                 use_average: bool = False
                 ) -> Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
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
    plt.tight_layout()
    plt.grid()
    plt.show()

    return None


def averager(counts_array: NDArray) -> NDArray[np.float_]:
    averaged_counts_array = copy(counts_array)
    last_true_value = counts_array[0]

    for i in range(len(counts_array) - 2):
        new_last_true_value = averaged_counts_array[i + 1]
        averaged_counts_array[i + 1] = np.mean([last_true_value, averaged_counts_array[i + 1],
                                                averaged_counts_array[i + 2]])
        last_true_value = new_last_true_value

    return averaged_counts_array


def run_parabolic_interpolation(y: Union[List[int], NDArray[np.int_]],
                                position_indices: Union[List[int], NDArray[np.int_]]
                                ) -> Tuple[List[float], List[float]]:
    """
    The function expects the y-data along with a sequence of positions (indices)
    that correspond to the local minima of each dip from one measurement.

    It will then use parabolic interpolation to compute and return the best guess
    for the real minima of the dip along with the corresponding uncertainty.
    """

    # Lists to store the obtained minima and uncertainties
    x_min_list = []
    delta_x_min_list = []

    # Compute the resolution in y direction
    y_diff = np.diff(y)
    y_resolution = np.min(np.abs(y_diff[y_diff != 0]))

    for p_i in position_indices:
        y_i_minus_one = y[p_i - 1]
        y_i = y[p_i]
        y_i_plus_one = y[p_i + 1]

        x_min = p_i + (y_i_minus_one - y_i_plus_one) / (2 * (y_i_minus_one - 2 * y_i + y_i_plus_one))
        x_min_list.append(x_min)

        delta_x_min = (y_resolution / ((y_i_minus_one - 2 * y_i + y_i_plus_one) ** 2) *
                       np.sqrt((y_i_minus_one - y_i_plus_one) ** 2 + (y_i - y_i_plus_one) ** 2 +
                               (y_i - y_i_minus_one) ** 2))
        delta_x_min_list.append(delta_x_min)

    return x_min_list, delta_x_min_list
