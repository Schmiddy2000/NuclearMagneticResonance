# Imports
import numpy as np
import matplotlib.pyplot as plt

from tools import show_basic_csv_plot, get_csv_data, run_parabolic_interpolation


# File indices range from 28 to 34
file_index = 33
file_indices = np.arange(28, 35)

show_basic_csv_plot(file_index)

teflon_dips = np.array([[269, 573, 879], []])
