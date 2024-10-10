# Imports
import numpy as np
import matplotlib.pyplot as plt

from tools import show_basic_csv_plot, get_csv_data, run_parabolic_interpolation


# File indices range from 23 to 27
file_index = 24
file_indices = np.arange(23, 28)

show_basic_csv_plot(file_index)

# The last dip is problematic (delta of 3)
# The first dip is problematic (delta of 5)
# The second and last dip id problematic (delta of 2)
# [274, 577.5, 886, 1189]
fluorine_dips = np.array([[239, 549, 851, 1161.5], [273, 577.5, 887.5, 1190], [251, 562.5, 863.5, 1174],
                          [209.5, 514.5, 823, 1126], [275.5, 584, 891, 1195.5]])


# Second measurement, second left is questionable
fluorine_fwhm_left = np.array([[236.7, 544.75, 848.8, 1158.73], [270.7, 575, 882.7, 1186.85]])
fluorine_fwhm_right = np.array([[240.5, 550.43, 852.53, 1164.24], [276.4, 580.3, 890.34, 1192.35]])

asymmetry = [3/2 * (fd[1] - fd[2]) + 1/2 * (fd[3] - fd[0]) for fd in fluorine_dips]
fluorine_frequencies = np.array([17.7895, 17.7835, 17.7818, 17.7885, 17.7752])
B_array = np.array([446, 446, 446, 447, 447])

gamma = 2 * np.pi * fluorine_frequencies / B_array

print(asymmetry)

plt.figure(figsize=(12, 5))
plt.scatter(gamma, asymmetry)
plt.show()

print(asymmetry)

