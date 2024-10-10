# Imports
import numpy as np
import matplotlib.pyplot as plt

from tools import get_csv_data, show_basic_csv_plot, run_parabolic_interpolation

from typing import List
from numpy.typing import NDArray


# Data
file_indices = np.arange(14, 23, 1)
hydrogen_dips = np.array([[128, 435, 740, 1048], [128, 435, 740, 1048], [128, 435, 740, 1048],
                          [49, 358, 662, 970], [41, 348, 653, 961], [28, 334, 641, 947],
                          [43, 350, 656, 963], [67, 371, 679, 984], [122, 431, 735, 1044]])

# show_basic_csv_plot(21)

for i, a in enumerate(hydrogen_dips):
    print(f'{file_indices[i]}: {a[2] - a[0], a[3] - a[1]}')
    print(f'\t{a[1] - a[0], a[3] - a[2]}')
    print(f'\t{a[2] - a[1]}')

exit()


# Computes the differences between the datapoints array (measurement) wise
def get_difference_array(_file_indices: List[int] | NDArray, dips_array: NDArray):
    parabolic_positions = []
    parabolic_position_uncertainties = []
    difference_array = []
    difference_uncertainties = []

    for i, arr in enumerate(dips_array):
        _, y_data, _ = get_csv_data(_file_indices[i])
        positions, uncertainties = run_parabolic_interpolation(y_data, arr)
        diff_arr = [positions[j + 1] - positions[j] for j in range(len(positions) - 1)]
        diff_uncertainty_arr = [np.sqrt(uncertainties[j + 1] ** 2 + uncertainties[j] ** 2)
                                for j in range(len(positions) - 1)]

        parabolic_positions.append(positions)
        parabolic_position_uncertainties.append(uncertainties)
        difference_array.append(diff_arr)
        difference_uncertainties.append(diff_uncertainty_arr)

    return parabolic_positions, parabolic_position_uncertainties, difference_array, difference_uncertainties


difference_results = get_difference_array(file_indices, hydrogen_dips)
parabolic_pos, parabolic_pos_uncertainties, difference_arr, difference_uncertainty_arr = difference_results

# Like this, the asymmetries should give lines with opposed signs (slope-wise).
# This could help with determining an asymmetry of 0, since we can use the intercept of the two lines.
first_asymmetries = np.array([pos[0] - pos[1] for pos in parabolic_pos])
second_asymmetries = np.array([pos[1] - pos[2] for pos in parabolic_pos])

# Actually, the baseline should be

combined_asymmetries = first_asymmetries - second_asymmetries

B_array = np.array([447, 447, 447, 448, 448, 448, 446, 446, 446])

B_array_alternative = np.array([447, 447, 447, 446.5, 446.5, 446.4, 446, 446, 446])

frequencies = [
    19.3955,  # Measurement 1
    19.3900,  # Measurement 2
    19.3850,  # Measurement 3
    19.3734,  # Measurement 4
    19.3686,  # Measurement 5
    19.3592,  # Measurement 6
    19.3643,  # Measurement 7
    19.3716,  # Measurement 8
    19.3826   # Measurement 9
]
frequencies = np.array(frequencies)

print(frequencies / (1e-3 * B_array))

# Gamma is the gyroscopic ratio
gamma = 2 * np.pi * frequencies / B_array

plt.figure(figsize=(12, 5))

# plt.scatter(gamma, first_asymmetries, c='orange', label='first asymmetries')
# plt.scatter(gamma, second_asymmetries, c='blue', label='second asymmetries')
plt.scatter(gamma, combined_asymmetries, c='blue', label='combined asymmetries')

plt.legend()
plt.tight_layout()
plt.show()