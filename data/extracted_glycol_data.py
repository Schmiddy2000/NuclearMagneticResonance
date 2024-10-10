import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from numpy.f2py.symbolic import as_symbol

from tools import get_csv_data, show_basic_csv_plot, run_parabolic_interpolation

from typing import List
from numpy.typing import NDArray
from tools import show_basic_csv_plot, get_csv_data, run_parabolic_interpolation


# File indices range from 5 to 13
file_index = 13
file_indices = np.arange(5, 14)

#show_basic_csv_plot(file_index)

glycol_dips = np.array([[126, 435, 738, 1048], [126, 435, 738, 1048],
                        [108, 417, 720, 1031], [90, 401, 702, 1013],
                        [53, 359, 665, 972], [40, 349, 654, 962],
                        [35, 342, 648, 955], [26, 332, 639, 945], [22, 329, 635, 942]])



def get_difference_array(_file_indices: list[int] | NDArray, dips_array: NDArray):
    parabolic_positions = []
    parabolic_position_uncertainties = []
    difference_array = []
    difference_uncertainties = []

    for i, arr in enumerate(dips_array):
        # print(arr, type(arr))
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


difference_results = get_difference_array(file_indices, glycol_dips)
parabolic_pos, parabolic_pos_uncertainties, difference_arr, difference_uncertainty_arr = difference_results
def linearize_differences(difference_array):
    asymmetries = []

    for diff_arr in difference_array:
        side_average = (diff_arr[2] + diff_arr[0]) / 2
        asymmetry = side_average - diff_arr[1]
        asymmetries.append(asymmetry)

    return asymmetries


asymmetry_array = linearize_differences(difference_arr)
frequencies = [
    19.4568,  # NewFile5
    19.4447,  # NewFile6 (after correction)
    19.4400,  # NewFile7 (after correction)
    19.4166,  # NewFile8 (after correction)
    19.4043,  # NewFile9
    19.3982,  # NewFile10
    19.3894,  # NewFile11
    19.3802,  # NewFile12
    19.3749   # NewFile13
]
B_values = [
    448,  # NewFile5
    449,  # NewFile6
    449,  # NewFile7
    448,  # NewFile8 (renamed from NewFile7)
    448,  # NewFile8 (after correction)
    448,  # NewFile9 (previous value used)
    448,  # NewFile10 (previous value used)
    448,  # NewFile11
    448,  # NewFile12 (previous value used)
]


frequencies= np.array(frequencies)
b_field = np.array(B_values)

frequencies = 2 * np.pi * frequencies / b_field

# frequencies = frequencies / B_array
frequencies = frequencies - min(frequencies)


plt.figure(figsize=(12, 5))
plt.title('Distance in ')
plt.xlabel('Dip indices between which the distance was measured')
plt.ylabel('Distance in [resolution units]')

plt.scatter(frequencies, asymmetry_array)
# plt.errorbar(frequencies, asymmetry_array, )

# Linear fit (degree 1)
p, cov = np.polyfit(frequencies, asymmetry_array, 1, cov=True)
slope, intercept = p
slope_err, intercept_err = np.sqrt(np.diag(cov))

x_fit = np.linspace(min(frequencies), max(frequencies), 100)
y_fit = slope * x_fit + intercept

print(f'asymmetry_array: {asymmetry_array}')
print(slope_err, intercept_err)

# Calculate 1-sigma confidence bands
y_fit_upper = (slope + slope_err) * x_fit + (intercept + intercept_err)
y_fit_lower = (slope - slope_err) * x_fit + (intercept - intercept_err)

# Plot the regression line
plt.plot(x_fit, y_fit)

# Plot the 1-sigma confidence band
plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color='gray', alpha=0.2)
plt.show()