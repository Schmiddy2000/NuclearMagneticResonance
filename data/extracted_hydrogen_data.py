import numpy as np
from matplotlib import pyplot as plt
from numpy.f2py.symbolic import as_symbol

from tools import get_csv_data, show_basic_csv_plot, run_parabolic_interpolation

from typing import List
from numpy.typing import NDArray


# Ranges from 14 to 22 for hydrogen
file_index = 21
file_indices = np.arange(14, 23, 1)

print(file_indices)


# Get the data
# _x, _ch1, _ch2 = get_csv_data(14)
# a, delta_a = run_parabolic_interpolation(_ch1, [128, 435, 740, 1048])
# print(a, delta_a)

# Show the plot
# show_basic_csv_plot(file_index)


# Datapoints
hydrogen_dips = np.array([[128, 435, 740, 1048], [128, 435, 740, 1048], [128, 435, 740, 1048],
                          [49, 358, 662, 970], [41, 348, 653, 961], [28, 334, 641, 947],
                          [43, 350, 656, 963], [67, 371, 679, 984], [122, 431, 735, 1044]])


# Computes the differences between the datapoints array (measurement) wise
# The first entry is the difference between the last and can be thought of as a wrap around. Here
# it still has to be determined if that is a sound approach.
def get_difference_array(_file_indices: List[int] | NDArray, dips_array: NDArray):
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


difference_results = get_difference_array(file_indices, hydrogen_dips)
parabolic_pos, parabolic_pos_uncertainties, difference_arr, difference_uncertainty_arr = difference_results


# Compute the difference between the central difference and the average of the ones on the left and right
# Add uncertainty calculations and return those as well
def linearize_differences(difference_array):
    asymmetries = []

    for diff_arr in difference_array:
        side_average = (diff_arr[2] + diff_arr[0]) / 2
        asymmetry = side_average - diff_arr[1]
        asymmetries.append(asymmetry)

    return asymmetries


asymmetry_array = linearize_differences(difference_arr)
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

# asymmetry_array.pop(7)
# frequencies.pop(7)

frequencies = np.array(frequencies)
B_array = np.array([447, 447, 447, 448, 448, 448, 446, 446, 446])


frequencies = 2 * np.pi * frequencies / B_array

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

# for i, d_a in enumerate(difference_arr):
#     # Scatter plot for data
#     plt.scatter(np.arange(len(d_a)), d_a, label=f'{file_indices[i]}')
#     plt.plot(np.arange(len(d_a)), d_a, ls='--', lw=0.75)
#
#     # Perform linear regression on the last three data points
#     x_last_three = np.arange(len(d_a))[-3:]
#     y_last_three = d_a[-3:]
#
#     # Linear fit (degree 1)
#     p, cov = np.polyfit(x_last_three, y_last_three, 1, cov=True)
#     slope, intercept = p
#     slope_err, intercept_err = np.sqrt(np.diag(cov))
#
#     # Generate x values for plotting the fit line
#     x_fit = np.linspace(x_last_three[0], x_last_three[-1], 100)
#     y_fit = slope * x_fit + intercept
#
#     # Calculate 1-sigma confidence bands
#     y_fit_upper = (slope + slope_err) * x_fit + (intercept + intercept_err)
#     y_fit_lower = (slope - slope_err) * x_fit + (intercept - intercept_err)

    # Plot the regression line
    # plt.plot(x_fit, y_fit, label=f'Fit {file_indices[i]}')

    # Plot the 1-sigma confidence band
    # plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color='gray', alpha=0.3, label=f'1-sigma band {file_indices[i]}')

# plt.legend()
# plt.xlim(-0.25, 2.25)
# plt.xticks([0, 1, 2], [r'$1 \rightarrow 2$', r'$2 \rightarrow 3$', r'$3 \rightarrow 4$'])
plt.tight_layout()
plt.show()

# Next step:
# - Use the difference data to do a fit that captures the difference between the dot in the
# middle to the one in the center in respect to the frequency. This should give us the necessary
# information to determine a best value for the resonance frequency along with uncertainties.
# Note: Also take the (very slightly) variable magnetic field into account and probably also use an
# ODR for the final assessment (resonance frequency).

# Alternative approach:
# Compute the difference between the central distance (2 -> 3) and the (average?) of the left (1 -> 2)
# and right distances (3 -> 4), and then plot this difference against the frequency.
# This should allow for a linear regression to determine the true resonance frequency.
