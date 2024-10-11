# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

from tools import show_basic_csv_plot, get_csv_data, run_parabolic_interpolation
from numpy.typing import NDArray
from typing import List


def odr_fit_with_plot(x, y, x_err, y_err):
    # Shift x to x = x - min(x)
    x_min = np.min(x)
    x_shifted = x - x_min

    # Internal linear model: y = a * x + b
    def linear_model(params, x):
        a, b = params
        return a * x + b

    # Create RealData object including the uncertainties
    data = RealData(x_shifted, y, sx=x_err, sy=y_err)

    # Create Model object using the internal linear function
    model = Model(linear_model)

    # Initial guess for the parameters [slope, intercept]
    initial_params = [1, 1]

    # Create ODR object
    odr = ODR(data, model, beta0=initial_params)

    # Run the ODR fit
    output = odr.run()

    # Get the optimal parameters and their uncertainties
    popt = output.beta
    perr = output.sd_beta

    # Generate x-values for plotting the best fit line
    x_fit = np.linspace(min(x_shifted), max(x_shifted), 100)

    # Best fit line
    y_fit = linear_model(popt, x_fit)

    # Confidence band: best fit plus/minus uncertainties
    y_fit_upper = linear_model([popt[0] + perr[0], popt[1] + perr[1]], x_fit)
    y_fit_lower = linear_model([popt[0] - perr[0], popt[1] - perr[1]], x_fit)

    # Find the x-values where the fit and confidence bounds intercept the y-axis (y=0)
    def find_x_intercept(slope, intercept):
        return -intercept / slope

    # Best fit intercept with y=0
    slope, intercept = popt[0], popt[1]
    x_intercept_best = find_x_intercept(slope, intercept)

    # Upper and lower confidence fit intercepts with y=0
    slope_upper, intercept_upper = popt[0] + perr[0], popt[1] + perr[1]
    x_intercept_upper = find_x_intercept(slope_upper, intercept_upper)

    slope_lower, intercept_lower = popt[0] - perr[0], popt[1] - perr[1]
    x_intercept_lower = find_x_intercept(slope_lower, intercept_lower)

    # Calculate the extended range for x-axis so that the lower line's intercept is included
    x_min_extended = min(x_intercept_best, x_intercept_upper, x_intercept_lower)
    x_max_extended = max(x_intercept_best, x_intercept_upper, x_intercept_lower)

    x_min_extended = -2# 270
    x_max_extended = 2# 273.5

    # Extend the x-axis range to cover the lower intercept as well
    x_fit_extended = np.linspace(x_min_extended, x_max_extended, 100)

    return_dict = {
        'parameters': popt,
        'uncertainties': perr,
        'x_intercept_best': x_intercept_best + x_min,  # Shifted back to original x
        'x_intercept_upper': x_intercept_upper + x_min,  # Shifted back to original x
        'x_intercept_lower': x_intercept_lower + x_min,  # Shifted back to original x
        'delta_min': x_intercept_best - x_intercept_upper,
        'delta_max': x_intercept_lower - x_intercept_best
    }

    print(return_dict)

    # Plotting
    plt.figure(figsize=(12, 5))

    # Scatter plot of data points with x shifted back
    plt.errorbar(x_shifted + x_min, y, xerr=x_err, yerr=y_err, fmt='o', label='Data points', capsize=5)

    # Best fit line, shifted back to the original x-values
    plt.plot(x_fit_extended + x_min, linear_model(popt, x_fit_extended), 'r-', label='Best fit')

    # 1-sigma confidence band (extend lower line to y=0)
    plt.fill_between(x_fit_extended + x_min, linear_model([popt[0] - perr[0], popt[1] - perr[1]], x_fit_extended),
                     linear_model([popt[0] + perr[0], popt[1] + perr[1]], x_fit_extended), color='red', alpha=0.2,
                     label=r'1-$\sigma$ confidence band')

    # Labels and legend
    plt.xlabel(r'Gyromagnetic moment $\gamma$ in [rad$\cdot$T$^{-1}\cdot$MHz]', fontsize=13)
    plt.ylabel('Asymmetry $a$', fontsize=13)
    plt.title('ODR Linear Fit of Hydrogen using parabolic interpolation - zoomed in', fontsize=16)
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    # plt.xlim(271.2, x_max_extended + x_min)
    plt.xlim(271.375, 273.5)
    plt.ylim(-15, 15)
    plt.savefig('gyromagnetic_moment_hydrogen_fit_zoomed.png', dpi=200)

    # Show the plot
    plt.show()

    return None


# Ranges from 14 to 22 for hydrogen
file_index = 14
file_indices = np.arange(14, 23, 1)

print(file_indices)

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


# Second measurement, second left is questionable
# Third measurement, third left is questionable
# Fourth measurement, third left
# fluorine_fwhm_left = np.array([])
# fluorine_fwhm_right = np.array([])
#
# fluorine_fwhm_left_uncertainties = np.array([[1, 1, 1, 1], [1, 3, 1, 1], [1, 1, 3, 1], [1, 1, 3, 1], [1, 1, 1, 1]])
# fluorine_fwhm_right_uncertainties = np.array([[1, 1, 1, 1], [1, 3, 1, 1], [1, 1, 3, 1], [1, 1, 3, 1], [1, 1, 1, 1]])
#
# fluorine_fwhm_uncertainties = np.array([np.sqrt(u_l ** 2 + u_r ** 2) for u_l, u_r
#                                         in zip(fluorine_fwhm_left_uncertainties, fluorine_fwhm_right_uncertainties)])

frequency_uncertainties = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]) * 1e-4 / np.sqrt(3)
B_uncertainties = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]) * 1e-3 / np.sqrt(3)

B_array = np.array([447, 447, 447, 448, 448, 448, 446, 446, 446]) * 1e-3

frequencies = np.array([
    19.3955,  # Measurement 1
    19.3900,  # Measurement 2
    19.3850,  # Measurement 3
    19.3734,  # Measurement 4
    19.3686,  # Measurement 5
    19.3592,  # Measurement 6
    19.3643,  # Measurement 7
    19.3716,  # Measurement 8
    19.3826  # Measurement 9
])


# Still need to calculate the real uncertainties here
def get_gamma_uncertainties(frequencies, B, delta_frequencies, delta_B):
    root_term = delta_frequencies ** 2 + (frequencies * delta_B / B) ** 2

    return 2 * np.pi / B * np.sqrt(root_term)


gamma = frequencies * 2 * np.pi / B_array

# fluorine_fwhm_dips = (fluorine_fwhm_left + fluorine_fwhm_right) / 2


# asymmetry_fwhm = [3/2 * (fd[1] - fd[2]) + 1/2 * (fd[3] - fd[0]) for fd in fluorine_fwhm_dips]
asymmetry = [3 / 2 * (fd[1] - fd[2]) + 1 / 2 * (fd[3] - fd[0]) for fd in parabolic_pos]
asymmetry_uncertainties = [np.sqrt(9 / 4 * (fd[1] ** 2 + fd[2] ** 2) + 1 / 4 * (fd[3] ** 2 + fd[0] ** 2)) for fd
                           in parabolic_pos_uncertainties]

# -----------------

# asymmetry_fwhm_uncertainties = [np.sqrt(9/4 * (fd[1] ** 2 + fd[2] ** 2) + 1/4 * (fd[3] ** 2 + fd[0] ** 2)) for fd
#                                 in fluorine_fwhm_uncertainties]
#
# asymmetry_fwhm = np.array(asymmetry_fwhm)
# asymmetry_fwhm_uncertainties = np.array(asymmetry_fwhm_uncertainties)
#
# print(np.array(asymmetry) - asymmetry_fwhm)
#
# fluorine_frequencies = np.array([17.7895, 17.7835, 17.7818, 17.7885, 17.7752])
# B_array = np.array([446, 446, 446, 447, 447]) * 1e-3
#
# gamma = 2 * np.pi * fluorine_frequencies / B_array
#

gamma_uncertainties = get_gamma_uncertainties(frequencies, B_array, frequency_uncertainties, B_uncertainties)
# print(gamma_uncertainties)

# odr_fit_with_plot(gamma, asymmetry_fwhm, fluorine_gamma_uncertainties, asymmetry_fwhm_uncertainties)
odr_fit_with_plot(gamma, asymmetry, gamma_uncertainties, asymmetry_uncertainties)
