# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

from tools import show_basic_csv_plot, get_csv_data, run_parabolic_interpolation


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

    # Extend the x-axis range to cover the lower intercept as well
    x_fit_extended = np.linspace(x_min_extended, x_max_extended, 100)

    return_dict = {
        'parameters': popt,
        'uncertainties': perr,
        'x_intercept_best': x_intercept_best + x_min,  # Shifted back to original x
        'x_intercept_upper': x_intercept_upper + x_min,  # Shifted back to original x
        'x_intercept_lower': x_intercept_lower + x_min,  # Shifted back to original x
        'delta_min': x_intercept_best - x_intercept_upper,
        'delta_max':  x_intercept_lower - x_intercept_best
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
    plt.xlabel(r'Gyromagnetic moment $\gamma$ in [rad$\cdot$T$^{-1}\cdot$MHz]')
    plt.ylabel('Asymmetry $a$')
    plt.title('ODR Linear Fit with Confidence Band')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.xlim(x_min_extended + x_min, x_max_extended + x_min)
    plt.savefig('gyromagnetic_moment_fluorine.png', dpi=200)

    # Show the plot
    plt.show()


# File indices range from 23 to 27
file_index = 26
file_indices = np.arange(23, 28)

# show_basic_csv_plot(file_index)

# The last dip is problematic (delta of 3)
# The first dip is problematic (delta of 5)
# The second and last dip id problematic (delta of 2)
# [274, 577.5, 886, 1189]
fluorine_dips = np.array([[239, 549, 851, 1161.5], [273, 577.5, 887.5, 1190], [251, 562.5, 863.5, 1174],
                          [209.5, 514.5, 823, 1126], [275.5, 584, 891, 1195.5]])


# Second measurement, second left is questionable
# Third measurement, third left is questionable
# Fourth measurement, third left
fluorine_fwhm_left = np.array([[236.7, 544.75, 848.8, 1158.73], [270.7, 575, 882.7, 1186.85],
                               [245, 558.77, 857, 1170.5], [206.62, 510.85, 819, 1124.6],
                               [269, 580.4, 886.78, 1192.65]])
fluorine_fwhm_right = np.array([[240.5, 550.43, 852.53, 1164.24], [276.4, 580.3, 890.34, 1192.35],
                                [254, 566, 866.27, 1176.53], [212.1, 516.35, 824.4, 1128.18],
                                [282, 586.55, 894.3, 1198.72]])

fluorine_fwhm_left_uncertainties = np.array([[1, 1, 1, 1], [1, 3, 1, 1], [1, 1, 3, 1], [1, 1, 3, 1], [1, 1, 1, 1]])
fluorine_fwhm_right_uncertainties = np.array([[1, 1, 1, 1], [1, 3, 1, 1], [1, 1, 3, 1], [1, 1, 3, 1], [1, 1, 1, 1]])

fluorine_fwhm_uncertainties = np.array([np.sqrt(u_l ** 2 + u_r ** 2) / np.sqrt(3) for u_l, u_r
                                        in zip(fluorine_fwhm_left_uncertainties, fluorine_fwhm_right_uncertainties)])

frequency_uncertainties = np.array([3, 3, 3, 2, 2]) * 1e-4
B_uncertainties = np.array([1, 1, 1, 1, 1]) * 1e-3 / np.sqrt(3)


# Still need to calculate the real uncertainties here
def get_gamma_uncertainties(frequencies, B, delta_frequencies, delta_B):
    root_term = delta_frequencies ** 2 + (frequencies * delta_B / B) ** 2

    return 2 * np.pi / B * np.sqrt(root_term)


# fluorine_gamma_uncertainties = np.array([0.05 for _ in range(len(fluorine_fwhm_uncertainties))])

fluorine_fwhm_dips = (fluorine_fwhm_left + fluorine_fwhm_right) / 2


asymmetry_fwhm = [3/2 * (fd[1] - fd[2]) + 1/2 * (fd[3] - fd[0]) for fd in fluorine_fwhm_dips]
asymmetry = [3/2 * (fd[1] - fd[2]) + 1/2 * (fd[3] - fd[0]) for fd in fluorine_dips]

asymmetry_fwhm_uncertainties = [np.sqrt(9/4 * (fd[1] ** 2 + fd[2] ** 2) + 1/4 * (fd[3] ** 2 + fd[0] ** 2)) for fd
                                in fluorine_fwhm_uncertainties]

asymmetry_fwhm_uncertainties = np.zeros(len(asymmetry_fwhm)) + 0.204


asymmetry_fwhm = np.array(asymmetry_fwhm)
asymmetry_fwhm_uncertainties = np.array(asymmetry_fwhm_uncertainties)

print(np.array(asymmetry) - asymmetry_fwhm)

fluorine_frequencies = np.array([17.7895, 17.7835, 17.7818, 17.7885, 17.7752])
B_array = np.array([446, 446, 446, 447, 447]) * 1e-3

gamma = 2 * np.pi * fluorine_frequencies / B_array

gamma_uncertainties = get_gamma_uncertainties(fluorine_frequencies, B_array, frequency_uncertainties, B_uncertainties)
# print(gamma_uncertainties)

# odr_fit_with_plot(gamma, asymmetry_fwhm, fluorine_gamma_uncertainties, asymmetry_fwhm_uncertainties)
results = odr_fit_with_plot(gamma, asymmetry_fwhm, gamma_uncertainties, asymmetry_fwhm_uncertainties)

print(results)
