# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

from data_preparation_tools import *

from tools import show_basic_csv_plot


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
    old_x_min_extended = min(x_intercept_best, x_intercept_upper, x_intercept_lower)
    old_x_max_extended = max(x_intercept_best, x_intercept_upper, x_intercept_lower)

    x_min_extended = -0.75
    x_max_extended = 2

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

    plt.axvline(x_intercept_best + x_min, -10, 10, color='k', ls='--', lw=1, label='Best value')
    # plt.fill_betweenx([-10, 30], old_x_min_extended + x_min, old_x_max_extended + x_min, color='k', alpha=0.1,
    #                   label=r'Best value 1-$\sigma$ confidence band')

    # Scatter plot of data points with x shifted back
    plt.errorbar(x_shifted + x_min, y, xerr=x_err, yerr=y_err, fmt='o', label='Data points', capsize=4, elinewidth=0.75)

    # Best fit line, shifted back to the original x-values
    plt.plot(x_fit_extended + x_min, linear_model(popt, x_fit_extended), 'r-', label='Best fit')

    # 1-sigma confidence band (extend lower line to y=0)
    plt.fill_between(x_fit_extended + x_min, linear_model([popt[0] - perr[0], popt[1] - perr[1]], x_fit_extended),
                     linear_model([popt[0] + perr[0], popt[1] + perr[1]], x_fit_extended), color='red', alpha=0.2,
                     label=r'1-$\sigma$ confidence band')

    # Labels and legend
    plt.xlabel(r'Gyromagnetic moment $\gamma$ in [rad$\cdot$T$^{-1}\cdot$MHz]', fontsize=13)
    plt.ylabel('Asymmetry $a$', fontsize=13)
    plt.title('ODR Linear Fit of Hydrogen using FWHM', fontsize=16)
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    # plt.xlim(x_min_extended + x_min, x_max_extended + x_min)
    plt.xlim(271, 273.5)
    # plt.ylim(-7.5, 17.5)
    plt.savefig('gyromagnetic_moment_hydrogen_fwhm.png', dpi=200)

    # Show the plot
    plt.show()

    return None


# File indices range from 5 to 13

file_index = 22
hydrogen_fwhm_indices = np.arange(5, 14)

# show_basic_csv_plot(file_index)

#
hydrogen_fwhm_fwhm_left = np.array([
    [124.90, 432.00, 736.80],
    [124.91, 432.05, 736.81],
    [124.89, 432.05, 736.78],
    [46.67, 356.24, 660.17],
    [38.72, 346.41, 652.17],
    [26.92, 332.50, 640.40],
    [42.26, 348.55, 654.45],
    [64.41, 370.01, 676.95],
    [118.86, 428.24, 732.20]
])
hydrogen_fwhm_fwhm_right = np.array([
    [180.41, 436.52, 742.07],
    [130.40, 436.52, 742.06],
    [130.40, 436.52, 742.07],
    [50.20, 358.46, 662.39],
    [42.21, 348.47, 654.32],
    [28.80, 334.34, 642.29],
    [44.35, 350.94, 656.69],
    [68.17, 372.50, 680.58],
    [124.14, 432.35, 736.31]
])

base_position_errors = np.array([[1, 1, 1] for _ in range(len(hydrogen_fwhm_indices))])

hydrogen_fwhm_fwhm_dips = get_fwhm_positions(hydrogen_fwhm_fwhm_left, hydrogen_fwhm_fwhm_right)
hydrogen_fwhm_dip_errors = get_fwhm_position_errors(base_position_errors, base_position_errors)

hydrogen_fwhm_B = np.array([447, 447, 447, 448, 448, 448, 446, 446, 446]) * 1e-3
hydrogen_fwhm_frequencies = np.array([
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

hydrogen_fwhm_B_uncertainties = np.ones(len(hydrogen_fwhm_B)) * 1e-3 / np.sqrt(3)
hydrogen_fwhm_frequency_uncertainties = np.ones(len(hydrogen_fwhm_indices)) * 1e-4 / np.sqrt(3)

hydrogen_fwhm_fwhm_asymmetries = get_asymmetry(hydrogen_fwhm_fwhm_dips)
hydrogen_fwhm_fwhm_asymmetries *= -1
hydrogen_fwhm_fwhm_asymmetry_uncertainties = get_asymmetry_errors(hydrogen_fwhm_dip_errors)

gamma = get_gamma(hydrogen_fwhm_frequencies, hydrogen_fwhm_B)

gamma_uncertainties = get_gamma_uncertainties(hydrogen_fwhm_frequencies, hydrogen_fwhm_B, hydrogen_fwhm_frequency_uncertainties,
                                              hydrogen_fwhm_B_uncertainties)

print(gamma_uncertainties)

odr_fit_with_plot(gamma, hydrogen_fwhm_fwhm_asymmetries, gamma_uncertainties, hydrogen_fwhm_fwhm_asymmetry_uncertainties)