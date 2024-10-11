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
    x_max_extended = 1.75

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
    plt.fill_betweenx([-10, 30], old_x_min_extended + x_min, old_x_max_extended + x_min, color='k', alpha=0.1,
                      label=r'Best value 1-$\sigma$ confidence band')

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
    plt.title('ODR Linear Fit of Glycol using FWHM', fontsize=16)
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    # plt.xlim(x_min_extended + x_min, x_max_extended + x_min)
    plt.xlim(271, 273.4)
    plt.ylim(-7.5, 17.5)
    plt.savefig('gyromagnetic_moment_glycol.png', dpi=200)

    # Show the plot
    plt.show()

    return None


# File indices range from 5 to 13

file_index = 22
glycol_indices = np.arange(5, 14)

# show_basic_csv_plot(file_index)

#
glycol_fwhm_left = np.array([[122.5, 432.7, 734.7], [122.5, 432.7, 734.7], [104.78, 414.26, 716.95],
                             [87, 398.4, 700], [50.47, 356.68, 664.14], [38.95, 346.6, 652.25], [34.20, 340.45, 646.59],
                             [24.70, 330.78, 638.35], [20.68, 328.41, 634.40]])
glycol_fwhm_right = np.array([[126.63, 435.94, 740.07], [126.64, 435.93, 740.07], [108.74, 418.4, 720.87],
                              [90.7, 402.46, 704.06], [54.26, 360.32, 666.49], [42.4, 350.33, 654.49],
                              [36.33, 342.54, 648.86], [26.89, 334.00, 640.48], [22.60, 330.55, 636.46]])

base_position_errors = np.array([[1, 1, 1] for _ in range(len(glycol_indices))])

glycol_fwhm_dips = get_fwhm_positions(glycol_fwhm_left, glycol_fwhm_right)
glycol_dip_errors = get_fwhm_position_errors(base_position_errors, base_position_errors)

glycol_B = np.array([448, 449, 448, 448, 448, 448, 448, 448, 448]) * 1e-3
glycol_frequencies = np.array([19.4568, 19.4447, 19.4400, 19.4166, 19.4043, 19.3982, 19.3894, 19.3802,
                               19.3749])

glycol_B_uncertainties = np.ones(len(glycol_B)) * 1e-3 / np.sqrt(3)
glycol_frequency_uncertainties = np.ones(len(glycol_indices)) * 1e-4 / np.sqrt(3)

glycol_fwhm_asymmetries = get_asymmetry(glycol_fwhm_dips)
# glycol_fwhm_asymmetries *= -1
glycol_fwhm_asymmetry_uncertainties = get_asymmetry_errors(glycol_dip_errors)

gamma = get_gamma(glycol_frequencies, glycol_B)

gamma_uncertainties = get_gamma_uncertainties(glycol_frequencies, glycol_B, glycol_frequency_uncertainties,
                                              glycol_B_uncertainties)

print(gamma_uncertainties)

odr_fit_with_plot(gamma, glycol_fwhm_asymmetries, gamma_uncertainties, glycol_fwhm_asymmetry_uncertainties)
