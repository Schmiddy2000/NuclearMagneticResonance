# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

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
    plt.xlim(x_min_extended + x_min, x_max_extended + x_min)
    # plt.xlim(271.375, 273.5)
    # plt.ylim(-15, 15)
    plt.savefig('gyromagnetic_moment_hydrogen_fit_zoomed.png', dpi=200)

    # Show the plot
    plt.show()

    return None


# File indices range from 28 to 34

file_index = 34
teflon_indices = np.arange(28, 35)

# show_basic_csv_plot(file_index)

#
teflon_fwhm_left = np.array([[255, 548.75, 866.9], [232.55, 528.84, 837], [204.62, 507.5, 806.75],
                             [172.64, 478.84, 784.88], [164.38, 472.75, 776.84], [156.88, 465, 770.8],
                             [194.8, 512.9, 802.95]])
teflon_fwhm_right = np.array([[276, 590.25, 892.2], [244, 560.15, 858.25], [220.34, 531.5, 830.25],
                              [180.6, 486.15, 792.27], [174.15, 478.4, 784.2], [162.5, 468.4, 776],
                              [238.27, 538.6, 848.34]])

teflon_fwhm_dips = (teflon_fwhm_left + teflon_fwhm_right) / 2

teflon_B = np.array([446, 446, 446, 447, 447, 447, 447]) * 1e-3
teflon_frequencies = np.array([17.7727, 17.7766, 17.7813, 17.7961, 17.8023, 17.8126, 17.7777])

teflon_fwhm_dips = np.delete(teflon_fwhm_dips, 2, axis=0)
teflon_B = np.delete(teflon_B, 2)
teflon_frequencies = np.delete(teflon_frequencies, 2)

print(teflon_fwhm_dips)

teflon_B_uncertainties = np.zeros(len(teflon_B)) * 1e-3 / np.sqrt(3)
teflon_frequency_uncertainties = np.array([2, 1, 3, 2, 1, 2]) * 1e-4 / np.sqrt(3)

teflon_fwhm_asymmetries = np.diff(np.diff(teflon_fwhm_dips)).flatten()
teflon_fwhm_asymmetry_uncertainties = np.zeros(len(teflon_fwhm_asymmetries)) + np.sqrt(0.204)

print('t', teflon_fwhm_asymmetries)

gamma = 2 * np.pi * teflon_frequencies / teflon_B


def get_gamma_uncertainties(frequencies, B, delta_frequencies, delta_B):
    root_term = delta_frequencies ** 2 + (frequencies * delta_B / B) ** 2

    return 2 * np.pi / B * np.sqrt(root_term)


gamma_uncertainties = get_gamma_uncertainties(teflon_frequencies, teflon_B, teflon_frequency_uncertainties,
                                              teflon_B_uncertainties)

odr_fit_with_plot(gamma, teflon_fwhm_asymmetries, gamma_uncertainties, teflon_fwhm_asymmetry_uncertainties)

# plt.figure(figsize=(12, 5))
#
# plt.scatter(gamma, teflon_fwhm_asymmetries)
#
# plt.show()

