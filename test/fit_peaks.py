import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf


# Gaussian function with skewness
def skew_gaussian(x, amp, mu, sigma, skew):
    t = (x - mu) / sigma
    return amp * np.exp(-0.5 * t ** 2) * (1 + np.sign(skew) * erf(skew * t / np.sqrt(2)))


# Function to fit a skewed Gaussian to a section of voltage data
def fit_skew_gaussian(voltage_array, start_idx, stop_idx, print_results=False, plot_results=False):
    # Create the x array for the voltage indices
    x_data = np.arange(len(voltage_array))

    # Select the section of data to fit
    x_fit = x_data[start_idx:stop_idx]
    y_fit = voltage_array[start_idx:stop_idx]

    # print(f'start_idx: {start_idx}, stop_idx: {stop_idx}')
    # print(f'len(x_fit): {len(x_fit)}, len(y_fit): {len(y_fit)}')
    #
    # plt.plot(x_fit, y_fit)
    # plt.show()

    # Initial guess for the fit parameters (amplitude, mu, sigma, skew)
    initial_guess = [np.max(y_fit), np.mean(x_fit), np.std(x_fit), 0]

    # Fit the skewed Gaussian function to the selected data
    popt, pcov = curve_fit(skew_gaussian, x_fit, y_fit, p0=initial_guess)

    # Extract the fitted parameters and uncertainties from covariance matrix
    amp, mu, sigma, skew = popt
    amp_err, mu_err, sigma_err, skew_err = np.sqrt(np.diag(pcov))

    if print_results:
        # Print the results with uncertainties (4 digits)
        print(f"Amplitude (A): {amp:.4f} ± {amp_err:.4f}")
        print(f"Mean (mu): {mu:.4f} ± {mu_err:.4f}")
        print(f"Sigma: {sigma:.4f} ± {sigma_err:.4f}")
        print(f"Skewness: {skew:.4f} ± {skew_err:.4f}")

    if plot_results:
        # Create a dense x range for plotting the fit and confidence band
        x_dense = np.linspace(start_idx, stop_idx, 1000)
        y_fit_dense = skew_gaussian(x_dense, *popt)

        # Calculate 1-sigma confidence intervals
        sigma_confidence = np.sqrt(np.diag(pcov))
        lower_bound = skew_gaussian(x_dense, *(popt - sigma_confidence))
        upper_bound = skew_gaussian(x_dense, *(popt + sigma_confidence))

        # Plot the full voltage data
        plt.plot(x_data, voltage_array, 'b-', label="Voltage Data", alpha=0.7)

        # Plot the fit over the selected section
        plt.plot(x_dense, y_fit_dense, 'r-', label="Skew Gaussian Fit")

        # Plot the 1-sigma confidence band
        plt.fill_between(x_dense, lower_bound, upper_bound, color='gray', alpha=0.3, label="1-sigma band")

        # Plot a vertical line at the mean (mu)
        plt.axvline(mu, color='g', linestyle='--', label=f"mu = {mu:.4f}")

        # Labels and legend
        plt.xlabel("Index")
        plt.ylabel("Voltage")
        plt.legend()

        # Show plot
        plt.show()

# Example usage:
# voltages = np.random.normal(0, 1, 100)  # Example voltage array
# fit_skew_gaussian(voltages, 10, 80, print_results=True, plot_results=True)

