import numpy as np
from matplotlib import pyplot as plt
from tools import averager, get_csv_data, show_basic_csv_plot, run_parabolic_interpolation


# Ranges from 14 to 22 for hydrogen
file_index = 22
file_indices = np.arange(14, 23, 1)

print(file_indices)


# Get the data
_x, _ch1, _ch2 = get_csv_data(14)
a, d_a = run_parabolic_interpolation(_ch1, [128, 435, 740, 1048])
print(a, d_a)

# Show the plot
show_basic_csv_plot(file_index)


# Datapoints
hydrogen_dips = np.array([[128, 435.5, 740, 1048], [128, 435.5, 740, 1048], [128, 435.5, 740, 1048],
                          [49, 358, 662, 970], [41, 348, 653.5, 961], [28, 334, 641.5, 947],
                          [43.5, 350, 656, 963.5], [67, 371.5, 679.5, 984], [122, 431, 735, 1044]])


# Computes the differences between the datapoints array (measurement) wise
# The first entry is the difference between the last and can be thought of as a wrap around. Here
# it still has to be determined if that is a sound approach.
def get_difference_array(dips_array: np.array([np.array, ...])):

    difference_array = []

    for arr in dips_array:
        # diff_arr = [1200 + arr[0] - arr[-1]] + [arr[i + 1] - arr[i] for i in range(len(arr) - 1)]
        diff_arr = [arr[i + 1] - arr[i] for i in range(len(arr) - 1)]
        difference_array.append(diff_arr)

    return np.array(difference_array)


diff_arrs = get_difference_array(hydrogen_dips)

plt.figure(figsize=(12, 5))
plt.title('Distance in ')
plt.xlabel('Dip indices used for distance measurement')
plt.ylabel('Distance in [resolution units]')

for i, d_a in enumerate(diff_arrs):
    # Scatter plot for data
    plt.scatter(np.arange(len(d_a)), d_a, label=f'{file_indices[i]}')
    plt.plot(np.arange(len(d_a)), d_a, ls='--', lw=0.75)

    # Perform linear regression on the last three data points
    x_last_three = np.arange(len(d_a))[-3:]
    y_last_three = d_a[-3:]

    # Linear fit (degree 1)
    p, cov = np.polyfit(x_last_three, y_last_three, 1, cov=True)
    slope, intercept = p
    slope_err, intercept_err = np.sqrt(np.diag(cov))

    # Generate x values for plotting the fit line
    x_fit = np.linspace(x_last_three[0], x_last_three[-1], 100)
    y_fit = slope * x_fit + intercept

    # Calculate 1-sigma confidence bands
    y_fit_upper = (slope + slope_err) * x_fit + (intercept + intercept_err)
    y_fit_lower = (slope - slope_err) * x_fit + (intercept - intercept_err)

    # Plot the regression line
    # plt.plot(x_fit, y_fit, label=f'Fit {file_indices[i]}')

    # Plot the 1-sigma confidence band
    # plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color='gray', alpha=0.3, label=f'1-sigma band {file_indices[i]}')

plt.legend()
plt.xlim(-0.25, 2.25)
plt.xticks([0, 1, 2], [r'$1 \rightarrow 2$', r'$2 \rightarrow 3$', r'$3 \rightarrow 4$'])
plt.tight_layout()
plt.show()

# Next step:
# - Use the difference data to do a fit that captures the difference between the dot in the
# middle to the one in the center in respect to the frequency. This should give us the necessary
# information to determine a best value for the resonance frequency along with uncertainties.
# Note: Also take the (very slightly) variable magnetic field into account and probably also use an
# ODR for the final assessment (resonance frequency).

